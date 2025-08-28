# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict, Counter
from pprint import pprint
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import os
import pdb
import pickle
import traceback
import json
import time

from verl.utils.debug.track_train_data import (
    track_batch,
    save_non_tensor,
    change_defaultdict_to_dict,
    change_dict_to_defaultdict,
)
import random
from verl import DataProto

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


def print_keys_and_shape(data: DataProto, name=""):
    if not data or len(data) == 0:
        print("data is empty")
        return
    strs = f"{name}: "
    strs += f"{data.batch.keys()=}"
    for k in data.batch.keys():
        strs += f"{k} shape: {data.batch[k].shape}"
    strs += f"{data.non_tensor_batch.keys()=}"
    for k in data.non_tensor_batch.keys():
        strs += f"{k} shape:  {np.array(data.non_tensor_batch[k]).shape}"
    print(strs)


class RayDCPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def update_reward(self, new_batch, metrics):

        # compute scores. Support both model and function-based.
        # We first compute the scores using reward model. Then, we call reward_fn to combine
        # the results from reward model and rule-based results.
        if self.use_rm:
            # we first compute reward model score
            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
            new_batch = new_batch.union(reward_tensor)

        # we combine with rule-based rm
        print("start reward function ing ...")
        reward_extra_infos_dict: dict[str, list]
        try:
            reward_result = self.reward_fn(new_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result["reward_extra_info"]
        except Exception as e:
            print(f"Error in reward_fn: {e}")
            reward_tensor = self.reward_fn(new_batch)
            reward_extra_infos_dict = {}

        new_batch.batch["token_level_scores"] = reward_tensor

        print(f"{list(reward_extra_infos_dict.keys())=}")
        if reward_extra_infos_dict:
            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        # compute rewards. apply_kl_penalty if available
        if self.config.algorithm.use_kl_in_reward:
            new_batch, kl_metrics = apply_kl_penalty(
                new_batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
        else:
            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.global_gen_steps = 0
        self.epoch = 0
        self.old_mean_std = defaultdict(dict)
        self.gen_steps = 0
        # load checkpoint before doing anything
        self._load_checkpoint()
        old_mean_file = os.path.join(self.config.track_data_path, f"old_mean_std_global_steps_{self.global_steps}.pt")
        if os.path.exists(old_mean_file):
            print(f"load from {old_mean_file}")
            try:
                with open(old_mean_file, "rb") as f:
                    old_mean_std = pickle.load(f)
                    for k, v in old_mean_std.items():
                        for kv, vv in v.items():
                            self.old_mean_std[k][kv] = float(vv)
            except:
                pass
            print(f"{len(self.old_mean_std)=}")
        else:
            print(f"{old_mean_file} not exist")
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log_all(data=val_metrics, back_step=self.global_steps, gen_step=self.global_gen_steps, epoch=self.epoch)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        std_ratio = 0.5

        new_batch = None
        metric_value_count = defaultdict(int)
        start_steps = float("inf")

        print("*****************start update actor******************")
        for epoch in range(self.config.trainer.total_epochs):
            epoch_std0_values_count = defaultdict(int)
            epoch_std_not0_values_count = defaultdict(int)
            use_train_n = 0
            epoch_new_gen_std_not0_prompt_n = 0
            epoch_new_gen_std_not0_use_prompt_n = 0
            self.epoch += 1
            metric_value_count = defaultdict(int)

            for _, batch_dict in enumerate(self.train_dataloader):
                metrics = {"epoch": epoch}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(not prev_step_profile and curr_step_profile if self.config.trainer.profile_continuous_steps else curr_step_profile)
                new_gen_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                if new_batch:
                    new_batch = DataProto.concat([new_batch, new_gen_batch])
                else:
                    new_batch = new_gen_batch
                if self.config.algorithm.filter_groups.enable:
                    if start_steps >= self.global_steps or epoch == 0:
                        train_batch_size = self.config.data.train_batch_size
                        need_gen_prompt_size = train_batch_size - num_prompt_in_batch
                        try_gen_prompt_size = min(need_gen_prompt_size / max(std_ratio, 0.01), train_batch_size * 4)
                        if try_gen_prompt_size % len(new_gen_batch) / len(new_gen_batch) >= 0.8:
                            try_gen_prompt_size += len(new_gen_batch)
                        if len(new_batch) < try_gen_prompt_size:
                            continue
                        print(f"{epoch=}, {self.global_steps=}, {train_batch_size=}, {need_gen_prompt_size=}, {std_ratio=}, {try_gen_prompt_size=}, {len(new_batch)=}")

                    else:
                        if len(new_batch) // self.config.data.train_batch_size < 1:
                            continue
                self.global_gen_steps += int(len(new_batch) // self.config.data.train_batch_size)
                if self.config.reward_model.use_old_mean.enable:
                    for i, idx in enumerate(new_batch.non_tensor_batch["index"]):
                        if idx in self.old_mean_std:
                            new_batch.non_tensor_batch["mean"][i] = self.old_mean_std[idx]["mean"]
                            new_batch.non_tensor_batch["std"][i] = self.old_mean_std[idx]["std"]
                            new_batch.non_tensor_batch["times"][i] = self.old_mean_std[idx]["times"]
                        else:
                            self.old_mean_std[idx]["mean"] = new_batch.non_tensor_batch["mean"][i]
                            self.old_mean_std[idx]["std"] = new_batch.non_tensor_batch["std"][i]
                            self.old_mean_std[idx]["times"] = new_batch.non_tensor_batch["times"][i]

                # pop those keys for generation

                print_keys_and_shape(new_batch, "start: new_batch")
                if "multi_modal_inputs" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                print_keys_and_shape(gen_batch, "gen_batch")
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    print(f"{epoch=}, generate a batch ing {len(gen_batch)=}, {self.global_steps=}, {num_prompt_in_batch=}")
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                        dtype=object,
                    )
                    new_batch.non_tensor_batch["epoch"] = np.array([epoch] * len(new_batch.batch), dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)
                    with marked_timer("reward", timing_raw, "yellow"):
                        self.update_reward(new_batch, metrics)

                    if self.config.track_data_path != "":
                        os.makedirs(self.config.track_data_path, exist_ok=True)
                        save_non_tensor(new_batch, f"{self.config.track_data_path}/infer_non_tensor_epoch_{epoch}.jsonl")

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                            metric_value_count[str(metric_val)] += 1

                        prompt_uid2metric_std = {}
                        prompt_uid2metrci_sum = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            prompt_uid2metrci_sum[prompt_uid] = np.sum(metric_vals)
                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        # 大于0的std占比
                        std_ratio = len(kept_prompt_uids) / max(len(prompt_uid2metric_std), 1)
                        
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        use_batch = new_batch[kept_traj_idxs]

                        if "acc" in use_batch.non_tensor_batch:
                            batch_acc = use_batch.non_tensor_batch["acc"]
                            batch_acc_counter = Counter(batch_acc)
                            # 统计每个epoch中的std>0情况 / rollout比例值
                            for v, n in batch_acc_counter.items():
                                n = n / self.config.actor_rollout_ref.rollout.n
                                epoch_std_not0_values_count[v] += n
                                metrics[f"stdgt0/step/all/{v}"] = n + metrics.get(f"stdgt0/step/all/{v}", 0)
                                metrics[f"stdgt0/epoch/all/{v}"] = epoch_std_not0_values_count[v]
                        # 统计生成新的std>0的情况
                        epoch_new_gen_std_not0_prompt_n += len(kept_prompt_uids)
                        metrics[f"stdgt0/epoch/promptN"] = epoch_new_gen_std_not0_prompt_n
                        metrics[f"stdgt0/step/promptN"] = len(kept_prompt_uids) + metrics.get(f"std>0/step/promptN", 0)

                        #  统计生成新的std>0的被用在训练中的数量
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if batch:
                            _n = len(set(use_batch[: traj_bsz - len(batch)].non_tensor_batch["uid"]))
                        else:
                            _n = len(set(use_batch[:traj_bsz].non_tensor_batch["uid"]))
                        epoch_new_gen_std_not0_use_prompt_n += _n
                        metrics[f"stdgt0/epoch/train_new_promptN"] = epoch_new_gen_std_not0_use_prompt_n
                        metrics[f"stdgt0/step/train_new_promptN"] = _n + metrics.get(f"stdgt0/step/train_new_promptN", 0)

                        dropped_traj_idxs = [idx for idx in range(len(new_batch.non_tensor_batch["uid"])) if idx not in kept_traj_idxs]

                        drop_batch = new_batch[dropped_traj_idxs]

                        if "acc" in drop_batch.non_tensor_batch:
                            batch_acc = drop_batch.non_tensor_batch["acc"]
                            batch_acc_counter = Counter(batch_acc)
                            # 统计每个epoch中的std=0情况 acc
                            for v, n in batch_acc_counter.items():
                                n = n / self.config.actor_rollout_ref.rollout.n
                                epoch_std0_values_count[v] += n
                                metrics[f"std=0/step/all/{v}"] = n + metrics.get(f"std=0/step/all/{v}", 0)
                                metrics[f"std=0/epoch/all/{v}"] = epoch_std0_values_count[v]

                        new_batch = use_batch

                        for v, n in Counter(new_batch.non_tensor_batch[metric_name]).most_common():
                            metrics[f"stdgt0/step/keep/{v}"] = n + metrics.get(f"stdgt0/step/keep/{v}", 0)

                        num_prompt_in_batch += len(set(new_batch.non_tensor_batch["uid"]))

                        if batch is None or len(batch) == 0:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        new_batch = None
                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{epoch=}, {self.global_steps=},{num_prompt_in_batch=} < {prompt_bsz=}, {len(batch)=} ")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{epoch=}, {num_prompt_in_batch=},{num_gen_batches=}, {max_num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.")
                        else:
                            # Align the batch
                            print(f"{epoch=}, {self.global_steps=}, {num_prompt_in_batch=}, {num_gen_batches=}, {self.config.data.train_batch_size=}, {len(batch)=}")
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                            uid2idx = defaultdict(list)
                            for ids, uid in enumerate(batch.non_tensor_batch["uid"]):
                                uid2idx[uid].append(ids)
                            shuffle_uid = list(uid2idx.keys())
                            random.shuffle(shuffle_uid)
                            shuffle_ids = []
                            for uid in shuffle_uid:
                                tmp = uid2idx[uid]
                                # print(tmp)
                                random.shuffle(tmp)
                                shuffle_ids.extend(tmp)
                            batch = batch[shuffle_ids]
                            print(f"{len(batch)=}, {type(batch)=}")
                        for v, n in Counter(batch.non_tensor_batch[metric_name]).most_common():
                            metrics[f"stdgt0/step/train/{v}"] = n + metrics.get(f"stdgt0/step/train/{v}", 0)
                    use_train_n += len(set(batch.non_tensor_batch["uid"]))
                    # print_keys_and_shape(batch, name="end: batch")
                    print_keys_and_shape(batch, name="end: batch")
                    new_batch = None
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # batch_shuffle = self._sort_batch(batch, self.config.trainer.batch_shuffle_type)
                    # if batch_shuffle:
                    #     self.config.actor_rollout_ref.actor.add_loss = False
                    # el
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                        # self.actor_rollout_wg.config.actor.config.add_loss = False

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        print(f"compute advantage, {self.global_steps=}, {self.config.algorithm.adv_estimator=}")
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        print(f"compute advantage finished,{len(batch)=}")
                        if self.config.reward_model.use_old_mean.enable:
                            for idx, _index in enumerate(batch.non_tensor_batch["index"]):
                                if "mean" in batch.non_tensor_batch:
                                    mean = batch.non_tensor_batch["mean"][idx]
                                    self.old_mean_std[_index]["mean"] = mean
                                if "std" in batch.non_tensor_batch:
                                    std = batch.non_tensor_batch["std"][idx]
                                    self.old_mean_std[_index]["std"] = std
                                if "times" in batch.non_tensor_batch:
                                    times = batch.non_tensor_batch["times"][idx]
                                    self.old_mean_std[_index]["times"] = times
                            try:
                                save_file = os.path.join(self.config.track_data_path, f"old_mean_std_global_steps_{self.global_steps}.pt")
                                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                                with open(save_file, "wb") as f:
                                    pickle.dump(change_defaultdict_to_dict(self.old_mean_std), f)
                                print(f"save old mean time to: {save_file}")
                            except Exception as e:
                                print(f"Error in writing old_mean_std.json: {e}")

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        print(f"update actor, {self.global_steps=}, {self.config.trainer.critic_warmup=}, {len(batch)=}")
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = self.global_steps + 1 in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                    self._stop_profiling(curr_step_profile and not next_step_profile if self.config.trainer.profile_continuous_steps else curr_step_profile)
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # TODO: make a canonical logger that supports various backend
                if self.config.track_data_path != "":
                    if not os.path.exists(self.config.track_data_path):
                        os.makedirs(self.config.track_data_path)
                    track_batch(
                        batch,
                        f"{self.config.track_data_path}/train_data.jsonl",
                        self.tokenizer,
                        step=self.global_steps,
                    )

                metrics["train/num_gen_batches"] = num_gen_batches
                logger.log_all(data=metrics, back_step=self.global_steps, gen_step=self.global_gen_steps, epoch=self.epoch)

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                new_batch = None
                metrics = {"epoch": epoch}

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
