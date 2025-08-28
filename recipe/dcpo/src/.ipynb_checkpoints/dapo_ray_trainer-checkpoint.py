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
from pprint import pprint
from copy import deepcopy
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import torch
import os
import pdb
import pickle
import traceback
from verl.utils.debug.track_train_data import (
    track_batch,
    save_batch,
    change_defaultdict_to_dict,
    change_dict_to_defaultdict,
)
import random
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    AdvantageEstimator,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


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


def replace_gen_batch_with_old_sampling(drop_batch: DataProto, fill_with_old_sampling_map, sample_n, metric_name, replace_values=[]):
    if isinstance(replace_values, str):
        try:
            replace_values = eval(replace_values)
        except:
            tmp = set()
            if "True" in replace_values:
                tmp.add(True)
            if "False" in replace_values:
                tmp.add(False)
            replace_values = tmp
    elif isinstance(replace_values, bool):
        replace_values = {replace_values}

    # data_index = set(drop_batch.non_tensor_batch["uid"])
    print(f"replace_gen_batch_with_old_sampling, {len(drop_batch)=}, {replace_values=} starting....")
    replace_new_values_count = defaultdict(int)
    data_index2ids = defaultdict(list)
    for ids, index in enumerate(drop_batch.non_tensor_batch["index"]):
        data_index2ids[index].append(ids)
    replace_gen_batch = []
    for index, ids_list in data_index2ids.items():
        # 不存在旧样
        if index not in fill_with_old_sampling_map:
            continue
        value = drop_batch.non_tensor_batch[metric_name][ids_list[0]]
        uid = drop_batch.non_tensor_batch["uid"][ids_list[0]]
        old_values = fill_with_old_sampling_map[index][metric_name]
        other_values = (set(old_values) - {value}) & set(replace_values)
        #
        # 旧样本中没有其他值
        if len(other_values) == 0:
            continue
        # 统计std=0的替换情况
        replace_new_values_count[value] += 1

        print(f"new value: {value}, old_value: {[f'{k}: {len(v)}' for k, v in old_values.items()]} , other_value: {set(other_values)}")
        # 替换的gen
        other_old_gen_sampling = []
        for old_v, samplings in old_values.items():
            #  只替换其他值
            if old_v not in other_values:
                continue
            other_old_gen_sampling.append(samplings)
        other_old_gen_sampling = DataProto.concat(other_old_gen_sampling)

        # 最多可替换的
        replace_n = min(len(other_old_gen_sampling), sample_n, len(ids_list) - 1)

        # 保持不变的gen
        keep_gen_ids = random.sample(ids_list, len(ids_list) - replace_n)
        if keep_gen_ids:
            replace_gen_batch.append(drop_batch[keep_gen_ids])

        old_gen_idx = random.sample(range(len(other_old_gen_sampling)), replace_n)
        if old_gen_idx:
            old_gen_batch = other_old_gen_sampling[old_gen_idx]
            old_gen_batch.non_tensor_batch["uid"] = np.array([uid] * len(old_gen_idx))
            replace_gen_batch.append(old_gen_batch)

    print(f"replace value: from {[f'{v}: {n}' for v,n in replace_new_values_count.items()]}")
    if replace_gen_batch:
        replace_gen_batch = DataProto.concat(replace_gen_batch)
        print(f"use old gen to replace for : {len(set(replace_gen_batch.non_tensor_batch['uid']))}")
        return replace_gen_batch, replace_new_values_count
    else:
        return None, replace_new_values_count


def add_old_sampling_to_N(new_batch: DataProto, fill_with_old_sampling_map, sample_n, metric_name, replace_values=[]):
    if not new_batch:
        return new_batch
    if isinstance(replace_values, str):
        try:
            replace_values = eval(replace_values)
        except:
            tmp = set()
            if "True" in replace_values:
                tmp.add(True)
            if "False" in replace_values:
                tmp.add(False)
            replace_values = tmp
    elif isinstance(replace_values, bool):
        replace_values = {replace_values}
    # data_index = set(drop_batch.non_tensor_batch["uid"])
    print(f"add_old_sampling_to_N, {sample_n=}, {replace_values=} starting....")
    replace_new_values_count = defaultdict(int)
    data_index2value_ids = defaultdict(lambda: defaultdict(list))
    nums = 0
    for ids, (index, value) in enumerate(zip(new_batch.non_tensor_batch["index"], new_batch.non_tensor_batch[metric_name])):
        data_index2value_ids[index][value].append(ids)

    replace_gen_batch = []
    for index, value_ids in data_index2value_ids.items():
        # 不存在旧样本
        if index not in fill_with_old_sampling_map:
            for value, ids_list in value_ids.items():
                replace_gen_batch.append(new_batch[ids_list])
            continue
        # uid = new_batch.non_tensor_batch["uid"][value_ids[True][0]]
        uid = new_batch[value_ids[list(value_ids.keys())[0]]].non_tensor_batch["uid"][0]
        new_values_ids = []
        other_values_ids = []
        old_value_data = []
        for value, ids_list in value_ids.items():
            if value not in replace_values:
                other_values_ids.extend(ids_list)
            else:
                new_values_ids.extend(ids_list)
                if value in fill_with_old_sampling_map[index][metric_name]:
                    old_value_data.append(fill_with_old_sampling_map[index][metric_name][value])

        if len(old_value_data) >= 1:
            old_value_data = DataProto.concat(old_value_data)

        if len(new_values_ids) >= sample_n or len(old_value_data) == 0:
            for value, ids_list in value_ids.items():
                replace_gen_batch.append(new_batch[ids_list])
            continue

        replace_n = min(sample_n - len(new_values_ids), sample_n, len(other_values_ids) - 1, len(old_value_data))

        print(f"replace to N: {len(new_values_ids)=}, {len(old_value_data)=}, {len(other_values_ids)=}, {replace_n=}, {sample_n=}")
        if replace_n <= 0:
            for value, ids_list in value_ids.items():
                replace_gen_batch.append(new_batch[ids_list])
            continue
        nums += replace_n
        keep_ids = random.sample(other_values_ids, len(other_values_ids) - replace_n)
        if keep_ids:
            replace_gen_batch.append(new_batch[keep_ids])
        replace_gen_batch.append(new_batch[new_values_ids])

        new_replace_data = old_value_data[random.sample(range(len(old_value_data)), replace_n)]
        new_replace_data.non_tensor_batch["uid"] = np.array([uid] * len(new_replace_data))
        replace_gen_batch.append(new_replace_data)

    # pdb.set_trace()
    replace_gen_batch = DataProto.concat(replace_gen_batch)
    print(f"add_old_sampling_to_N: {len(new_batch)=}, {len(replace_gen_batch)=}, {nums=}")
    return replace_gen_batch


def update_fill_with_old_sampling_map(new_batch: DataProto, fill_with_old_sampling_map, sample_n, metric_name):
    print(f"update_fill_with_old_sampling_map starting, {len(fill_with_old_sampling_map)}")
    gen_index2metric_ids = defaultdict(lambda: defaultdict(list))
    for i, (index, metric_value) in enumerate(zip(new_batch.non_tensor_batch["index"], new_batch.non_tensor_batch[metric_name])):
        gen_index2metric_ids[index][metric_value].append(i)
    for index, metric_value_ids in gen_index2metric_ids.items():
        for metric_value, ids in metric_value_ids.items():
            n = min(len(ids), sample_n)
            have_n = len(fill_with_old_sampling_map[index][metric_name][metric_value])
            sample_ids = random.sample(ids, n)
            sample_gen_batch = new_batch[sample_ids]
            if n == sample_n or have_n == 0:
                fill_with_old_sampling_map[index][metric_name][metric_value] = sample_gen_batch
            elif n + have_n <= sample_n:
                fill_with_old_sampling_map[index][metric_name][metric_value] = DataProto.concat([fill_with_old_sampling_map[index][metric_name][metric_value], sample_gen_batch])
            else:
                keep_n = sample_n - n
                # keep_ids = random.sample(range(have_n), keep_n)
                fill_with_old_sampling_map[index][metric_name][metric_value] = DataProto.concat([fill_with_old_sampling_map[index][metric_name][metric_value][-keep_n:], sample_gen_batch])
    print(f"update_fill_with_old_sampling_map finished, {len(fill_with_old_sampling_map)}")


class RayDAPOTrainer(RayPPOTrainer):
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
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        if self.config.actor_rollout_ref.rollout.fill_with_old_sampling:
            fill_with_old_sampling_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            try:
                if os.path.exists(os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt")):
                    with open(os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt"), "rb") as f:
                        fill_with_old_sampling_map = change_dict_to_defaultdict(pickle.load(f))
                    print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt")} loaded success!!!')
            except:
                traceback.format_exc()
                print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt")} loaded failed!!!')
        new_batch = None
        metric_value_count = defaultdict(int)
        start_steps = float("inf")
        if self.config.actor_rollout_ref.rollout.fill_with_old_sampling.enable:
            start_steps = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.start_steps

        for epoch in range(self.config.trainer.total_epochs):
            epoch_replace_new_values_count = defaultdict(int)
            epoch_std0_values_count = defaultdict(int)
            epoch_std_not0_values_count = defaultdict(int)
            use_train_n = 0
            epoch_new_gen_std_not0_prompt_n = 0
            epoch_new_gen_std_not0_use_prompt_n = 0

            if epoch != 0:
                print(f"{epoch=}, {self.global_steps=}, {use_train_n=}, {use_train_n/len(self.train_dataloader)/self.config.data.train_batch_size=}")
                if self.config.track_data_path is not None:
                    os.makedirs(self.config.track_data_path, exist_ok=True)
                    try:
                        with open(os.path.join(self.config.track_data_path, f"full_old_sampling_epoch_{epoch}.pt"), "wb") as f:
                            pickle.dump(change_defaultdict_to_dict(fill_with_old_sampling_map), f)
                        print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_epoch_{epoch}.pt")} save success!!!!')
                    except:
                        traceback.format_exc()
                        print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_epoch_{epoch}.pt")} save failed!!!!')

            metric_value_count = defaultdict(int)
            metrics = {"epoch": epoch}

            for rollout_step, batch_dict in enumerate(self.train_dataloader):

                new_gen_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                if new_batch:
                    new_batch = DataProto.concat([new_batch, new_gen_batch])
                else:
                    new_batch = new_gen_batch
                if self.config.algorithm.filter_groups.enable:
                    if start_steps >= self.global_steps or epoch == 0:
                        train_batch_size = self.config.data.train_batch_size
                        if len(new_batch) // train_batch_size < 3:
                            continue
                    else:
                        if len(new_batch) // self.config.data.train_batch_size < 1:
                            continue

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

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    print(f"{epoch=}, generate a batch ing {len(gen_batch)=}, {self.global_steps=}, {num_prompt_in_batch=}")
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
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
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)
                    with _timer("reward", timing_raw):
                        self.update_reward(new_batch, metrics)

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                            #
                            metric_value_count[str(metric_val)] += 1

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]

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
                        metrics[f"stdgt0/step/prompt"] = len(kept_prompt_uids) + metrics.get(f"std>0/step/promptN", 0)

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
                        # 添加旧 old
                        replace_gen_batch = None
                        if self.config.actor_rollout_ref.rollout.fill_with_old_sampling.enable:
                            # 保持历史条数
                            save_n = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.save_n
                            # 开始步数
                            start_steps = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.start_steps
                            # less_n = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.less_n
                            std0_n = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.std0_n
                            stdgt0_n = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.stdgt0_n
                            replace_values = self.config.actor_rollout_ref.rollout.fill_with_old_sampling.replace_values
                            print(f"global_steps: {self.global_steps}, stdgt0_n: {stdgt0_n}, std0_n: {std0_n}, save_n: {save_n}, start_steps: {start_steps}, replace_values: {replace_values}")

                            if save_n > 0 and self.global_steps > start_steps:
                                # add old_sampling
                                if stdgt0_n > 0:
                                    use_batch = add_old_sampling_to_N(
                                        use_batch,
                                        fill_with_old_sampling_map,
                                        stdgt0_n,
                                        metric_name,
                                        replace_values,
                                    )
                                    self.update_reward(use_batch, metrics)

                                # std=0替换
                                if std0_n > 0:
                                    replace_gen_batch, replace_new_values_count = replace_gen_batch_with_old_sampling(
                                        drop_batch,
                                        fill_with_old_sampling_map,
                                        std0_n,
                                        metric_name,
                                        replace_values,
                                    )
                                    # 统计std=0时，用旧response 替补数据
                                    for v, n in replace_new_values_count.items():
                                        epoch_replace_new_values_count[v] += n
                                        metrics[f"std=0/step/replace/{v}"] = n + metrics.get(f"std=0/step/replace/{v}", 0)
                                        metrics[f"std=0/epoch/replace/{v}"] = epoch_replace_new_values_count[v]

                            update_fill_with_old_sampling_map(
                                new_batch,
                                fill_with_old_sampling_map,
                                save_n,
                                metric_name,
                            )
                        # 有更新 则拼接
                        if replace_gen_batch:
                            self.update_reward(replace_gen_batch, metrics)
                            print(f"std>0 new gen prompt size: {len(set(use_batch.non_tensor_batch['uid']))}, old replace prompt size: {len(set(replace_gen_batch.non_tensor_batch['uid']))}")
                            new_batch = DataProto.concat([use_batch, replace_gen_batch])
                        else:
                            print(f"std>0 new gen prompt size: {len(set(use_batch.non_tensor_batch['uid']))}, old replace prompt size: 0")
                            new_batch = use_batch

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

                    use_train_n += len(set(batch.non_tensor_batch["uid"]))
                    # print_keys_and_shape(batch, name="end: batch")
                    print_keys_and_shape(batch, name="end: batch")
                    new_batch = None
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        print(f"compute advantage, {self.global_steps=}, {self.config.algorithm.adv_estimator=}")
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )
                        print(f"compute advantage finished,{len(batch)=}")

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # pdb.set_trace()
                        print(f"update actor, {self.global_steps=}, {self.config.trainer.critic_warmup=}, {len(batch)=}")
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        if self.config.track_data_path is not None and self.config.actor_rollout_ref.rollout.fill_with_old_sampling.enable:
                            os.makedirs(self.config.track_data_path, exist_ok=True)
                            try:
                                with open(
                                    os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt"),
                                    "wb",
                                ) as f:
                                    pickle.dump(change_defaultdict_to_dict(fill_with_old_sampling_map), f)
                                print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt")} save sucess!!!!')
                            except:
                                traceback.format_exc()
                                print(f'{os.path.join(self.config.track_data_path, f"full_old_sampling_{self.global_steps}.pt")} save failed!!!!')

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
                    save_batch(
                        batch,
                        f"{self.config.track_data_path}/{epoch}_batch.jsonl",
                        self.tokenizer,
                        epoch=epoch,
                        step=self.global_steps,
                        model_path=self.config.actor_rollout_ref.model.path,
                    )

                metrics["train/num_gen_batches"] = num_gen_batches
                logger.log(data=metrics, step=self.global_steps)

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
