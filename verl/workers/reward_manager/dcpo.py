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

from verl import DataProto
from verl.utils.reward_score import default_compute_score
import torch

from collections import defaultdict
import numpy as np
import random

from multiprocessing import Pool
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.workers.reward_manager import register


@register("dcpo")
class DCPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        config=None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.frist_print = True
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def get_length_reward_dapo(self, valid_response_length):
        overlong_buffer_len = self.overlong_buffer_cfg.len
        expected_len = self.max_resp_len - overlong_buffer_len
        exceed_len = valid_response_length - expected_len
        overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        return overlong_reward

    def get_reward_for_one(self, data_item):
        prompt_ids = data_item.batch["prompts"]

        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        if self.frist_print:
            self.frist_print = False
            print(f"{data_source=}")
            print(f"{response_str=}")
            print(f"{ground_truth=}")
            print(f"{extra_info=}")
        max_try = 3
        result = {}
        while max_try > 0:
            max_try -= 1
            try:
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                if isinstance(result, (dict, float, int)):
                    return result
            except:
                traceback.format_exc()
                pass
        print(f"{result=}")
        return result

    def get_reward(self, data):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_dict = {executor.submit(self.get_reward_for_one, data[i]): i for i in range(len(data))}
            for future in as_completed(future_dict):
                idx = future_dict[future]
                try:
                    result = future.result()
                    result_dict[idx] = result
                except:
                    traceback.format_exc()
                    continue
        print(f"{len(result_dict)=}, {len(future_dict)=}")
        return result_dict

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        uid2response_length = defaultdict(list)
        for i in range(len(data)):
            uid = data.non_tensor_batch["uid"][i]
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            uid2response_length[uid].append(valid_response_length)

        id2response_length_profix = defaultdict(list)
        for uid, response_lengths in uid2response_length.items():
            mean = np.mean(response_lengths)
            std = np.std(response_lengths)
            median = np.median(response_lengths)
            low, high = np.percentile(response_lengths, (10, 90), interpolation="midpoint")
            id2response_length_profix[uid] = {
                "mean": mean,
                "std": std,
                "median": median,
                "low": low,
                "high": high,
            }

        already_print_data_sources = {}
        is_overlongs = [False] * len(data)
        is_right_list = [False] * len(data)

        print(f"{len(data)=}, start reward compute score...")
        reward_result = self.get_reward(data)
        print(f"{reward_result[0]=}, {type(reward_result[0])=}")
        print("finished score")

        uid_acc_index = defaultdict(lambda: defaultdict(list))

        for i in range(len(data)):
            uid = data.non_tensor_batch["uid"][i]
            index = data.non_tensor_batch["index"][i]
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            reward_extra_info["uid"].append(uid)
            reward_extra_info["index"].append(index)
            reward_extra_info["prompt_str"].append(prompt_str)
            reward_extra_info["prompt_len"].append(f"{prompt_length}")
            reward_extra_info["response_len"].append(f"{valid_response_length}")
            reward_extra_info["response_str"].append(f"{response_str}")

            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            reward_extra_info["data_source"].append(data_source)

            result = reward_result[i]
            score: float
            if isinstance(result, dict):
                score = result.get("score", -1)
                # Store the information including original reward
                for key, value in result.items():
                    if key == "right_brace_idx" and value == None:
                        value = 0
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score
            is_right = False
            if isinstance(result, dict) and result.get("acc", False):
                is_right = True
            elif score > 0:
                is_right = True
            is_right_list[i] = is_right
            uid_acc_index[uid][is_right].append(i)

            if self.overlong_buffer_cfg.version == "dcpo":
                reward -= score
                if result.get("acc", False):
                    reward += 1
                elif result.get("pred", "[INVALID]") == "[INVALID]":
                    reward += -1
                else:
                    reward += 0
            elif self.overlong_buffer_cfg.enable:
                # dapo
                if self.overlong_buffer_cfg.version == "dapo":  # dcpo
                    length_reward = self.get_length_reward_dapo(valid_prompt_length)
                    reward += length_reward

            reward_tensor[i, valid_response_length - 1] = reward
            reward_extra_info["orign_reward"].append(f"{reward}")
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        try:
            if "length_reward" in reward_extra_info:
                print(f"length_reward: {random.choice(reward_extra_info['length_reward'])=}")
            if "langeuage_reward_mean" in reward_extra_info:
                print(f"langeuage_reward_mean: {random.choice(reward_extra_info['langeuage_reward_mean'])=}")

            print(f"reward: {len(data)=}, {reward_tensor.size()=}, {len(is_overlongs)=}")
            strs = "dcpo reward_extra_info/len: "
            for k, v in reward_extra_info.items():
                strs += f"{k}_len: {len(v)},"
            print("reward finished !!!")
        except:
            pass
        if return_dict:
            returns = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
            if self.overlong_buffer_cfg.enable:
                returns["is_overlongs"] = is_overlongs
            return returns
        else:
            return reward_tensor
