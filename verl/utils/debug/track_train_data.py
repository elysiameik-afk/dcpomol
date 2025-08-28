from typing import List
import json
from verl.protocol import DataProto
import torch
import numpy as np
import traceback
from collections import defaultdict
import copy
import os


def data2json(batch: DataProto, steps: int, tokenizer) -> List:
    result = []
    for sample in batch:
        data = {}
        attention_mask = sample.batch["attention_mask"]
        prompt = sample.batch["prompts"]
        response = sample.batch["responses"]
        prompt_mask = attention_mask[: prompt.shape[0]]
        response_mask = attention_mask[prompt.shape[0] :]

        data["data_source"] = sample.non_tensor_batch.get("data_source", "")
        data["ground_truth"] = sample.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
        data["prompt"] = tokenizer.decode(prompt[prompt_mask == 1])
        data["response"] = tokenizer.decode(response[response_mask == 1])
        data["response_tokens"] = [tokenizer.decode(token_id) for token_id in response[response_mask == 1]]
        data["logprobs"] = sample.batch["old_log_probs"][response_mask == 1].tolist()
        # data["ref_logprobs"] = sample.batch["ref_log_prob"][response_mask == 1].tolist()
        data["token_rewards"] = sample.batch["advantages"][response_mask == 1].tolist()
        data["reward"] = sample.batch["token_level_rewards"][response_mask == 1][-1].tolist()
        data["advantage"] = sample.batch["advantages"][response_mask == 1][-1].tolist()
        data["step"] = steps

        result.append(data)
    return result


def write_json(data, output_file, mode="a"):
    with open(output_file, mode) as fout:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    return


def track_batch(batch, filename, tokenizer, step=0):
    data = data2json(batch, step, tokenizer)
    for item in data:
        write_json(item, filename)


def turn_to_list(data):
    if data is None or isinstance(data, (str, int, float, bool)):
        return f"{data}"
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = data.tolist()
        return data
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = turn_to_list(v)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = turn_to_list(data[i])
    elif isinstance(data, tuple):
        res = []
        for t in data:
            res.append(turn_to_list(t))
        return res
    return data


def save_non_tensor(batch: DataProto, file):
    non_tensor = []
    for sample in batch:
        str_sample = {}
        for k, v in sample.non_tensor_batch.items():
            str_sample[k] = turn_to_list(v)
        non_tensor.append(str_sample)
    try:
        with open(file, "a") as f:
            for item in non_tensor:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except:
        traceback.format_exc()


def save_batch(batch: DataProto, filename, tokenizer, epoch=-1, step=-1, model_path=None, **kwargs):
    tensor_batch = {}
    for k, v in batch.batch.items():
        tensor_batch[k] = turn_to_list(v)

    no_tensor_batch = {}
    for k, v in batch.non_tensor_batch.items():
        no_tensor_batch[k] = turn_to_list(v)
    data = {
        "tensor_batch": tensor_batch,
        "non_tensor_batch": no_tensor_batch,
        "step": step,
        "epoch": epoch,
        "model_path": model_path,
    }
    data.update(kwargs)
    try:
        write_json(data, filename)
    except Exception as e:
        print("save batch error:")
        traceback.print_exc()


def DataProto_turn2json(batch: DataProto):

    tensor_batch = {}
    for k, v in batch.batch.items():
        tensor_batch[k] = turn_to_list(v)
    no_tensor_batch = {}
    for k, v in batch.non_tensor_batch.items():
        no_tensor_batch[k] = turn_to_list(v)

    return {"tensor_batch": tensor_batch, "no_tensor_batch": no_tensor_batch}


def save_sampling(fill_with_old_sampling_map, file_name, epoch=-1, step=-1, model_path=None, **kwargs):
    js_result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, metric_name_value_list in fill_with_old_sampling_map.items():
        for metric_name, value_list in metric_name_value_list.items():
            js_result["metric_name"] = metric_name
            for metric_value, _DataProto in value_list.items():
                try:
                    js = DataProto_turn2json(_DataProto)
                    js_result[index][metric_name][str(metric_value)] = js
                except:
                    traceback.format_exc()
    js_result.update(kwargs)
    js_result["epoch"] = epoch
    js_result["model_path"] = model_path
    js_result["step"] = step
    os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
    with open(file_name, "w") as f:
        try:
            f.write(json.dumps(js_result, ensure_ascii=False, indent=1))
            f.flush()
        except:
            traceback.format_exc()


# fill_with_old_sampling_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
def change_defaultdict_to_dict(d):
    d = copy.deepcopy(d)
    res = {}
    if isinstance(d, defaultdict):
        for k, v in d.items():
            if isinstance(v, defaultdict):
                res[k] = change_defaultdict_to_dict(v)
            else:
                res[k] = turn_to_list(v)
    else:
        return turn_to_list(d)
    return res


def change_dict_to_defaultdict(d):
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = change_dict_to_defaultdict(v)
            else:
                res[k] = v
    else:
        return d
    return res
