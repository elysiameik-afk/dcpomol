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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
# from .dapo_ray_trainer import RayDAPOTrainer

import os
import ray
import hydra
import pdb
import json
from omegaconf import OmegaConf, DictConfig


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path="config", config_name="dcpo_trainer", version_base=None)
def main(config):
    # pdb.set_trace()
    config_dict = OmegaConf.to_container(config)
    with open("/global_data/med/yangsh/data/RL/config/merge.json", "w") as f:
        f.write(json.dumps(config_dict, ensure_ascii=False))


#     from hydra.core.global_hydra import GlobalHydra

#     # 清理现有 Hydra 实例
#     GlobalHydra.instance().clear()
#     main_old()


# @hydra.main(config_path="config", config_name="dcpo_trainer_old", version_base=None)
# def main_old(config):
#     config_dict = OmegaConf.to_container(config)
#     with open("/global_data/med/yangsh/data/RL/config/old.json", "w") as f:
#         f.write(json.dumps(config_dict, ensure_ascii=False))


if __name__ == "__main__":
    main()
