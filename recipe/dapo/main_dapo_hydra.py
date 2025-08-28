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

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
import pdb
# from verl.trainer.ppo.reward import load_reward_manager
# from verl.utils.device import is_cuda_available

# from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    pdb.set_trace()
    run_ppo(config)


if __name__ == "__main__":
    main()
