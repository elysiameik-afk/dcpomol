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
# from verl.experimental.dataset.sampler import AbstractSampler
# from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
# from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
# from verl.trainer.ppo.reward import load_reward_manager
# from verl.utils.device import is_cuda_available
# from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    pdb.set_trace()
    
if __name__ == "__main__":
    main()
