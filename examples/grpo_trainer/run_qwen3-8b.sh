# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
ray stop --force
ps -ef | grep "python" | awk '{print $2}' | xargs kill -9

##qwen
data_root=/global_data/med/yangsh/data/RL/train_msg/qwen
train_files="$data_root/clean_data.parquet,"
# train_files="$data_root/dapo17k.parquet,"
val_files="$data_root/aime_2024.parquet,$data_root/aime_2025.parquet,$data_root/math500.parquet,$data_root/amc2023.parquet,"

train_files="[${train_files::-1}]"
val_files="[${val_files::-1}]"



export CODE_PATH=/yangshihui/code
export CHECKPOINT_SAVE=${CHECKPOINT_SAVE:-/checkpoint_save}
export CHECKPOINT_LOAD=${CHECKPOINT_LOAD:-/checkpoint_load}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MODEL_PATH=$CHECKPOINT_LOAD/small_models/Qwen3-8B-Base

mkdir -p ${CHECKPOINT_SAVE}
# cp -rf ${CODE_PATH}/verl_qwen3 ${CHECKPOINT_SAVE}/
# cp -rf ${CODE_PATH}/rl-board ${CHECKPOINT_SAVE}/
# cp $0 ${CHECKPOINT_SAVE}/

# cd ${CHECKPOINT_SAVE}/verl_qwen3
cd ${CODE_PATH}/verl_qwen3

project_name=grpo
experiment_name=grpo_8b
wandb_key="70ea7733c68df6088253955f00cbce3d002b4ead"
export WANDB_API_KEY=$wandb_key

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb",'tensorboard']' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.wandb_key=$wandb_key \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@