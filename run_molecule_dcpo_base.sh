#!/usr/bin/env bash
# ============================================================================
# DCPO Base Training Script for Molecule Generation
# This script contains the actual training command
# ============================================================================
set -euxo pipefail

# ============================================================================
# NCCL Configuration (for multi-GPU training)
# ============================================================================
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export TORCH_DISTRIBUTED_DEBUG=INFO

# Clean up any stale Ray processes (but don't kill all python processes)
ray stop --force 2>/dev/null || true

# ============================================================================
# vLLM Configuration
# ============================================================================
export VLLM_CONFIG_ROOT=${CHECKPOINT_SAVE}
export VLLM_CACHE_ROOT=${CHECKPOINT_SAVE}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Network Interface (adjust according to your cluster)
# ============================================================================
nccl=eth0  # Change to your network interface (e.g., eth0, bond0, ib0)

export NCCL_SOCKET_IFNAME=$nccl
export GLOO_SOCKET_IFNAME=$nccl
export TP_SOCKET_IFNAME=$nccl
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8

# ============================================================================
# Ray Configuration
# ============================================================================
export RAY_MASTER_PORT=6379
declare -A ray_start_params=(
  ["metrics-export-port"]=20100
  ["runtime-env-agent-port"]=20101
  ["dashboard-agent-grpc-port"]=20102
  ["dashboard-agent-listen-port"]=20103
)

export HYDRA_FULL_ERROR=1

ray_start_args=()
for key in "${!ray_start_params[@]}"; do
  ray_start_args+=("--$key" "${ray_start_params[$key]}")
done

# ============================================================================
# Model-specific Configuration
# ============================================================================
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-256}

# Performance parameters
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=True
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-768}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-768}
offload=True
gen_tp=1  # Tensor parallel size for generation (1 for single GPU)

# ============================================================================
# Training Configuration
# ============================================================================
project_name=${PROJECT_NAME:-'molecule_generation'}
exp_name=${EXP_NAME:-'dcpo_egfr'}

adv_estimator=${ADV_ESTIMATOR:-"dcpo"}

# KL configuration
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS_ENABLE:-False}
kl_loss_coef=0.001

# Clipping configuration
clip_ratio_low=0.2
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_type=${CLIP_TYPE:-"dynamic"}

# Overlong buffer (disabled for short SMILES)
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_penalty_factor=1.0
overlong_buffer_version=${OVERLONG_BUFFER_VERSION:-"dcpo"}
overlong_buffer_error_penalty=${OVERLONG_BUFFER_ERROR_PENALTY:-0.0}

# Loss aggregation
loss_agg_mode=${LOSS_AGG_MODE:-"only-token-mean"}

# Filter groups
enable_filter_groups=${ENABLE_FILTER_GROUPS:-False}
filter_groups_metric=acc
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}

# Batch sizes
if [ ${WORLD_SIZE} -eq 1 ]; then
  train_prompt_bsz=${TRAIN_PROMPT_BSZ:-130}
else
  train_prompt_bsz=${TRAIN_PROMPT_BSZ:-130}
fi

gen_prompt_bsz=${GEN_PROMPT_BSZ:-130}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
ppo_mini_batch_size_per_gpu=$n_resp_per_prompt
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

# ============================================================================
# Directory Configuration
# ============================================================================
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
WORKING_DIR="${PWD}"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/root/autodl-tmp/LlaSMol-EGFR-Final-exp3-chat-template-fast"}
CKPTS_DIR=${CKPTS_DIR:-"./ckpts/molecule_dcpo"}
TRAIN_FILE=${TRAIN_FILE:-"./data/molecule_generation_train.parquet"}
TEST_FILE=${TEST_FILE:-"./data/molecule_generation_val.parquet"}

# ============================================================================
# Generation Configuration
# ============================================================================
temperature=1.0
top_p=0.95
top_k=-1

# ============================================================================
# Wait for all workers
# ============================================================================
echo "Waiting for all workers to be ready..."
while true; do
  touch ${CHECKPOINT_SAVE}/_worker_${RANK}_ready
  count=0
  
  for ((x = 0; x < $WORLD_SIZE; x++)); do
    if [[ -f "${CHECKPOINT_SAVE}/_worker_${x}_ready" ]]; then
      count=$((count + 1))
    fi
  done
  
  echo "Progress: ${count}/${WORLD_SIZE} workers ready"
  
  if [[ $count -eq ${WORLD_SIZE} ]]; then
    echo "All workers ready!"
    break
  else
    sleep 5s
  fi
done

# ============================================================================
# Start Training
# ============================================================================
echo "Starting Ray cluster and training job..."
which python
set -x

if [ ${RANK} == 0 ]; then
  # Start Ray head node
  ray start --head --port=${RAY_MASTER_PORT} --dashboard-host 0.0.0.0 ${ray_start_args[@]}

  # Submit training job
  ray job submit --address="http://127.0.0.1:8265" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -m recipe.dcpo.src.main_dcpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.test_n=${TEST_N:-0} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO:-True} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C:-10.0} \
    actor_rollout_ref.actor.clip_type=${clip_type} \
    actor_rollout_ref.actor.ppo_mini_batch_size_per_gpu=${ppo_mini_batch_size_per_gpu:-0} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE:-"dcpo"} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dcpo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${max_response_length} \
    reward_model.overlong_buffer.version=${overlong_buffer_version} \
    reward_model.overlong_buffer.error_penalty=${overlong_buffer_error_penalty} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    track_data_path=${CHECKPOINT_SAVE}/train_sample \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.total_epochs=${TOTAL_EPOCHS:-100} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${RESUME_MODE:-auto} 2>&1 | tee ${CHECKPOINT_SAVE}/run.log
    
  exit_code=${PIPESTATUS[0]}
  ray status
  ray stop --force
  exit ${exit_code}
else
  sleep 15s
  ray start --address ${MASTER_ADDR}:${RAY_MASTER_PORT} --num-gpus 8 --block ${ray_start_args[@]}
fi

