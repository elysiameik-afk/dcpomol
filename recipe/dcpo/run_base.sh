#!/usr/bin/env bash
set -euxo pipefail
# 配置环境变量 A800使用
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export TORCH_DISTRIBUTED_DEBUG=INFO
# vllm缓存目录
export VLLM_CONFIG_ROOT=${CHECKPOINT_SAVE}
export VLLM_CACHE_ROOT=${CHECKPOINT_SAVE}

# Python环境
# export PYTHONPATH="$(pwd):$PYTHONPATH"
# ray环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
# nccl=bond0
# h20
nccl=eth0

export NCCL_SOCKET_IFNAME=$nccl
export GLOO_SOCKET_IFNAME=$nccl
export TP_SOCKET_IFNAME=$nccl
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8

export RAY_MASTER_PORT=6379
declare -A ray_start_params=(
  ["metrics-export-port"]=20100
  ["runtime-env-agent-port"]=20101
  ["dashboard-agent-grpc-port"]=20102
  ["dashboard-agent-listen-port"]=20103
)
# Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
export HYDRA_FULL_ERROR=1
ray_start_args=()
for key in "${!ray_start_params[@]}"; do
  ray_start_args+=("--$key" "${ray_start_params[$key]}")
done
less_model="Qwen2.5-Math-7B|7b|Qwen2.5-Math-1.5B-Instruct"
if [[ "$MODEL_PATH" =~ $less_model ]]; then
  max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
  max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 2))}
  overlong_buffer_len=${OVERLONG_BUFFER_LEN:-512}

  # Performance Related Parameter
  sp_size=${SP_SIZE:-1}
  #  AssertionError: num_attention_heads 28 must be divisible by ulysses_sp_size 8
  use_dynamic_bsz=True
  actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
  infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
  offload=True
  gen_tp=4

else
  max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
  max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 18))}
  overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
  # Performance Related Parameter
  sp_size=${SP_SIZE:-4}
  #  AssertionError: num_attention_heads 28 must be divisible by ulysses_sp_size 8
  use_dynamic_bsz=True
  actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-5120}
  infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-5120}
  offload=True
  gen_tp=4
fi

project_name=${PROJECT_NAME:-'DCPO'}
exp_name=${EXP_NAME:-'DCPO-Qwen2.5-7B'}

adv_estimator=${ADV_ESTIMATOR:-"dcpo"}

# 是否使用std标准化
if [[ "$adv_estimator" =~ 'grpo|dapo' ]]; then
  NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-True}
fi


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS_ENABLE:-False}
kl_loss_coef=0.001

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_type=${CLIP_TYPE:-"origin"} # dual dynamic origin

enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_penalty_factor=1.0
overlong_buffer_version=${OVERLONG_BUFFER_VERSION:-"v0"}
overlong_buffer_error_penalty=${OVERLONG_BUFFER_ERROR_PENALTY:-0.0}

loss_agg_mode=${LOSS_AGG_MODE:-"only-token-mean"}

enable_filter_groups=${ENABLE_FILTER_GROUPS:-True}
filter_groups_metric=acc
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
if [ ${WORLD_SIZE} -eq 1 ]; then
  train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
else
  train_prompt_bsz=${TRAIN_PROMPT_BSZ:-512}
fi

gen_prompt_bsz=${GEN_PROMPT_BSZ:-$(train_prompt_bsz)}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
ppo_mini_batch_size_per_gpu=$n_resp_per_prompt
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

####################
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
WORKING_DIR=$(dirname $(dirname $SCRIPT_DIR))

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# --no-wait

# 等待所有worker就绪
echo "-----------------work-------------"
while true; do
  touch ${CHECKPOINT_SAVE}/_worker_${RANK}_ready
  count=0
  # 遍历所有worker编号
  for ((x = 0; x < $WORLD_SIZE; x++)); do
    # 检测对应worker的就绪文件
    if [[ -f "${CHECKPOINT_SAVE}/_worker_${x}_ready" ]]; then
      echo $count
      count=$((count + 1)) # 存在则计数器+1
    fi
  done
  echo "Progress: ${count}/${WORLD_SIZE} workers ready"

  # 判断是否全部就绪
  if [[ $count -eq ${WORLD_SIZE} ]]; then
    echo "全部就绪"
    break # 全部就绪继续执行
  else
    sleep 5s # 等待5秒后再次检查
  fi
done

echo "-----------------job-------------"
which python
set -x

if [ ${RANK} == 0 ]; then
  # 启动ray集群head节点
  ray start --head --port=${RAY_MASTER_PORT} --dashboard-host 0.0.0.0 ${ray_start_args[@]}

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
    algorithm.filter_groups.start_steps.enable=${FILTER_GROUPS_START_STEPS_ENABLE:-False} \
    algorithm.filter_groups.start_steps.value=${FILTER_GROUPS_START_STEPS_VALUE:-0} \
    algorithm.filter_groups.start_steps.low_nums=${FILTER_GROUPS_START_STEPS_LOW_NUMS:-0} \
    algorithm.filter_groups.start_steps.high_nums=${FILTER_GROUPS_START_STEPS_HIGH_NUMS:-$n_resp_per_prompt} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE:-"vanilla"} \
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
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.version=${overlong_buffer_version} \
    reward_model.overlong_buffer.error_penalty=${overlong_buffer_error_penalty} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.use_old_mean.enable=${USE_OLD_MEAN_ENABLE:-False} \
    reward_model.use_old_mean.coeff=${USE_OLD_MEAN_COEFF:-1} \
    track_data_path=${CHECKPOINT_SAVE}/train_sample \
    tensorboard_log_dir="${CHECKPOINT_SAVE}/runs" \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes="${NNODES}" \
    trainer.balance_batch=${BALANCE_BATCH:-True} \
    trainer.batch_shuffle_type=${BATCH_SHUFFLE_TYPE:-False} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.save_freq=10 \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${RESUME_MODE:-auto} 2>&1 | tee ${CHECKPOINT_SAVE}/run.log
  exit_code=${PIPESTATUS[0]}
  ray status
  # 停止ray集群
  ray stop --force
  exit ${exit_code}
else
  sleep 15s
  # 启动ray集群
  ray start --address ${MASTER_ADDR}:${RAY_MASTER_PORT} --num-gpus 8 --block ${ray_start_args[@]}
fi
