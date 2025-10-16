#!/usr/bin/env bash
# ============================================================================
# DCPO Training Script for Molecule Generation
# ============================================================================
set -x

# Install dependencies for molecule generation
pip install rdkit
pip install wandb

# ============================================================================
# Basic Configuration
# ============================================================================
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export CHECKPOINT_SAVE=${CHECKPOINT_SAVE:-"./ckpts/molecule_dcpo"}
export CHECKPOINT_LOAD=${CHECKPOINT_LOAD:-"/root/autodl-tmp"}

# Create checkpoint directory if it doesn't exist
mkdir -p ${CHECKPOINT_SAVE}

# Clean up previous run
ray stop --force
# Note: Removed "kill all python" command to avoid killing terminal/jupyter/vscode processes

for ((x = 0; x < $WORLD_SIZE; x++)); do
    rm -rf ${CHECKPOINT_SAVE}/_worker_${x}_ready
done

# ============================================================================
# Wandb Configuration
# ============================================================================
# Wandb API key should be already configured in your environment
# If not, set it here or run: wandb login
# export WANDB_API_KEY="your_key_here"

# ============================================================================
# Model and Tokenizer Paths
# ============================================================================
# IMPORTANT: Copy tokenizer with chat_template to model directory
export MODEL_PATH=${CHECKPOINT_LOAD}/LlaSMol-EGFR-Final-exp3
export TOKENIZER_PATH=${CHECKPOINT_LOAD}/LlaSMol-EGFR-Final-exp3-chat-template-fast

# Copy tokenizer files with chat_template to model directory
echo "Copying tokenizer with chat_template to model directory..."
cp -f ${TOKENIZER_PATH}/tokenizer.json ${MODEL_PATH}/ 2>/dev/null || true
cp -f ${TOKENIZER_PATH}/tokenizer_config.json ${MODEL_PATH}/
cp -f ${TOKENIZER_PATH}/chat_template.jinja ${MODEL_PATH}/ 2>/dev/null || true
echo "Tokenizer files copied."

# ============================================================================
# Dataset Configuration
# ============================================================================
data_root=./data
train_files="$data_root/molecule_generation_train.parquet,"
val_files="$data_root/molecule_generation_val.parquet,"

train_files="[${train_files::-1}]"
val_files="[${val_files::-1}]"

# ============================================================================
# Ray Configuration
# ============================================================================
export NNODES=${WORLD_SIZE:-1}

# ============================================================================
# Training Hyperparameters
# ============================================================================
# Batch sizes
export TRAIN_PROMPT_BSZ=130  # Use all 130 training data per batch
export GEN_PROMPT_BSZ=130    # Generate for all prompts at once
export N_RESP_PER_PROMPT=8   # Generate 8 responses per prompt

# Response length (SMILES are much shorter than math solutions)
export MAX_PROMPT_LENGTH=512     # Prompts are shorter for molecule generation
export MAX_RESPONSE_LENGTH=256   # SMILES strings are typically 50-200 characters
export ACTOR_PPO_MAX_TOKEN_LEN=768
export INFER_PPO_MAX_TOKEN_LEN=768

# Sequence parallel (set to 1 for smaller models)
export SP_SIZE=1

# Mini batch size
export TRAIN_PROMPT_MINI_BSZ=$(($WORLD_SIZE * 8))

# Validation
export VAL_BEFORE_TRAIN=True
export TEST_FREQ=5
export SAVE_FREQ=10

# Resume mode
export RESUME_MODE="auto"

# ============================================================================
# DCPO Algorithm Configuration
# ============================================================================
# Filter groups (disable for small dataset)
export ENABLE_FILTER_GROUPS=False

# Advantage estimator
export ADV_ESTIMATOR="dcpo"
export NORM_ADV_BY_STD_IN_GRPO=True

# KL loss (disable)
export USE_KL_LOSS_ENABLE=False

# Dynamic clipping
export CLIP_RATIO_C=10.0
export CLIP_RATIO_HIGH=0.28
export CLIP_TYPE="dynamic"  # dynamic adaptive clipping

# Loss configuration
export LOSS_AGG_MODE="only-token-mean"
export LOSS_MODE="dcpo"

# Overlong buffer (disable for short SMILES)
export ENABLE_OVERLONG_BUFFER=False
export OVERLONG_BUFFER_VERSION="dcpo"
export OVERLONG_BUFFER_ERROR_PENALTY=1.0

# ============================================================================
# Experiment Naming
# ============================================================================
export PROJECT_NAME="molecule_generation"
export EXP_NAME="dcpo_egfr_${WORLD_SIZE}gpu"

# ============================================================================
# Training Loop Configuration
# ============================================================================
export CKPTS_DIR=$CHECKPOINT_SAVE
export TRAIN_FILE=$train_files
export TEST_FILE=$val_files
export TOTAL_EPOCHS=100
export MAX_NUM_GEN_BATCHES=10

# ============================================================================
# Wait for all workers to be ready
# ============================================================================
pwd
echo "Starting training loop..."

while true; do
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
            echo "All workers ready, starting training..."
            break
        else
            sleep 2s
        fi
    done
    
    # Run the actual training script
    bash ./run_molecule_dcpo_base.sh
done

