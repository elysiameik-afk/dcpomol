# Molecule Generation with DCPO

This document provides instructions for training a molecule generation model using DCPO (Dynamic Clipping Policy Optimization).

## 📋 Overview

This setup implements DCPO reinforcement learning for generating EGFR inhibitor molecules with improved drug-like properties. The reward function evaluates generated molecules based on 9 key indicators:

1. **Format**: Correct `<SMILES>...</SMILES>` format
2. **Validity**: Valid SMILES string
3. **QED**: Drug-likeness score
4. **SA Score**: Synthetic accessibility
5. **LogP**: Lipophilicity
6. **MW**: Molecular weight
7. **TPSA**: Polar surface area
8. **Lipinski**: Lipinski's Rule of Five
9. **EGFR-specific**: Structural features typical of EGFR inhibitors

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install RDKit for molecular property calculations
pip install rdkit

# Install other dependencies
pip install tensorboard
```

### 2. Prepare Data

Generate the training and validation datasets:

```bash
python generate_rl_dataset.py
```

This will create:
- `data/molecule_generation_train.parquet` (130 samples)
- `data/molecule_generation_val.parquet` (13 samples)

### 3. Test Reward Function

Before training, verify the reward function works correctly:

```bash
python test_reward_function.py
```

Expected output: Test cases with various SMILES strings and their reward scores.

### 4. Start Training

#### Single GPU Training:

```bash
bash run_molecule_dcpo.sh
```

#### Multi-GPU Training (e.g., 8 GPUs):

```bash
export WORLD_SIZE=8
bash run_molecule_dcpo.sh
```

---

## 📁 File Structure

```
DCPO/
├── data/
│   ├── molecule_generation_train.parquet  # Training data (130 samples)
│   └── molecule_generation_val.parquet     # Validation data (13 samples)
├── verl/utils/reward_score/
│   ├── __init__.py                         # Reward function registry
│   └── molecule_generation.py              # Molecule reward implementation
├── generate_rl_dataset.py                  # Dataset generation script
├── test_reward_function.py                 # Reward function test suite
├── run_molecule_dcpo.sh                    # Main training launcher
├── run_molecule_dcpo_base.sh               # Detailed training configuration
└── MOLECULE_GENERATION_README.md           # This file
```

---

## ⚙️ Configuration

### Key Parameters

Edit `run_molecule_dcpo.sh` to adjust:

```bash
# Model paths
export MODEL_PATH="/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"
export TOKENIZER_PATH="/root/autodl-tmp/LlaSMol-EGFR-Final-exp3-chat-template-fast"

# Batch sizes
export TRAIN_PROMPT_BSZ=130      # Training batch size
export N_RESP_PER_PROMPT=8       # Responses per prompt

# Sequence lengths
export MAX_PROMPT_LENGTH=512     # Max prompt tokens
export MAX_RESPONSE_LENGTH=256   # Max response tokens

# Training epochs
export TOTAL_EPOCHS=100

# Checkpointing
export CHECKPOINT_SAVE="./ckpts/molecule_dcpo"
export SAVE_FREQ=10
export TEST_FREQ=5
```

### DCPO Hyperparameters

```bash
# Advantage estimator
export ADV_ESTIMATOR="dcpo"

# Dynamic clipping
export CLIP_TYPE="dynamic"
export CLIP_RATIO_HIGH=0.28
export CLIP_RATIO_C=10.0

# Loss configuration
export LOSS_MODE="dcpo"
export LOSS_AGG_MODE="only-token-mean"
```

---

## 📊 Monitoring Training

### Tensorboard

```bash
tensorboard --logdir ./ckpts/molecule_dcpo/runs
```

### Log Files

Training logs are saved to:
```
./ckpts/molecule_dcpo/run.log
```

---

## 🧪 Reward Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| -5.0 | Format error (missing or incorrect SMILES tags) |
| -4.0 to -2.0 | Valid format but invalid SMILES |
| 0.0 to 3.0 | Valid molecule with poor drug properties |
| 3.0 to 5.0 | Valid molecule with moderate properties |
| 5.0 to 7.0 | Good drug-like molecule |
| 7.0 to 10.0 | Excellent drug-like molecule |

### Component Scores

- **Format**: +1.0 (correct) / -5.0 (incorrect)
- **Validity**: +0.5 (valid) / -3.0 (invalid)
- **QED**: 0-2.0 (higher is better)
- **SA**: 0-2.0 (lower synthetic accessibility is better)
- **LogP**: 0-1.5 (ideal range: 1-4)
- **MW**: 0-1.0 (ideal range: 200-500)
- **TPSA**: 0-1.0 (ideal range: 40-120)
- **Lipinski**: 0-1.5 (0 violations = 1.5)
- **EGFR**: 0-1.0 (structural features)

---

## 📝 Dataset Format

Each training sample has the following structure:

```json
{
    "data_source": "molecule_generation",
    "prompt": [
        {
            "role": "user",
            "content": "Generate a highly potent EGFR inhibitor for lung cancer"
        }
    ],
    "ability": "molecule_generation",
    "reward_model": {
        "ground_truth": "",
        "style": "molecular_property"
    },
    "extra_info": {
        "index": 0,
        "task": "egfr_molecule_generation",
        "template_id": 1,
        "description_id": 1
    }
}
```

---

## 🔧 Troubleshooting

### Issue: "RDKit not available"

**Solution**: Install RDKit
```bash
pip install rdkit
```

### Issue: "SA_Score not available"

**Solution**: This is optional. The reward function will use a neutral SA score if unavailable.

### Issue: Tokenizer format errors

**Solution**: Ensure you're using the tokenizer with chat_template:
```bash
export TOKENIZER_PATH="/root/autodl-tmp/LlaSMol-EGFR-Final-exp3-chat-template-fast"
```

### Issue: Out of memory during training

**Solution**: Reduce batch size or response length:
```bash
export TRAIN_PROMPT_BSZ=65  # Half of default
export MAX_RESPONSE_LENGTH=128  # Reduce from 256
```

---

## 📈 Expected Training Behavior

1. **Initial Phase** (epochs 1-10):
   - Many format errors (-5.0 scores)
   - Model learning to generate `<SMILES>...</SMILES>` format
   - Average reward: -3.0 to 0.0

2. **Learning Phase** (epochs 10-30):
   - Fewer format errors
   - More valid SMILES strings
   - Average reward: 0.0 to 3.0

3. **Optimization Phase** (epochs 30-100):
   - Mostly valid molecules
   - Improving drug-like properties
   - Average reward: 3.0 to 7.0

4. **Convergence** (epochs 100+):
   - Stable generation of drug-like molecules
   - Average reward: 5.0 to 8.0

---

## 🎯 Next Steps

After training:

1. **Generate molecules** from the trained model
2. **Evaluate diversity** of generated molecules
3. **Docking studies** for top candidates
4. **Experimental validation** of promising compounds

---

## 📚 References

- **DCPO Paper**: [Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333)
- **VERL Framework**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
- **RDKit**: [https://www.rdkit.org/](https://www.rdkit.org/)

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `./ckpts/molecule_dcpo/run.log` for error messages
3. Test the reward function with `test_reward_function.py`

---

**Happy Molecule Generation!** 🧪

