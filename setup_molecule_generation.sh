#!/usr/bin/env bash
# ============================================================================
# Quick Setup Script for Molecule Generation with DCPO
# ============================================================================

set -e

echo "============================================================================"
echo "DCPO Molecule Generation Setup"
echo "============================================================================"
echo

# ============================================================================
# Step 1: Check Python version
# ============================================================================
echo "Step 1: Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "  ✅ Python version OK (>= 3.8)"
else
    echo "  ❌ Python version must be >= 3.8"
    exit 1
fi

# ============================================================================
# Step 2: Install dependencies
# ============================================================================
echo
echo "Step 2: Installing dependencies..."

echo "  Installing RDKit (for molecular property calculations)..."
pip install rdkit -q

echo "  Installing TensorBoard (for training monitoring)..."
pip install tensorboard -q

echo "  ✅ Dependencies installed"

# ============================================================================
# Step 3: Verify RDKit installation
# ============================================================================
echo
echo "Step 3: Verifying RDKit installation..."

if python -c "from rdkit import Chem; from rdkit.Chem import QED; print('RDKit OK')" 2>/dev/null; then
    echo "  ✅ RDKit working correctly"
else
    echo "  ❌ RDKit installation failed"
    echo "  Please manually install: conda install -c conda-forge rdkit"
    exit 1
fi

# ============================================================================
# Step 4: Create necessary directories
# ============================================================================
echo
echo "Step 4: Creating directories..."

mkdir -p data
mkdir -p ckpts/molecule_dcpo
mkdir -p ckpts/molecule_dcpo/runs
mkdir -p ckpts/molecule_dcpo/train_sample

echo "  ✅ Directories created"

# ============================================================================
# Step 5: Check tokenizer
# ============================================================================
echo
echo "Step 5: Checking tokenizer setup..."

TOKENIZER_PATH="/root/autodl-tmp/LlaSMol-EGFR-Final-exp3-chat-template-fast"

if [ -d "$TOKENIZER_PATH" ]; then
    echo "  ✅ Tokenizer found: $TOKENIZER_PATH"
else
    echo "  ⚠️  Tokenizer not found at: $TOKENIZER_PATH"
    echo "  Please run: python add_chat_template.py"
fi

# ============================================================================
# Step 6: Generate dataset
# ============================================================================
echo
echo "Step 6: Checking dataset..."

if [ -f "data/molecule_generation_train.parquet" ] && [ -f "data/molecule_generation_val.parquet" ]; then
    echo "  ✅ Dataset already exists"
    echo "    - data/molecule_generation_train.parquet"
    echo "    - data/molecule_generation_val.parquet"
else
    echo "  📊 Generating dataset..."
    python generate_rl_dataset.py
    echo "  ✅ Dataset generated"
fi

# ============================================================================
# Step 7: Test reward function
# ============================================================================
echo
echo "Step 7: Testing reward function..."

if python test_reward_function.py > /tmp/reward_test.log 2>&1; then
    echo "  ✅ Reward function test passed"
    echo "    (Full output: /tmp/reward_test.log)"
else
    echo "  ❌ Reward function test failed"
    echo "    Check /tmp/reward_test.log for details"
    exit 1
fi

# ============================================================================
# Step 8: Make scripts executable
# ============================================================================
echo
echo "Step 8: Making scripts executable..."

chmod +x run_molecule_dcpo.sh
chmod +x run_molecule_dcpo_base.sh
chmod +x generate_rl_dataset.py
chmod +x test_reward_function.py

echo "  ✅ Scripts are executable"

# ============================================================================
# Summary
# ============================================================================
echo
echo "============================================================================"
echo "✅ Setup Complete!"
echo "============================================================================"
echo
echo "Next steps:"
echo "  1. Start training:"
echo "     bash run_molecule_dcpo.sh"
echo
echo "  2. Monitor training:"
echo "     tensorboard --logdir ./ckpts/molecule_dcpo/runs"
echo
echo "  3. View logs:"
echo "     tail -f ./ckpts/molecule_dcpo/run.log"
echo
echo "Configuration:"
echo "  - Training data: data/molecule_generation_train.parquet (130 samples)"
echo "  - Validation data: data/molecule_generation_val.parquet (13 samples)"
echo "  - Checkpoint dir: ./ckpts/molecule_dcpo"
echo "  - Model path: /root/autodl-tmp/LlaSMol-EGFR-Final-exp3"
echo "  - Tokenizer path: /root/autodl-tmp/LlaSMol-EGFR-Final-exp3-chat-template-fast"
echo
echo "For more information, see: MOLECULE_GENERATION_README.md"
echo "============================================================================"

