#!/bin/bash
# RunPod Setup Script for RetailSim Next-Basket Training
#
# Prerequisites:
# 1. Create a RunPod instance with GPU (A100/H100 recommended)
# 2. Create a persistent volume and mount at /workspace
# 3. Upload data files to the volume:
#    - /workspace/data/prepared/*.parquet
#    - /workspace/data/tensor_cache/*.npy
#    - /workspace/raw_data/transactions.csv

set -e

echo "=== RetailSim Training Setup ==="

# Clone repository
cd /workspace
if [ ! -d "retail_sim" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/retail_sim.git
else
    echo "Updating repository..."
    cd retail_sim && git pull && cd ..
fi

cd retail_sim

# Install Poetry
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    pip install poetry
fi

# Install dependencies
echo "Installing dependencies..."
poetry install --no-interaction

# Create symlinks to data on persistent volume
echo "Linking data directories..."
rm -rf data raw_data 2>/dev/null || true
ln -sf /workspace/data ./data
ln -sf /workspace/raw_data ./raw_data

# Create checkpoints directory on persistent volume
mkdir -p /workspace/checkpoints
ln -sf /workspace/checkpoints ./checkpoints

echo "=== Setup Complete ==="
echo ""
echo "To start training, run:"
echo ""
echo "cd /workspace/retail_sim/src/training"
echo "poetry run python train_next_basket.py \\"
echo "    --epochs 20 \\"
echo "    --batch-size 128 \\"
echo "    --device cuda \\"
echo "    --gradient-checkpointing \\"
echo "    --eval-every 5000 \\"
echo "    --save-every-steps 10000 \\"
echo "    --log-every 500"
