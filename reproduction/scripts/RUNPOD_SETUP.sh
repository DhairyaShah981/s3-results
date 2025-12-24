#!/bin/bash
# =============================================================================
# S3 RunPod Setup Script
# Author: Dhairya Shah (reproduction of s3 paper results)
# =============================================================================
# 
# Prerequisites:
#   - RunPod instance with 2x A100 80GB GPUs
#   - Container Disk: 100GB, Volume Disk: 150GB
#   - Base image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
#
# Usage:
#   bash RUNPOD_SETUP.sh
# =============================================================================

set -e
echo "=============================================="
echo "S3 RunPod Setup - Starting..."
echo "=============================================="

# Navigate to workspace
cd /workspace

# Clone the repository
if [ ! -d "s3-replication" ]; then
    mkdir -p s3-replication
    cd s3-replication
else
    cd s3-replication
fi

# Install core dependencies
echo "[1/6] Installing core dependencies..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install ray transformers datasets pyserini flask requests tqdm
pip install sentence-transformers
pip install langchain-openai langchain-community langchain-core
pip install huggingface_hub
pip install "outlines>=0.0.43,<0.1"

# Install FAISS for GPU
echo "[2/6] Installing FAISS..."
pip install faiss-gpu

# Create necessary directories
echo "[3/6] Creating directories..."
mkdir -p data logs verl_checkpoints generator_llms

# Download s3 data
echo "[4/6] Downloading s3 data..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Download processed data
snapshot_download(
    repo_id="pat-jj/s3_processed_data",
    repo_type="dataset",
    local_dir="data/s3_official",
    local_dir_use_symlinks=False
)
print("Data downloaded to data/s3_official/")
EOF

# Download model checkpoint
echo "[5/6] Downloading model checkpoint..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os
import shutil

# Download checkpoint
snapshot_download(
    repo_id="pat-jj/s3-8-3-3-20steps",
    local_dir="verl_checkpoints/s3_8_3_3_20steps",
    local_dir_use_symlinks=False
)

# Restructure checkpoint directory
src = "verl_checkpoints/s3_8_3_3_20steps"
dst = "verl_checkpoints/s3_8_3_3_20steps/actor/global_step_20"
os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if f.endswith('.safetensors') or f.endswith('.json'):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
        
print("Checkpoint ready at verl_checkpoints/s3_8_3_3_20steps/actor/global_step_20/")
EOF

# Download retrieval index
echo "[6/6] Downloading retrieval index..."
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

os.makedirs("data/retrieval_index", exist_ok=True)

# Download E5 index
hf_hub_download(
    repo_id="pat-jj/s3_processed_data",
    repo_type="dataset",
    filename="e5_Flat.index",
    local_dir="data/retrieval_index",
    local_dir_use_symlinks=False
)

# Download corpus
hf_hub_download(
    repo_id="pat-jj/s3_processed_data", 
    repo_type="dataset",
    filename="wiki-18.jsonl",
    local_dir="data/retrieval_index",
    local_dir_use_symlinks=False
)

print("Retrieval index downloaded!")
EOF

# Create dummy API key files (not used but required by imports)
echo "[Creating dummy API key files...]"
mkdir -p generator_llms
echo "dummy" > generator_llms/aws_access.key
echo "dummy" > generator_llms/claude_api_aws.key
echo "dummy" > generator_llms/together_api_key.key
echo "dummy" > generator_llms/ali_api.key
echo "dummy" > generator_llms/openai_api_azure.key
echo "dummy" > generator_llms/deepinfra.key

echo "=============================================="
echo "S3 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Start servers: bash start_servers.sh"
echo "  2. Run quick test: bash run_quick_test.sh"
echo "  3. Run evaluation: python3 evaluate_with_paper_metrics.py --input_dir data/output_quick_test"
echo ""

