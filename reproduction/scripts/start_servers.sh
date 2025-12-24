#!/bin/bash
# =============================================================================
# Start Retriever and Generator Servers for S3 Inference
# Author: Dhairya Shah
# =============================================================================
#
# This script starts:
#   1. Retriever Server (E5 embeddings + FAISS index) on GPU 1
#   2. Generator Server (Qwen2.5-7B-Instruct-GPTQ-Int4) on GPU 0
#
# Prerequisites:
#   - Run RUNPOD_SETUP.sh first
#   - 2x A100 80GB GPUs available
#
# Usage:
#   bash start_servers.sh
# =============================================================================

set -e

echo "=============================================="
echo "Starting S3 Servers..."
echo "=============================================="

cd /workspace/s3-replication
mkdir -p logs

# Kill any existing Python processes
echo "[1/4] Cleaning up existing processes..."
pkill -9 python 2>/dev/null || true
sleep 3

# Check GPU status
echo "[2/4] Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Start Retriever Server on GPU 1
echo "[3/4] Starting Retriever Server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python s3/s3/search/retrieval_server.py \
    --index_path data/retrieval_index/e5_Flat.index \
    --corpus_path data/retrieval_index/wiki-18.jsonl \
    --port 5001 \
    > logs/retriever.log 2>&1 &

echo "Waiting for Retriever to load (this takes ~2 minutes)..."
for i in {1..24}; do
    sleep 5
    echo "  Still loading... ($((i*5))s)"
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo "  Retriever is ready!"
        break
    fi
done

# Start Generator Server on GPU 0
echo "[4/4] Starting Generator Server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --port 8000 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    > logs/generator.log 2>&1 &

echo "Waiting for Generator to load (this takes ~1 minute)..."
for i in {1..12}; do
    sleep 5
    echo "  Still loading... ($((i*5))s)"
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  Generator is ready!"
        break
    fi
done

# Verify servers
echo ""
echo "=============================================="
echo "Server Status:"
echo "=============================================="

if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✓ Retriever Server: http://localhost:5001"
else
    echo "✗ Retriever Server: NOT READY (check logs/retriever.log)"
fi

if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✓ Generator Server: http://localhost:8000"
else
    echo "✗ Generator Server: NOT READY (check logs/generator.log)"
fi

echo ""
echo "Logs:"
echo "  tail -f logs/retriever.log"
echo "  tail -f logs/generator.log"
echo ""

