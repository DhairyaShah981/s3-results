#!/bin/bash
# =============================================================================
# Run Quick S3 Inference Test (10 samples)
# Author: Dhairya Shah
# =============================================================================
#
# This script runs inference on 10 test samples to verify the setup.
# For full evaluation, use run_full_inference.sh
#
# Prerequisites:
#   - Run RUNPOD_SETUP.sh first
#   - Run start_servers.sh (servers must be running)
#
# Usage:
#   bash run_quick_test.sh
# =============================================================================

set -e

echo "=============================================="
echo "S3 Quick Test (10 samples)"
echo "=============================================="

cd /workspace/s3-replication

# Configuration
CHECKPOINT_DIR="verl_checkpoints/s3_8_3_3_20steps/actor/global_step_20"
OUTPUT_DIR="data/output_quick_test"
NUM_SAMPLES=10

# Check if servers are running
echo "[1/3] Checking servers..."
if ! curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "ERROR: Retriever server not running. Start with: bash start_servers.sh"
    exit 1
fi
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: Generator server not running. Start with: bash start_servers.sh"
    exit 1
fi
echo "✓ Both servers are running"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference
echo "[2/3] Running inference..."
python3 -m verl.trainer.main_ppo \
    data.train_files=data/s3_official/train_e5_s3.parquet \
    data.val_files=data/s3_official/test_e5_s3.parquet \
    data.val_data_num=$NUM_SAMPLES \
    data.output_dir=$OUTPUT_DIR \
    actor_rollout_ref.model.path=$CHECKPOINT_DIR \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n_gpus_per_node=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    critic.ppo_mini_batch_size=4 \
    trainer.val_only=True \
    trainer.val_batch_size=2 \
    +generator_llm="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" \
    2>&1 | tee logs/quick_test.log

echo "[3/3] Inference complete!"
echo ""
echo "Output files: $OUTPUT_DIR/"
echo "Log file: logs/quick_test.log"
echo ""
echo "To evaluate results, run:"
echo "  python3 evaluate_with_paper_metrics.py --input_dir $OUTPUT_DIR"
echo ""

