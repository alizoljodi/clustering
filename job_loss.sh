#!/bin/bash
#SBATCH -J TestJob
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

# Loss Function Ablation Testing Job Script
# This script runs systematic testing of different loss function configurations
# for neural network quantization.

echo "=========================================="
echo "Loss Function Ablation Testing Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Activate the virtual environment
source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate

# Install required packages
echo "Installing required packages..."
pip install scikit-learn pandas matplotlib numpy

python run_script_loss.py resnet18
