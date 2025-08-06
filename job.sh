#!/bin/bash
#SBATCH -J TestJob
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
pip install scikit-learn
python run_script.py resnet18
