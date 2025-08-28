# Loss Function Ablation Job Script Usage Guide

This guide explains how to use the `job_loss.sh` SLURM script for running loss function ablation testing on a cluster.

## Overview

The `job_loss.sh` script is designed to run systematic loss function ablation testing on a SLURM cluster. It automatically handles:
- Environment setup
- Package installation
- Comprehensive logging
- Results analysis
- Summary report generation

## SLURM Configuration

The script includes the following SLURM directives:

```bash
#SBATCH -J LossAblation          # Job name
#SBATCH -c 8                     # Number of CPU cores
#SBATCH --mem=128G               # Memory requirement
#SBATCH -p gpu_computervision_long # Partition
#SBATCH --gres=gpu:1             # GPU requirement
#SBATCH --tmp=5G                 # Temporary disk space
#SBATCH --time=72:00:00          # Time limit (72 hours)
#SBATCH --output=loss_ablation_%j.out # Output log
#SBATCH --error=loss_ablation_%j.err  # Error log
```

## Usage

### 1. Basic Usage (Default Architecture)

```bash
# Submit job with default architecture (resnet18)
sbatch job_loss.sh
```

### 2. Specify Architecture

```bash
# Submit job for specific architecture
sbatch job_loss.sh resnet18
sbatch job_loss.sh resnet50
sbatch job_loss.sh mobilenetv2
sbatch job_loss.sh regnetx_600m
sbatch job_loss.sh regnetx_3200m
sbatch job_loss.sh mnasnet
```

### 3. Submit Multiple Jobs

```bash
# Submit jobs for different architectures
sbatch job_loss.sh resnet18
sbatch job_loss.sh resnet50
sbatch job_loss.sh mobilenetv2
```

## What the Job Does

### 1. Environment Setup
- Activates the virtual environment
- Installs required packages (scikit-learn, pandas, matplotlib, numpy)
- Sets performance environment variables

### 2. Testing Execution
- Runs `run_script_loss.py` for the specified architecture
- Tests 7 different loss configurations
- Runs 140 total experiments (5 seeds × 4 bit configs × 7 loss configs)

### 3. Results Analysis
- Automatically runs `analyze_loss_results.py`
- Creates summary reports
- Organizes all outputs

### 4. Logging and Monitoring
- Creates timestamped log directories
- Logs all progress with timestamps
- Captures system information (GPU, memory, disk)

## Expected Duration

- **Per experiment**: 10-30 minutes
- **Total per architecture**: 23-70 hours
- **Job time limit**: 72 hours (should be sufficient for most architectures)

## Output Structure

### Job Logs
```
loss_ablation_logs_{JOB_ID}_{TIMESTAMP}/
├── job_progress.log          # Main progress log
├── {architecture}_test.log   # Architecture-specific test log
├── analysis.log              # Results analysis log
└── job_summary_{TIMESTAMP}.txt # Final summary report
```

### Results
```
results_alpha0.60_clusters64_pca50_{architecture}_w{bits}bit_a{bits}bit/
├── experiment_parameters.csv
├── cluster_comparison.png
├── combined_logits.png
├── *_histogram.png
├── cluster_visualization_*.png
├── logits_*.csv
└── README.md
```

### Analysis Summary
```
loss_ablation_summary/
└── ablation_results_summary.csv
```

## Monitoring Your Job

### 1. Check Job Status
```bash
squeue -u $USER
```

### 2. Monitor Progress
```bash
# Check the output log
tail -f loss_ablation_{JOB_ID}.out

# Check the error log
tail -f loss_ablation_{JOB_ID}.err

# Check progress in the log directory
tail -f loss_ablation_logs_{JOB_ID}_*/job_progress.log
```

### 3. Check Results
```bash
# List results directories
ls -la results_alpha0.60_clusters64_pca50_*

# Check analysis summary
ls -la loss_ablation_summary/
```

## Customization

### 1. Modify SLURM Parameters
Edit the SLURM directives in `job_loss.sh`:
```bash
#SBATCH --mem=256G              # Increase memory
#SBATCH --time=120:00:00        # Increase time limit
#SBATCH --gres=gpu:2            # Request 2 GPUs
```

### 2. Change Environment Path
Update the virtual environment path:
```bash
source /path/to/your/venv/bin/activate
```

### 3. Modify Package Installation
Add or remove packages:
```bash
pip install scikit-learn pandas matplotlib numpy torch torchvision
```

## Troubleshooting

### Common Issues

1. **Job Fails Immediately**
   - Check if the virtual environment path is correct
   - Verify that required packages can be installed
   - Check the error log: `loss_ablation_{JOB_ID}.err`

2. **Out of Memory**
   - Increase memory allocation: `#SBATCH --mem=256G`
   - Check if the model fits in GPU memory

3. **Time Limit Exceeded**
   - Increase time limit: `#SBATCH --time=120:00:00`
   - Consider testing with fewer seeds or bit configurations

4. **GPU Issues**
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation
   - Check GPU memory usage

### Debugging Tips

1. **Check Job Logs**
   ```bash
   # Find your job ID
   squeue -u $USER
   
   # Check logs
   ls -la loss_ablation_logs_*
   tail -f loss_ablation_logs_*/job_progress.log
   ```

2. **Test Locally First**
   ```bash
   # Test the script locally before submitting
   bash job_loss.sh resnet18
   ```

3. **Monitor Resources**
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Check disk space
   df -h
   
   # Check memory usage
   free -h
   ```

## Best Practices

1. **Start Small**: Begin with ResNet18 to test the setup
2. **Monitor Resources**: Keep track of GPU memory and disk usage
3. **Use Screen/Tmux**: For long-running jobs on login nodes
4. **Backup Results**: Save important results before running long experiments
5. **Check Logs Regularly**: Monitor progress and catch issues early

## Example Job Submission

```bash
# Submit job for ResNet18
echo "Submitting ResNet18 loss ablation job..."
JOB_ID=$(sbatch job_loss.sh resnet18 | awk '{print $4}')
echo "Job submitted with ID: $JOB_ID"

# Monitor the job
echo "Monitoring job progress..."
tail -f loss_ablation_${JOB_ID}.out
```

## Support

If you encounter issues:

1. Check the job logs in `loss_ablation_logs_{JOB_ID}_*/`
2. Review the SLURM output and error logs
3. Verify system resources (GPU, memory, disk)
4. Check that all dependencies are available
5. Test the script locally if possible

---

**Note**: This job script is designed for systematic loss function ablation testing. It will run 140 experiments per architecture, so ensure you have sufficient time and resources allocated.
