#!/bin/bash
#SBATCH -J LossAblation
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH --time=72:00:00
#SBATCH --output=loss_ablation_%j.out
#SBATCH --error=loss_ablation_%j.err

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

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Timestamp: $TIMESTAMP"

# Create a log directory for this job
LOG_DIR="loss_ablation_logs_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p $LOG_DIR

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/job_progress.log"
}

# Function to run a single architecture test
run_architecture_test() {
    local arch=$1
    local log_file="$LOG_DIR/${arch}_test.log"
    
    log_message "Starting testing for architecture: $arch"
    log_message "Log file: $log_file"
    
    # Run the loss ablation testing for this architecture
    python run_script_loss.py $arch 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_message "✓ Architecture $arch completed successfully"
        return 0
    else
        log_message "✗ Architecture $arch failed with exit code $exit_code"
        return 1
    fi
}

# Function to analyze results after completion
analyze_results() {
    log_message "Starting results analysis..."
    
    # Run the analysis script
    python analyze_loss_results.py 2>&1 | tee "$LOG_DIR/analysis.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_message "✓ Results analysis completed successfully"
    else
        log_message "✗ Results analysis failed with exit code $exit_code"
    fi
}

# Function to create a summary report
create_summary_report() {
    local report_file="$LOG_DIR/job_summary_${TIMESTAMP}.txt"
    
    log_message "Creating job summary report: $report_file"
    
    cat > "$report_file" << EOF
LOSS FUNCTION ABLATION TESTING JOB SUMMARY
==========================================

Job Information:
- Job ID: $SLURM_JOB_ID
- Start Time: $TIMESTAMP
- End Time: $(date +"%Y%m%d_%H%M%S")
- Node: $SLURM_NODELIST
- Architecture: $ARCHITECTURE

Test Configuration:
- Fixed Parameters:
  * Alpha: 0.6
  * Number of clusters: 64
  * PCA dimensions: 50
- Variable Parameters:
  * Seeds: 5 (1001-1005)
  * Bit configurations: 4 (W2/A2, W4/A2, W2/A4, W4/A4)
  * Loss configurations: 7
- Total experiments per architecture: 140

Results:
- Results directories: results_alpha0.60_clusters64_pca50_*
- Analysis summary: loss_ablation_summary/
- Individual logs: $LOG_DIR/

Loss Configurations Tested:
1. full_loss: rec_loss + round_loss + pd_loss
2. no_rec_loss: round_loss + pd_loss
3. no_round_loss: rec_loss + pd_loss
4. no_pd_loss: rec_loss + round_loss
5. rec_loss_only: rec_loss
6. round_loss_only: round_loss
7. pd_loss_only: pd_loss

Estimated Duration:
- Per experiment: 10-30 minutes
- Total estimated time: 23-70 hours per architecture

Notes:
- Check individual experiment directories for detailed results
- Use analyze_loss_results.py for comprehensive analysis
- Monitor GPU memory and disk space during execution
EOF

    log_message "Summary report created: $report_file"
}

# Main execution
main() {
    # Check if architecture argument is provided
    if [ $# -eq 0 ]; then
        log_message "No architecture specified. Using default: resnet18"
        ARCHITECTURE="resnet18"
    else
        ARCHITECTURE=$1
        log_message "Testing architecture: $ARCHITECTURE"
    fi
    
    # Validate architecture
    case $ARCHITECTURE in
        resnet18|resnet50|mobilenetv2|regnetx_600m|regnetx_3200m|mnasnet)
            log_message "Valid architecture: $ARCHITECTURE"
            ;;
        *)
            log_message "Invalid architecture: $ARCHITECTURE"
            log_message "Valid options: resnet18, resnet50, mobilenetv2, regnetx_600m, regnetx_3200m, mnasnet"
            exit 1
            ;;
    esac
    
    # Log system information
    log_message "System Information:"
    log_message "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    log_message "  Memory: $(free -h | grep Mem | awk '{print $2}')"
    log_message "  Disk: $(df -h . | tail -1 | awk '{print $4}') available"
    
    # Start the testing
    log_message "Starting loss function ablation testing..."
    
    # Run the architecture test
    if run_architecture_test $ARCHITECTURE; then
        log_message "✓ All tests completed successfully for $ARCHITECTURE"
        
        # Analyze results
        analyze_results
        
        # Create summary report
        create_summary_report
        
        log_message "=========================================="
        log_message "Loss Function Ablation Testing Completed"
        log_message "=========================================="
        log_message "Check the following for results:"
        log_message "  - Results directories: results_alpha0.60_clusters64_pca50_*"
        log_message "  - Analysis summary: loss_ablation_summary/"
        log_message "  - Job logs: $LOG_DIR/"
        
    else
        log_message "✗ Testing failed for architecture $ARCHITECTURE"
        log_message "Check logs in $LOG_DIR/ for details"
        exit 1
    fi
}

# Trap signals to ensure cleanup
trap 'log_message "Job interrupted. Cleaning up..."; exit 1' INT TERM

# Run main function with command line arguments
main "$@"

# Final cleanup and status
log_message "Job completed at $(date)"
log_message "Final log directory: $LOG_DIR"
log_message "=========================================="
