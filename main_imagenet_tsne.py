import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import random
import time
import hubconf  # noqa: F401
import copy
import pandas as pd
import gc
from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)
from data.imagenet import build_imagenet_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


def save_logits_to_csv(all_q_logits, all_fp_logits, all_corrected_logits, all_affine_corrected_logits, results_dir, arch, n_bit_w, n_bit_a, seed, chunk_size=1000):
    """
    Save all logits data as CSV files for analysis.
    Automatically uses chunking for large datasets.
    """
    try:
        # Concatenate all batches
        q_logits = torch.cat(all_q_logits, dim=0)
        fp_logits = torch.cat(all_fp_logits, dim=0)
        corrected_logits = torch.cat(all_corrected_logits, dim=0)
        affine_corrected_logits = torch.cat(all_affine_corrected_logits, dim=0)
        
        total_samples = len(q_logits)
        print(f"Saving logits data for {total_samples} samples...")
        
        # Use chunking for large datasets
        if total_samples > chunk_size:
            print(f"Large dataset detected ({total_samples} samples), using chunking...")
            return save_logits_in_chunks(all_q_logits, all_fp_logits, all_corrected_logits, all_affine_corrected_logits,
                                      results_dir, arch, n_bit_w, n_bit_a, seed, chunk_size)
        
        # Create a base filename with model parameters
        base_filename = f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
        
        # Save quantized logits
        q_df = pd.DataFrame(q_logits.numpy())
        q_csv_filename = os.path.join(results_dir, f"{base_filename}_quantized.csv")
        q_df.to_csv(q_csv_filename, index=False)
        print(f"Quantized logits saved as: {q_csv_filename}")
        
        # Save full-precision logits
        fp_df = pd.DataFrame(fp_logits.numpy())
        fp_csv_filename = os.path.join(results_dir, f"{base_filename}_fullprecision.csv")
        fp_df.to_csv(fp_csv_filename, index=False)
        print(f"Full-precision logits saved as: {fp_csv_filename}")
        
        # Save corrected logits
        corrected_df = pd.DataFrame(corrected_logits.numpy())
        corrected_csv_filename = os.path.join(results_dir, f"{base_filename}_corrected.csv")
        corrected_df.to_csv(corrected_csv_filename, index=False)
        print(f"Corrected logits saved as: {corrected_csv_filename}")

        # Save affine corrected logits
        affine_corrected_df = pd.DataFrame(affine_corrected_logits.numpy())
        affine_corrected_csv_filename = os.path.join(results_dir, f"{base_filename}_affine_corrected.csv")
        affine_corrected_df.to_csv(affine_corrected_csv_filename, index=False)
        print(f"Affine corrected logits saved as: {affine_corrected_csv_filename}")
        
        # Save metadata about the logits
        metadata = {
            'architecture': arch,
            'weight_bits': n_bit_w,
            'activation_bits': n_bit_a,
            'seed': seed,
            'num_samples': total_samples,
            'num_classes': q_logits.shape[1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_csv_filename = os.path.join(results_dir, f"{base_filename}_metadata.csv")
        metadata_df.to_csv(metadata_csv_filename, index=False)
        print(f"Logits metadata saved as: {metadata_csv_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error saving logits to CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_logits_in_chunks(all_q_logits, all_fp_logits, all_corrected_logits, all_affine_corrected_logits, results_dir, arch, n_bit_w, n_bit_a, seed, chunk_size=1000):
    """
    Save logits data in chunks to handle large datasets efficiently.
    """
    try:
        # Concatenate all batches
        q_logits = torch.cat(all_q_logits, dim=0)
        fp_logits = torch.cat(all_fp_logits, dim=0)
        corrected_logits = torch.cat(all_corrected_logits, dim=0)
        affine_corrected_logits = torch.cat(all_affine_corrected_logits, dim=0)
        
        total_samples = len(q_logits)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        print(f"Saving logits data for {total_samples} samples in {num_chunks} chunks...")
        
        # Create a base filename with model parameters
        base_filename = f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
        
        # Save chunked data
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
            
            chunk_suffix = f"_chunk{chunk_idx+1:03d}_of_{num_chunks:03d}"
            
            # Save quantized logits chunk
            q_chunk = q_logits[start_idx:end_idx]
            q_df = pd.DataFrame(q_chunk.numpy())
            q_csv_filename = os.path.join(results_dir, f"{base_filename}_quantized{chunk_suffix}.csv")
            q_df.to_csv(q_csv_filename, index=False)
            
            # Save full-precision logits chunk
            fp_chunk = fp_logits[start_idx:end_idx]
            fp_df = pd.DataFrame(fp_chunk.numpy())
            fp_csv_filename = os.path.join(results_dir, f"{base_filename}_fullprecision{chunk_suffix}.csv")
            fp_df.to_csv(fp_csv_filename, index=False)
            
            # Save corrected logits chunk
            corrected_chunk = corrected_logits[start_idx:end_idx]
            corrected_df = pd.DataFrame(corrected_chunk.numpy())
            corrected_csv_filename = os.path.join(results_dir, f"{base_filename}_corrected{chunk_suffix}.csv")
            corrected_df.to_csv(corrected_csv_filename, index=False)

            # Save affine corrected logits chunk
            affine_corrected_chunk = affine_corrected_logits[start_idx:end_idx]
            affine_corrected_df = pd.DataFrame(affine_corrected_chunk.numpy())
            affine_corrected_csv_filename = os.path.join(results_dir, f"{base_filename}_affine_corrected{chunk_suffix}.csv")
            affine_corrected_df.to_csv(affine_corrected_csv_filename, index=False)
            
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: {start_idx}-{end_idx} samples saved")
        
        # Save metadata about the chunked logits
        metadata = {
            'architecture': arch,
            'weight_bits': n_bit_w,
            'activation_bits': n_bit_a,
            'seed': seed,
            'total_samples': total_samples,
            'num_classes': q_logits.shape[1],
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_csv_filename = os.path.join(results_dir, f"{base_filename}_chunked_metadata.csv")
        metadata_df.to_csv(metadata_csv_filename, index=False)
        print(f"Chunked logits metadata saved as: {metadata_csv_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error saving chunked logits to CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_logits_summary_csv(arch, n_bit_w, n_bit_a, seed, results_summary):
    """
    Create a summary CSV file listing all saved logits files for this model configuration.
    """
    try:
        summary_dir = f"logits_summary_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create summary dataframe
        summary_data = []
        for result in results_summary:
            alpha = result['alpha']
            num_clusters = result['num_clusters']
            pca_dim = result['pca_dim']
            top1_acc = result['top1_accuracy']
            top5_acc = result['top5_accuracy']
            
            # Define the expected results directory
            results_dir = f"results_alpha{alpha:.2f}_clusters{num_clusters}_pca{pca_dim}_{arch}_w{n_bit_w}bit_a{n_bit_a}bit"
            
            # List expected CSV files
            base_filename = f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
            
            summary_data.append({
                'alpha': alpha,
                'num_clusters': num_clusters,
                'pca_dim': pca_dim,
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc,
                'results_directory': results_dir,
                'quantized_logits_file': f"{base_filename}_quantized.csv",
                'fullprecision_logits_file': f"{base_filename}_fullprecision.csv",
                'corrected_logits_file': f"{base_filename}_corrected.csv",
                'affine_corrected_logits_file': f"{base_filename}_affine_corrected.csv",
                'metadata_file': f"{base_filename}_metadata.csv",
                'quantized_statistics_tensor': f"{base_filename}_quantized_statistics.pt",
                'fullprecision_statistics_tensor': f"{base_filename}_fullprecision_statistics.pt",
                'corrected_statistics_tensor': f"{base_filename}_corrected_statistics.pt",
                'affine_corrected_statistics_tensor': f"{base_filename}_affine_corrected_statistics.pt"
            })
        
        # Add initial logits entry
        initial_results_dir = f"initial_logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
        summary_data.append({
            'alpha': 'initial',
            'num_clusters': 'N/A',
            'pca_dim': 'N/A',
            'top1_accuracy': 'N/A',
            'top5_accuracy': 'N/A',
            'results_directory': initial_results_dir,
            'quantized_logits_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_quantized.csv",
            'fullprecision_logits_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_fullprecision.csv",
            'corrected_logits_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_corrected.csv",
            'affine_corrected_logits_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_affine_corrected.csv",
            'metadata_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_metadata.csv",
            'quantized_statistics_tensor': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_quantized_statistics.pt",
            'fullprecision_statistics_tensor': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_fullprecision_statistics.pt",
            'corrected_statistics_tensor': 'N/A',
            'affine_corrected_statistics_tensor': 'N/A'
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_filename = os.path.join(summary_dir, f"logits_summary_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}.csv")
        summary_df.to_csv(summary_csv_filename, index=False)
        
        print(f"Logits summary saved as: {summary_csv_filename}")
        return summary_csv_filename
        
    except Exception as e:
        print(f"Error creating logits summary CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_logits_statistics_as_tensor(all_logits_list, results_dir, logits_type, arch, n_bit_w, n_bit_a, seed):
    """
    Save logits statistics as a tensor with 1000 rows containing mean and standard deviation for each entry.
    
    Args:
        all_logits_list: List of logits tensors from batches
        results_dir: Directory to save the results
        logits_type: Type of logits (e.g., 'quantized', 'fullprecision', 'corrected', 'affine_corrected')
        arch: Model architecture
        n_bit_w: Weight bit width
        n_bit_a: Activation bit width
        seed: Random seed
    """
    try:
        # Concatenate all batches
        all_logits = torch.cat(all_logits_list, dim=0)  # [N, C]
        
        total_samples = all_logits.shape[0]
        num_classes = all_logits.shape[1]
        
        print(f"Computing statistics for {logits_type} logits: {total_samples} samples, {num_classes} classes")
        
        # Compute mean and standard deviation across all samples for each class
        mean_logits = all_logits.mean(dim=0)  # [C]
        std_logits = all_logits.std(dim=0, unbiased=False)  # [C]
        
        # Create statistics tensor with 1000 rows
        # Each row contains: [class_id, mean_value, std_value]
        stats_tensor = torch.zeros(1000, 3)
        
        # Fill the first num_classes rows with actual statistics
        stats_tensor[:num_classes, 0] = torch.arange(num_classes)  # class_id
        stats_tensor[:num_classes, 1] = mean_logits  # mean_value
        stats_tensor[:num_classes, 2] = std_logits   # std_value
        
        # Fill remaining rows with zeros (padding)
        stats_tensor[num_classes:, 0] = -1  # -1 indicates no class
        
        # Save as tensor file
        base_filename = f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}"
        tensor_filename = os.path.join(results_dir, f"{base_filename}_{logits_type}_statistics.pt")
        torch.save(stats_tensor, tensor_filename)
        print(f"{logits_type.capitalize()} logits statistics saved as tensor: {tensor_filename}")
        
        # Also save as CSV for easy viewing
        csv_filename = os.path.join(results_dir, f"{base_filename}_{logits_type}_statistics.csv")
        stats_df = pd.DataFrame({
            'class_id': stats_tensor[:num_classes, 0].numpy(),
            'mean_value': stats_tensor[:num_classes, 1].numpy(),
            'std_value': stats_tensor[:num_classes, 2].numpy()
        })
        stats_df.to_csv(csv_filename, index=False)
        print(f"{logits_type.capitalize()} logits statistics saved as CSV: {csv_filename}")
        
        # Print summary statistics
        print(f"  Mean of means: {mean_logits.mean():.6f}")
        print(f"  Mean of stds: {std_logits.mean():.6f}")
        print(f"  Max mean: {mean_logits.max():.6f}")
        print(f"  Min mean: {mean_logits.min():.6f}")
        print(f"  Max std: {std_logits.max():.6f}")
        print(f"  Min std: {std_logits.min():.6f}")
        
        return stats_tensor
        
    except Exception as e:
        print(f"Error saving {logits_type} logits statistics: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='model name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='/mimer/NOBACKUP/groups/naiss2025-22-91/imagenet', type=str, help='path to ImageNet data')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset and training samples for logits extraction')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')
    
    # cluster affine parameters
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha blending parameter for cluster affine correction')
    parser.add_argument('--num_clusters', default=64, type=int, help='number of clusters for cluster affine correction')
    parser.add_argument('--pca_dim', default=50, type=int, help='PCA dimension for clustering (None to disable)')
    parser.add_argument('--use_global_tensors', action='store_true', help='if True, use global alpha/beta tensors instead of clustering')
    
    # Multiple parameter testing
    parser.add_argument('--alpha_list', nargs='+', type=float, help='list of alpha values to test')
    parser.add_argument('--num_clusters_list', nargs='+', type=int, help='list of cluster numbers to test')
    parser.add_argument('--pca_dim_list', nargs='+', type=int, help='list of PCA dimensions to test')
    args = parser.parse_args()

    seed_all(args.seed)
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    # load model
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()
    fp_model = copy.deepcopy(cnn)
    fp_model.cuda()
    fp_model.eval()

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    fp_model.set_quant_state(False, False)
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    qnn.disable_network_output_quantization()
    print('the quantized model is below!')
    print(qnn)
    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
                lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu, 
                lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c)


    '''init weight quantizer'''
    set_weight_quantize_params(qnn)

    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)
    # Start calibration
    recon_model(qnn, fp_model)

    qnn.set_quant_state(weight_quant=True, act_quant=True)
    print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
                                                           validate_model(test_loader, qnn)))
    
    def extract_model_logits(q_model, fp_model, dataloader, device, num_samples=None):
        """
        Extract logits from both quantized and full-precision models.
        Returns concatenated logits tensors.
        
        Args:
            q_model: Quantized model
            fp_model: Full-precision model
            dataloader: Data loader for training data
            device: Device to run models on
            num_samples: Maximum number of samples to process (None for all samples)
        """
        q_model.eval()
        fp_model.eval()

        all_q, all_fp = [],[]
        samples_processed = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                # Check if we've reached the desired number of samples
                if num_samples is not None and samples_processed >= num_samples:
                    break
                
                images = images.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)
                all_q.append(q_logits.cpu())
                all_fp.append(fp_logits.cpu())
                
                samples_processed += images.size(0)
                
                # Print progress every 100 batches
                if (i + 1) % 100 == 0:
                    print(f"Processed {samples_processed} samples...")

        all_q = torch.cat(all_q, dim=0)  # [N, C]
        all_fp = torch.cat(all_fp, dim=0)  # [N, C]
        
        print(f"Extracted logits from {all_q.shape[0]} samples")
        
        return all_q, all_fp
    
    def create_tsne_clustering_representation(logits, num_clusters=10, perplexity=30, random_state=42):
        """
        Create t-SNE clustering representation from logits.
        
        Args:
            logits: Tensor of shape [N, C] where N is number of samples, C is number of classes
            num_clusters: Number of clusters for K-means clustering
            perplexity: t-SNE perplexity parameter
            random_state: Random state for reproducibility
            
        Returns:
            tsne_coords: t-SNE coordinates [N, 2]
            cluster_ids: Cluster assignments [N]
            kmeans_model: Fitted K-means model
            tsne_model: Fitted t-SNE model
        """
        print(f"Creating t-SNE clustering representation for {logits.shape[0]} samples...")
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(logits):
            logits_np = logits.numpy()
        else:
            logits_np = logits
            
        # Apply K-means clustering first
        print(f"Applying K-means clustering with {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        cluster_ids = kmeans.fit_predict(logits_np)
        
        # Apply t-SNE for dimensionality reduction
        print("Applying t-SNE dimensionality reduction...")
        # Adjust perplexity based on data size
        adjusted_perplexity = min(perplexity, len(logits_np) // 4)
        tsne = TSNE(n_components=2, random_state=random_state, 
                    perplexity=adjusted_perplexity, n_iter=1000, learning_rate='auto')
        tsne_coords = tsne.fit_transform(logits_np)
        
        print(f"t-SNE completed. Final shape: {tsne_coords.shape}")
        
        return tsne_coords, cluster_ids, kmeans, tsne
    
    def visualize_tsne_clustering(tsne_coords, cluster_ids, save_path=None, title="t-SNE Clustering of Quantized Logits"):
        """
        Visualize t-SNE clustering results.
        
        Args:
            tsne_coords: t-SNE coordinates [N, 2]
            cluster_ids: Cluster assignments [N]
            save_path: Path to save the plot (optional)
            title: Plot title
        """
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with different colors for each cluster
        unique_clusters = np.unique(cluster_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_ids == cluster_id
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=20)
        
        plt.title(title, fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE visualization saved to: {save_path}")
        
        plt.show()
        
        # Print cluster statistics
        print("\nCluster Statistics:")
        for cluster_id in unique_clusters:
            cluster_size = np.sum(cluster_ids == cluster_id)
            cluster_percentage = (cluster_size / len(cluster_ids)) * 100
            print(f"Cluster {cluster_id}: {cluster_size} samples ({cluster_percentage:.1f}%)")
    
    # Extract logits
    all_q_logits, all_fp_logits = extract_model_logits(qnn, fp_model, test_loader, device, num_samples=None)
    
    # Create t-SNE clustering representation
    print("\n" + "="*60)
    print("CREATING T-SNE CLUSTERING REPRESENTATION")
    print("="*60)
    
    # Apply t-SNE clustering
    tsne_coords, cluster_ids, kmeans_model, tsne_model = create_tsne_clustering_representation(
        all_q_logits, 
        num_clusters=10,  # You can adjust this parameter
        perplexity=30,    # You can adjust this parameter
        random_state=42
    )
    
    # Visualize the clustering
    results_dir = f"results/{args.arch}_W{args.n_bits_w}A{args.n_bits_a}_seed{args.seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    tsne_plot_path = os.path.join(results_dir, f"tsne_clustering_W{args.n_bits_w}A{args.n_bits_a}_seed{args.seed}.png")
    visualize_tsne_clustering(tsne_coords, cluster_ids, save_path=tsne_plot_path)
    
    # Save t-SNE data and cluster assignments
    tsne_data = {
        'tsne_component_1': tsne_coords[:, 0],
        'tsne_component_2': tsne_coords[:, 1],
        'cluster_id': cluster_ids,
        'sample_index': np.arange(len(tsne_coords))
    }
    
    tsne_df = pd.DataFrame(tsne_data)
    tsne_csv_path = os.path.join(results_dir, f"tsne_clustering_data_W{args.n_bits_w}A{args.n_bits_a}_seed{args.seed}.csv")
    tsne_df.to_csv(tsne_csv_path, index=False)
    print(f"\nt-SNE clustering data saved to: {tsne_csv_path}")
    
    # Additional analysis: Cluster quality metrics
    print("\n" + "="*60)
    print("CLUSTER QUALITY ANALYSIS")
    print("="*60)
    
    # Calculate silhouette score for cluster quality
    try:
        silhouette_avg = silhouette_score(tsne_coords, cluster_ids)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print("(Higher values indicate better-defined clusters)")
    except Exception as e:
        print(f"Could not calculate silhouette score: {e}")
    
    # Calculate cluster separation (average distance between cluster centers)
    cluster_centers = kmeans_model.cluster_centers_
    if len(cluster_centers) > 1:
        center_distances = euclidean_distances(cluster_centers)
        avg_center_distance = np.mean(center_distances[center_distances > 0])
        print(f"Average distance between cluster centers: {avg_center_distance:.4f}")
    
    # Save cluster centers
    centers_df = pd.DataFrame(cluster_centers, columns=[f'feature_{i}' for i in range(cluster_centers.shape[1])])
    centers_df['cluster_id'] = range(len(cluster_centers))
    centers_csv_path = os.path.join(results_dir, f"cluster_centers_W{args.n_bits_w}A{args.n_bits_a}_seed{args.seed}.csv")
    centers_df.to_csv(centers_csv_path, index=False)
    print(f"Cluster centers saved to: {centers_csv_path}")
    
    print("\n" + "="*60)
    print("T-SNE CLUSTERING COMPLETED SUCCESSFULLY!")
    print("="*60)
