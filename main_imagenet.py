import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import random
import time
import hubconf  # noqa: F401
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
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


def save_logits_to_csv(all_q_logits, all_fp_logits, all_corrected_logits, results_dir, arch, n_bit_w, n_bit_a, seed, chunk_size=1000):
    """
    Save all logits data as CSV files for analysis.
    Automatically uses chunking for large datasets.
    """
    try:
        # Concatenate all batches
        q_logits = torch.cat(all_q_logits, dim=0)
        fp_logits = torch.cat(all_fp_logits, dim=0)
        corrected_logits = torch.cat(all_corrected_logits, dim=0)
        
        total_samples = len(q_logits)
        print(f"Saving logits data for {total_samples} samples...")
        
        # Use chunking for large datasets
        if total_samples > chunk_size:
            print(f"Large dataset detected ({total_samples} samples), using chunking...")
            return save_logits_in_chunks(all_q_logits, all_fp_logits, all_corrected_logits, 
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


def save_logits_in_chunks(all_q_logits, all_fp_logits, all_corrected_logits, results_dir, arch, n_bit_w, n_bit_a, seed, chunk_size=1000):
    """
    Save logits data in chunks to handle large datasets efficiently.
    """
    try:
        # Concatenate all batches
        q_logits = torch.cat(all_q_logits, dim=0)
        fp_logits = torch.cat(all_fp_logits, dim=0)
        corrected_logits = torch.cat(all_corrected_logits, dim=0)
        
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
                'metadata_file': f"{base_filename}_metadata.csv"
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
            'metadata_file': f"logits_{arch}_w{n_bit_w}bit_a{n_bit_a}bit_seed{seed}_metadata.csv"
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
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
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

    def extract_model_logits(q_model, fp_model, dataloader, device):
        """
        Extract logits from both quantized and full-precision models.
        Returns concatenated logits tensors.
        """
        q_model.eval()
        fp_model.eval()

        all_q, all_fp = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                #if i>=10:
                #    break
                images = images.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)
                all_q.append(q_logits.cpu())
                all_fp.append(fp_logits.cpu())

        all_q = torch.cat(all_q, dim=0)  # [N, C]
        all_fp = torch.cat(all_fp, dim=0)  # [N, C]
        
        return all_q, all_fp

    def build_cluster_affine(all_q, all_fp, num_clusters=64, pca_dim=None):
        """
        Build cluster affine correction model from pre-extracted logits.
        """
        # Optional PCA for clustering only
        pca = None
        if pca_dim is not None and pca_dim < all_q.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q.numpy())
        else:
            q_features = all_q.numpy()

        # Cluster quantized outputs
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = cluster_model.fit_predict(q_features)

        # For each cluster: learn gamma, beta (per-class)
        gamma_dict = {}
        beta_dict = {}

        for cid in range(num_clusters):
            idxs = (cluster_ids == cid)
            if idxs.sum() == 0:
                # Empty cluster, default to identity
                gamma_dict[cid] = torch.ones(all_q.shape[1])
                beta_dict[cid] = torch.zeros(all_q.shape[1])
                continue

            q_c = all_q[idxs]  # [Nc, C]
            fp_c = all_fp[idxs]  # [Nc, C]

            # Closed-form least squares: fp ≈ gamma * q + beta
            mean_q = q_c.mean(dim=0)
            mean_fp = fp_c.mean(dim=0)

            # Compute variance, avoid div by zero
            var_q = q_c.var(dim=0, unbiased=False)
            var_q[var_q < 1e-8] = 1e-8

            gamma = ((q_c - mean_q) * (fp_c - mean_fp)).mean(dim=0) / var_q
            beta = mean_fp - gamma * mean_q

            gamma_dict[cid] = gamma
            beta_dict[cid] = beta

        return cluster_model, gamma_dict, beta_dict, pca

    def apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=None, alpha=0.4):
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        """
        q_np = q_logits.cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            gamma = gamma_dict[cid].to(q.device)
            beta = beta_dict[cid].to(q.device)
            affine_corrected = q * gamma + beta
            blended = q + alpha * (affine_corrected - q)
            corrected.append(blended)
        return torch.stack(corrected)
    
    def evaluate_cluster_affine_with_alpha(q_model, fp_model, cluster_model, gamma_dict, beta_dict, dataloader, device, pca=None, alpha=0.4):
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        
        # Store logits for plotting
        all_q_logits = []
        all_fp_logits = []
        all_corrected_logits = []
        all_cluster_ids = []

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)

                corrected_logits = apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha)

                # Store logits for plotting
                all_q_logits.append(q_logits.cpu())
                all_fp_logits.append(fp_logits.cpu())
                all_corrected_logits.append(corrected_logits.cpu())
                
                # Get cluster IDs for this batch
                q_np = q_logits.cpu().numpy()
                if pca is not None:
                    q_np = pca.transform(q_np)
                cluster_ids = cluster_model.predict(q_np)
                all_cluster_ids.append(cluster_ids)

                acc1, acc5 = accuracy(corrected_logits, targets, topk=(1, 5))
                total_top1 += acc1.item() * images.size(0)
                total_top5 += acc5.item() * images.size(0)
                total += images.size(0)

        print(f"[Alpha={alpha:.2f}] Top-1 Accuracy: {total_top1 / total:.2f}%")
        print(f"[Alpha={alpha:.2f}] Top-5 Accuracy: {total_top5 / total:.2f}%")
        
        # Plot randomly selected values from each cluster
        plot_cluster_comparisons(all_q_logits, all_fp_logits, all_corrected_logits, 
                               all_cluster_ids, alpha, pca_dim=pca.n_components_ if pca else None, 
                               num_clusters=cluster_model.n_clusters, 
                               arch=args.arch, n_bit_w=args.n_bits_w, n_bit_a=args.n_bits_a)
        
        # Save logits data as CSV files
        results_dir = f"results_alpha{alpha:.2f}_clusters{cluster_model.n_clusters}_pca{pca.n_components_ if pca else 'none'}_{args.arch}_w{args.n_bits_w}bit_a{args.n_bits_a}bit"
        os.makedirs(results_dir, exist_ok=True)
        
        save_logits_to_csv(all_q_logits, all_fp_logits, all_corrected_logits, 
                          results_dir, args.arch, args.n_bits_w, args.n_bits_a, args.seed)
        
        return total_top1 / total, total_top5 / total
    
    def plot_cluster_comparisons(all_q_logits, all_fp_logits, all_corrected_logits, all_cluster_ids, alpha, pca_dim=None, num_clusters=None, arch=None, n_bit_w=None, n_bit_a=None):
        """
        Plot randomly selected 5 values from each cluster comparing q_logits, fp_logits, and corrected_logits.
        Also create combined logits plot and histogram comparison.
        """
        try:
            # Create results directory structure with model parameters
            pca_suffix = f"_pca{pca_dim}" if pca_dim else ""
            arch_suffix = f"_{arch}" if arch else ""
            w_suffix = f"_w{n_bit_w}bit" if n_bit_w else ""
            a_suffix = f"_a{n_bit_a}bit" if n_bit_a else ""
            
            results_dir = f"results_alpha{alpha:.2f}_clusters{num_clusters}{pca_suffix}{arch_suffix}{w_suffix}{a_suffix}"
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"Results will be saved in: {results_dir}")
            
            # Save experiment parameters summary
            params_summary = {
                'alpha': alpha,
                'num_clusters': num_clusters,
                'pca_dim': pca_dim,
                'architecture': arch,
                'weight_bits': n_bit_w,
                'activation_bits': n_bit_a,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save parameters to CSV
            params_df = pd.DataFrame([params_summary])
            params_csv_filename = os.path.join(results_dir, f"experiment_parameters.csv")
            params_df.to_csv(params_csv_filename, index=False)
            print(f"Experiment parameters saved as: {params_csv_filename}")
            
            # Create README file explaining the results
            readme_content = f"""# Experiment Results

## Parameters
- Alpha: {alpha:.2f}
- Number of Clusters: {num_clusters}
- PCA Dimensions: {pca_dim if pca_dim else 'None'}
- Architecture: {arch if arch else 'Unknown'}
- Weight Bits: {n_bit_w if n_bit_w else 'Unknown'}
- Activation Bits: {n_bit_a if n_bit_a else 'Unknown'}
- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Files Description

### Plots (PNG)
- `cluster_comparison.png` - Cluster comparison plots (one row per cluster, 3 columns)
- `combined_logits.png` - Combined logits visualization (all three types on same diagram)
- `quantized_histogram.png` - Quantized logits histogram with entropy
- `fullprecision_histogram.png` - Full-precision logits histogram with entropy
- `corrected_histogram.png` - Corrected logits histogram with entropy
- `cluster_visualization_tsne.png` - 2D t-SNE cluster visualization
- `cluster_visualization_pca.png` - 2D PCA cluster visualization
- `cluster_visualization_pca_3d.png` - 3D PCA cluster visualization

### Data (CSV)
- `experiment_parameters.csv` - All experiment parameters
- `cluster_comparison_data.csv` - Cluster comparison data
- `combined_logits_data.csv` - Combined logits data
- `quantized_histogram_data.csv` - Quantized histogram data + entropy
- `fullprecision_histogram_data.csv` - Full-precision histogram data + entropy
- `corrected_histogram_data.csv` - Corrected histogram data + entropy
- `cluster_visualization_tsne_data.csv` - t-SNE coordinates + cluster IDs
- `cluster_visualization_pca_data.csv` - PCA coordinates + cluster IDs + explained variance
- `cluster_visualization_pca_3d_data.csv` - 3D PCA coordinates + cluster IDs + explained variance

## Analysis
Use these files to analyze:
1. How well the clustering separates different logit patterns
2. The effectiveness of the correction method
3. Distribution differences between quantized, full-precision, and corrected logits
4. Entropy changes across different model configurations
"""
            
            readme_filename = os.path.join(results_dir, f"README.md")
            with open(readme_filename, 'w') as f:
                f.write(readme_content)
            print(f"README file created: {readme_filename}")
            
            # Concatenate all batches
            q_logits = torch.cat(all_q_logits, dim=0)
            fp_logits = torch.cat(all_fp_logits, dim=0)
            corrected_logits = torch.cat(all_corrected_logits, dim=0)
            cluster_ids = np.concatenate(all_cluster_ids)
            
            print(f"Plotting data for {len(cluster_ids)} total samples")
            
            # Get unique cluster IDs
            unique_clusters = np.unique(cluster_ids)
            num_clusters_actual = len(unique_clusters)
            
            # Create subplots: one row per cluster, 3 columns for q, fp, corrected
            fig, axes = plt.subplots(num_clusters_actual, 3, figsize=(15, 5*num_clusters_actual))
            if num_clusters_actual == 1:
                axes = axes.reshape(1, -1)
            
            # Set figure title
            pca_info = f" (PCA: {pca_dim})" if pca_dim else ""
            fig.suptitle(f'Cluster Comparison: Quantized vs Full-Precision vs Corrected Logits (α={alpha:.2f}){pca_info}', 
                         fontsize=16, y=0.98)
            
            # For each cluster
            for i, cluster_id in enumerate(unique_clusters):
                # Get indices for this cluster
                cluster_mask = cluster_ids == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                print(f"Cluster {cluster_id}: {len(cluster_indices)} samples")
                
                # Randomly select 5 samples from this cluster
                if len(cluster_indices) >= 5:
                    selected_indices = np.random.choice(cluster_indices, 5, replace=False)
                else:
                    selected_indices = cluster_indices
                
                # Plot quantized logits
                ax1 = axes[i, 0]
                q_data = q_logits[selected_indices].numpy()
                ax1.plot(q_data.T, 'b-', alpha=0.7, linewidth=1)
                ax1.set_title(f'Cluster {cluster_id}: Quantized Logits ({len(selected_indices)} samples)')
                ax1.set_xlabel('Class Index')
                ax1.set_ylabel('Logit Value')
                ax1.grid(True, alpha=0.3)
                
                # Plot full-precision logits
                ax2 = axes[i, 1]
                fp_data = fp_logits[selected_indices].numpy()
                ax2.plot(fp_data.T, 'g-', alpha=0.7, linewidth=1)
                ax2.set_title(f'Cluster {cluster_id}: Full-Precision Logits ({len(selected_indices)} samples)')
                ax2.set_xlabel('Class Index')
                ax2.set_ylabel('Logit Value')
                ax2.grid(True, alpha=0.3)
                
                # Plot corrected logits
                ax3 = axes[i, 2]
                corrected_data = corrected_logits[selected_indices].numpy()
                ax3.plot(corrected_data.T, 'r-', alpha=0.7, linewidth=1)
                ax3.set_title(f'Cluster {cluster_id}: Corrected Logits ({len(selected_indices)} samples)')
                ax3.set_xlabel('Class Index')
                ax3.set_ylabel('Logit Value')
                ax3.grid(True, alpha=0.3)
                
                # Add legend for this row
                legend_elements = [
                    mpatches.Patch(color='blue', label='Quantized'),
                    mpatches.Patch(color='green', label='Full-Precision'),
                    mpatches.Patch(color='red', label='Corrected')
                ]
                ax3.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save the cluster comparison plot
            filename = os.path.join(results_dir, f"cluster_comparison.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Cluster comparison plot saved as: {filename}")
            
            # Save cluster comparison data as CSV
            cluster_data = []
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_ids == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) >= 5:
                    selected_indices = np.random.choice(cluster_indices, 5, replace=False)
                else:
                    selected_indices = cluster_indices
                
                for idx in selected_indices:
                    q_vals = q_logits[idx].numpy()
                    fp_vals = fp_logits[idx].numpy()
                    corr_vals = corrected_logits[idx].numpy()
                    
                    for class_idx in range(len(q_vals)):
                        cluster_data.append({
                            'cluster_id': cluster_id,
                            'sample_index': idx,
                            'class_index': class_idx,
                            'quantized_logit': q_vals[class_idx],
                            'fullprecision_logit': fp_vals[class_idx],
                            'corrected_logit': corr_vals[class_idx]
                        })
            
            # Save cluster comparison CSV
            cluster_df = pd.DataFrame(cluster_data)
            csv_filename = os.path.join(results_dir, f"cluster_comparison_data.csv")
            cluster_df.to_csv(csv_filename, index=False)
            print(f"Cluster comparison data saved as: {csv_filename}")
            
            plt.show()
            
            # Create combined logits plot (all three types on same diagram)
            plt.figure(figsize=(12, 8))
            
            # Randomly select 100 samples for visualization (to avoid overcrowding)
            if len(q_logits) > 100:
                sample_indices = np.random.choice(len(q_logits), 100, replace=False)
            else:
                sample_indices = np.arange(len(q_logits))
            
            # Plot all three logit types on the same diagram
            for idx in sample_indices:
                # Quantized logits (blue)
                plt.plot(q_logits[idx].numpy(), 'b-', alpha=0.3, linewidth=0.8, label='Quantized' if idx == sample_indices[0] else "")
                # Full-precision logits (green)
                plt.plot(fp_logits[idx].numpy(), 'g-', alpha=0.3, linewidth=0.8, label='Full-Precision' if idx == sample_indices[0] else "")
                # Corrected logits (red)
                plt.plot(corrected_logits[idx].numpy(), 'r-', alpha=0.3, linewidth=0.8, label='Corrected' if idx == sample_indices[0] else "")
            
            plt.title(f'Combined Logits Comparison (α={alpha:.2f}){pca_info}', fontsize=16)
            plt.xlabel('Class Index')
            plt.ylabel('Logit Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the combined logits plot
            filename_combined = os.path.join(results_dir, f"combined_logits.png")
            plt.savefig(filename_combined, dpi=300, bbox_inches='tight')
            print(f"Combined logits plot saved as: {filename_combined}")
            
            # Save combined logits data as CSV
            combined_data = []
            for idx in sample_indices:
                q_vals = q_logits[idx].numpy()
                fp_vals = fp_logits[idx].numpy()
                corr_vals = corrected_logits[idx].numpy()
                
                for class_idx in range(len(q_vals)):
                    combined_data.append({
                        'sample_index': idx,
                        'class_index': class_idx,
                        'quantized_logit': q_vals[class_idx],
                        'fullprecision_logit': fp_vals[class_idx],
                        'corrected_logit': corr_vals[class_idx]
                    })
            
            # Save combined logits CSV
            combined_df = pd.DataFrame(combined_data)
            csv_combined_filename = os.path.join(results_dir, f"combined_logits_data.csv")
            combined_df.to_csv(csv_combined_filename, index=False)
            print(f"Combined logits data saved as: {csv_combined_filename}")
            
            plt.show()
            
            # Create three separate histogram plots
            # 1. Quantized logits histogram
            plt.figure(figsize=(10, 6))
            q_logits_flat = q_logits.flatten().numpy()
            plt.hist(q_logits_flat, bins=100, alpha=0.8, color='blue', edgecolor='black', linewidth=0.5)
            
            # Calculate entropy for quantized logits
            hist_q, bins_q = np.histogram(q_logits_flat, bins=100, density=True)
            hist_q = hist_q[hist_q > 0]  # Remove zero bins for entropy calculation
            entropy_q = -np.sum(hist_q * np.log2(hist_q))
            
            plt.title(f'Quantized Logits Distribution (α={alpha:.2f}){pca_info}\nEntropy: {entropy_q:.4f}', fontsize=16)
            plt.xlabel('Logit Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save the quantized logits histogram
            filename_q_hist = os.path.join(results_dir, f"quantized_histogram.png")
            plt.savefig(filename_q_hist, dpi=300, bbox_inches='tight')
            print(f"Quantized logits histogram saved as: {filename_q_hist}")
            
            # Save quantized logits histogram data as CSV
            hist_q, bins_q = np.histogram(q_logits_flat, bins=100)
            hist_df_q = pd.DataFrame({
                'bin_center': (bins_q[:-1] + bins_q[1:]) / 2,
                'frequency': hist_q,
                'bin_start': bins_q[:-1],
                'bin_end': bins_q[1:],
                'entropy': entropy_q
            })
            csv_q_hist_filename = os.path.join(results_dir, f"quantized_histogram_data.csv")
            hist_df_q.to_csv(csv_q_hist_filename, index=False)
            print(f"Quantized histogram data saved as: {csv_q_hist_filename}")
            print(f"Quantized logits entropy: {entropy_q:.4f}")
            
            plt.show()
            
            # 2. Full-precision logits histogram
            plt.figure(figsize=(10, 6))
            fp_logits_flat = fp_logits.flatten().numpy()
            plt.hist(fp_logits_flat, bins=100, alpha=0.8, color='green', edgecolor='black', linewidth=0.5)
            
            # Calculate entropy for full-precision logits
            hist_fp, bins_fp = np.histogram(fp_logits_flat, bins=100, density=True)
            hist_fp = hist_fp[hist_fp > 0]  # Remove zero bins for entropy calculation
            entropy_fp = -np.sum(hist_fp * np.log2(hist_fp))
            
            plt.title(f'Full-Precision Logits Distribution (α={alpha:.2f}){pca_info}\nEntropy: {entropy_fp:.4f}', fontsize=16)
            plt.xlabel('Logit Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save the full-precision logits histogram
            filename_fp_hist = os.path.join(results_dir, f"fullprecision_histogram.png")
            plt.savefig(filename_fp_hist, dpi=300, bbox_inches='tight')
            print(f"Full-precision logits histogram saved as: {filename_fp_hist}")
            
            # Save full-precision logits histogram data as CSV
            hist_fp, bins_fp = np.histogram(fp_logits_flat, bins=100)
            hist_df_fp = pd.DataFrame({
                'bin_center': (bins_fp[:-1] + bins_fp[1:]) / 2,
                'frequency': hist_fp,
                'bin_start': bins_fp[:-1],
                'bin_end': bins_fp[1:],
                'entropy': entropy_fp
            })
            csv_fp_hist_filename = os.path.join(results_dir, f"fullprecision_histogram_data.csv")
            hist_df_fp.to_csv(csv_fp_hist_filename, index=False)
            print(f"Full-precision histogram data saved as: {csv_fp_hist_filename}")
            print(f"Full-precision logits entropy: {entropy_fp:.4f}")
            
            plt.show()
            
            # 3. Corrected logits histogram
            plt.figure(figsize=(10, 6))
            corrected_logits_flat = corrected_logits.flatten().numpy()
            plt.hist(corrected_logits_flat, bins=100, alpha=0.8, color='red', edgecolor='black', linewidth=0.6)
            
            # Calculate entropy for corrected logits
            hist_corr, bins_corr = np.histogram(corrected_logits_flat, bins=100, density=True)
            hist_corr = hist_corr[hist_corr > 0]  # Remove zero bins for entropy calculation
            entropy_corr = -np.sum(hist_corr * np.log2(hist_corr))
            
            plt.title(f'Corrected Logits Distribution (α={alpha:.2f}){pca_info}\nEntropy: {entropy_corr:.4f}', fontsize=16)
            plt.xlabel('Logit Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save the corrected logits histogram
            filename_corr_hist = os.path.join(results_dir, f"corrected_histogram.png")
            plt.savefig(filename_corr_hist, dpi=300, bbox_inches='tight')
            print(f"Corrected logits histogram saved as: {filename_corr_hist}")
            
            # Save corrected logits histogram data as CSV
            hist_corr, bins_corr = np.histogram(corrected_logits_flat, bins=100)
            hist_df_corr = pd.DataFrame({
                'bin_center': (bins_corr[:-1] + bins_corr[1:]) / 2,
                'frequency': hist_corr,
                'bin_start': bins_corr[:-1],
                'bin_end': bins_corr[1:],
                'entropy': entropy_corr
            })
            csv_corr_hist_filename = os.path.join(results_dir, f"corrected_histogram_data.csv")
            hist_df_corr.to_csv(csv_corr_hist_filename, index=False)
            print(f"Corrected histogram data saved as: {csv_corr_hist_filename}")
            print(f"Corrected logits entropy: {entropy_corr:.4f}")
            
            plt.show()
            
            # Create cluster visualization plots using t-SNE and PCA
            print("Creating cluster visualizations...")
            
            # Prepare data for visualization (use a subset if too large)
            max_samples = 10000  # Limit samples for t-SNE performance
            if len(q_logits) > max_samples:
                sample_indices = np.random.choice(len(q_logits), max_samples, replace=False)
                q_logits_viz = q_logits[sample_indices].numpy()
                fp_logits_viz = fp_logits[sample_indices].numpy()
                corrected_logits_viz = corrected_logits[sample_indices].numpy()
                cluster_ids_viz = cluster_ids[sample_indices]
            else:
                q_logits_viz = q_logits.numpy()
                fp_logits_viz = fp_logits.numpy()
                corrected_logits_viz = corrected_logits.numpy()
                cluster_ids_viz = cluster_ids
            
            # 1. t-SNE visualization of clusters
            try:
                print("Computing t-SNE for cluster visualization...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(q_logits_viz)//4))
                q_logits_tsne = tsne.fit_transform(q_logits_viz)
                
                # Create t-SNE plot
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(q_logits_tsne[:, 0], q_logits_tsne[:, 1], 
                                    c=cluster_ids_viz, cmap='tab20', alpha=0.7, s=20)
                plt.colorbar(scatter, label='Cluster ID')
                plt.title(f'Cluster Visualization using t-SNE (α={alpha:.2f}){pca_info}', fontsize=16)
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.grid(True, alpha=0.3)
                
                # Save t-SNE plot
                tsne_filename = os.path.join(results_dir, f"cluster_visualization_tsne.png")
                plt.savefig(tsne_filename, dpi=300, bbox_inches='tight')
                print(f"t-SNE cluster visualization saved as: {tsne_filename}")
                
                # Save t-SNE data as CSV
                tsne_df = pd.DataFrame({
                    'tsne_component_1': q_logits_tsne[:, 0],
                    'tsne_component_2': q_logits_tsne[:, 1],
                    'cluster_id': cluster_ids_viz,
                    'sample_index': sample_indices if len(q_logits) > max_samples else np.arange(len(q_logits))
                })
                tsne_csv_filename = os.path.join(results_dir, f"cluster_visualization_tsne_data.csv")
                tsne_df.to_csv(tsne_csv_filename, index=False)
                print(f"t-SNE visualization data saved as: {tsne_csv_filename}")
                
                plt.show()
                
            except Exception as e:
                print(f"Error in t-SNE visualization: {e}")
            
            # 2. PCA visualization of clusters (2D)
            try:
                print("Computing PCA for cluster visualization...")
                pca_viz = PCA(n_components=2, random_state=42)
                q_logits_pca = pca_viz.fit_transform(q_logits_viz)
                
                # Create PCA plot
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(q_logits_pca[:, 0], q_logits_pca[:, 1], 
                                    c=cluster_ids_viz, cmap='tab20', alpha=0.7, s=20)
                plt.colorbar(scatter, label='Cluster ID')
                plt.title(f'Cluster Visualization using PCA (α={alpha:.2f}){pca_info}', fontsize=16)
                plt.xlabel(f'PCA Component 1 (Explained Variance: {pca_viz.explained_variance_ratio_[0]:.3f})')
                plt.ylabel(f'PCA Component 2 (Explained Variance: {pca_viz.explained_variance_ratio_[1]:.3f})')
                plt.grid(True, alpha=0.3)
                
                # Save PCA plot
                pca_viz_filename = os.path.join(results_dir, f"cluster_visualization_pca.png")
                plt.savefig(pca_viz_filename, dpi=300, bbox_inches='tight')
                print(f"PCA cluster visualization saved as: {pca_viz_filename}")
                
                # Save PCA data as CSV
                pca_viz_df = pd.DataFrame({
                    'pca_component_1': q_logits_pca[:, 0],
                    'pca_component_2': q_logits_pca[:, 1],
                    'cluster_id': cluster_ids_viz,
                    'sample_index': sample_indices if len(q_logits) > max_samples else np.arange(len(q_logits)),
                    'explained_variance_ratio_1': pca_viz.explained_variance_ratio_[0],
                    'explained_variance_ratio_2': pca_viz.explained_variance_ratio_[1]
                })
                pca_viz_csv_filename = os.path.join(results_dir, f"cluster_visualization_pca_data.csv")
                pca_viz_df.to_csv(pca_viz_csv_filename, index=False)
                print(f"PCA visualization data saved as: {pca_viz_csv_filename}")
                
                plt.show()
                
            except Exception as e:
                print(f"Error in PCA visualization: {e}")
            
            # 3. 3D PCA visualization if enough components
            try:
                if q_logits_viz.shape[1] >= 3:
                    print("Computing 3D PCA for cluster visualization...")
                    pca_3d = PCA(n_components=3, random_state=42)
                    q_logits_pca_3d = pca_3d.fit_transform(q_logits_viz)
                    
                    # Create 3D PCA plot
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(q_logits_pca_3d[:, 0], q_logits_pca_3d[:, 1], q_logits_pca_3d[:, 2], 
                                       c=cluster_ids_viz, cmap='tab20', alpha=0.7, s=20)
                    ax.set_xlabel(f'PCA Component 1 ({pca_3d.explained_variance_ratio_[0]:.3f})')
                    ax.set_ylabel(f'PCA Component 2 ({pca_3d.explained_variance_ratio_[1]:.3f})')
                    ax.set_zlabel(f'PCA Component 3 ({pca_3d.explained_variance_ratio_[2]:.3f})')
                    ax.set_title(f'3D Cluster Visualization using PCA (α={alpha:.2f}){pca_info}', fontsize=16)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID')
                    
                    # Save 3D PCA plot
                    pca_3d_filename = os.path.join(results_dir, f"cluster_visualization_pca_3d.png")
                    plt.savefig(pca_3d_filename, dpi=300, bbox_inches='tight')
                    print(f"3D PCA cluster visualization saved as: {pca_3d_filename}")
                    
                    # Save 3D PCA data as CSV
                    pca_3d_df = pd.DataFrame({
                        'pca_component_1': q_logits_pca_3d[:, 0],
                        'pca_component_2': q_logits_pca_3d[:, 1],
                        'pca_component_3': q_logits_pca_3d[:, 2],
                        'cluster_id': cluster_ids_viz,
                        'sample_index': sample_indices if len(q_logits) > max_samples else np.arange(len(q_logits)),
                        'explained_variance_ratio_1': pca_3d.explained_variance_ratio_[0],
                        'explained_variance_ratio_2': pca_3d.explained_variance_ratio_[1],
                        'explained_variance_ratio_3': pca_3d.explained_variance_ratio_[2]
                    })
                    pca_3d_csv_filename = os.path.join(results_dir, f"cluster_visualization_pca_3d_data.csv")
                    pca_3d_df.to_csv(pca_3d_csv_filename, index=False)
                    print(f"3D PCA visualization data saved as: {pca_3d_csv_filename}")
                    
                    plt.show()
                else:
                    print("Skipping 3D PCA visualization - not enough dimensions")
                    
            except Exception as e:
                print(f"Error in 3D PCA visualization: {e}")
            
        except Exception as e:
            print(f"Error in plotting: {e}")
            import traceback
            traceback.print_exc()
    
    # Extract logits from both models
    print("Extracting logits from quantized and full-precision models...")
    all_q, all_fp = extract_model_logits(qnn, fp_model, train_loader, device)
    
    # Save initial extracted logits for all models
    initial_results_dir = f"initial_logits_{args.arch}_w{args.n_bits_w}bit_a{args.n_bits_a}bit_seed{args.seed}"
    os.makedirs(initial_results_dir, exist_ok=True)
    
    # Convert to list format for the save function
    all_q_list = [all_q]
    all_fp_list = [all_fp]
    all_corrected_list = [all_q]  # Use quantized as placeholder for corrected
    
    save_logits_to_csv(all_q_list, all_fp_list, all_corrected_list, 
                      initial_results_dir, args.arch, args.n_bits_w, args.n_bits_a, args.seed)
    
    print(f"Initial logits saved in: {initial_results_dir}")
    
    # Determine parameter lists for testing
    alpha_list = args.alpha_list if args.alpha_list else [args.alpha]
    num_clusters_list = args.num_clusters_list if args.num_clusters_list else [args.num_clusters]
    pca_dim_list = args.pca_dim_list if args.pca_dim_list else [args.pca_dim]
    
    print(f"Testing combinations:")
    print(f"  Alpha values: {alpha_list}")
    print(f"  Cluster numbers: {num_clusters_list}")
    print(f"  PCA dimensions: {pca_dim_list}")
    
    # Store results
    results = []
    
    # Loop through all parameter combinations
    for alpha in alpha_list:
        for num_clusters in num_clusters_list:
            for pca_dim in pca_dim_list:
                print(f"\n{'='*60}")
                print(f"Testing: alpha={alpha}, clusters={num_clusters}, pca_dim={pca_dim}")
                print(f"{'='*60}")
                
                # Build cluster affine model
                cluster_model, gamma_dict, beta_dict, pca = build_cluster_affine(
                    all_q, all_fp, num_clusters=num_clusters, pca_dim=pca_dim
                )
                
                # Evaluate with current parameters
                top1_acc, top5_acc = evaluate_cluster_affine_with_alpha(
                    qnn, fp_model, cluster_model, gamma_dict, beta_dict, test_loader, device, 
                    pca=pca, alpha=alpha
                )
                
                # Store results
                result = {
                    'alpha': alpha,
                    'num_clusters': num_clusters,
                    'pca_dim': pca_dim,
                    'top1_accuracy': top1_acc,
                    'top5_accuracy': top5_acc
                }
                results.append(result)
                
                print(f"Result: Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
    
    # Print summary of all results
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    print(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Top-1':<10} {'Top-5':<10}")
    print(f"{'-'*50}")
    
    for result in results:
        print(f"{result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
              f"{result['top1_accuracy']:<10.2f} {result['top5_accuracy']:<10.2f}")
    
    # Find best result
    best_result = max(results, key=lambda x: x['top1_accuracy'])
    print(f"\nBEST RESULT:")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  Clusters: {best_result['num_clusters']}")
    print(f"  PCA_dim: {best_result['pca_dim']}")
    print(f"  Top-1 Accuracy: {best_result['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {best_result['top5_accuracy']:.2f}%")
    
    # Create summary CSV of all saved logits files
    print(f"\nCreating logits summary...")
    summary_csv = create_logits_summary_csv(args.arch, args.n_bits_w, args.n_bits_a, args.seed, results)
    if summary_csv:
        print(f"Logits summary created successfully: {summary_csv}")
    else:
        print("Failed to create logits summary")

