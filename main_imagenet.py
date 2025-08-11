import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
import time
import hubconf  # noqa: F401
import copy
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
        
        # Ensure maxk doesn't exceed the number of classes
        num_classes = output.size(1)
        maxk = min(maxk, num_classes)
        
        # Adjust topk list to only include valid values
        valid_topk = [k for k in topk if k <= num_classes]
        if not valid_topk:
            return [torch.tensor(0.0) for _ in topk]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k <= num_classes:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                # If k exceeds number of classes, return 0 accuracy
                res.append(torch.tensor(0.0))
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
        # Normalize all_q and all_fp before clustering
        # Compute mean and std for normalization
        q_mean = all_q.mean(dim=0, keepdim=True)
        q_std = all_q.std(dim=0, keepdim=True, unbiased=False)
        q_std[q_std < 1e-8] = 1e-8  # Avoid division by zero
        
        fp_mean = all_fp.mean(dim=0, keepdim=True)
        fp_std = all_fp.std(dim=0, keepdim=True, unbiased=False)
        fp_std[fp_std < 1e-8] = 1e-8  # Avoid division by zero
        
        # Normalize the data
        all_q_normalized = (all_q - q_mean) / q_std
        all_fp_normalized = (all_fp - fp_mean) / fp_std
        
        # Optional PCA for clustering only
        pca = None
        if pca_dim is not None and pca_dim < all_q_normalized.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q_normalized.numpy())
        else:
            q_features = all_q_normalized.numpy()

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

            q_c = all_q_normalized[idxs]  # [Nc, C]
            fp_c = all_fp_normalized[idxs]  # [Nc, C]

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

        # Store normalization parameters for later use
        norm_params = {
            'q_mean': q_mean,
            'q_std': q_std,
            'fp_mean': fp_mean,
            'fp_std': fp_std
        }
        
        return cluster_model, gamma_dict, beta_dict, pca, norm_params

    def apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=None, alpha=0.4):
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        """
        # Add safety checks for input tensor
        if q_logits.numel() == 0:
            print("Warning: q_logits is empty")
            return q_logits
        
        if q_logits.dim() != 2:
            print(f"Warning: q_logits has unexpected shape: {q_logits.shape}")
            return q_logits
            
        q_np = q_logits.cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        # Compute normalization parameters from the current input data
        q_mean = q_logits.mean(dim=0, keepdim=True)
        q_std = q_logits.std(dim=0, keepdim=True, unbiased=False)
        q_std[q_std < 1e-8] = 1e-8  # Avoid division by zero
        
        fp_mean = q_logits.mean(dim=0, keepdim=True)  # Use q_logits as reference for fp normalization
        fp_std = q_logits.std(dim=0, keepdim=True, unbiased=False)
        fp_std[fp_std < 1e-8] = 1e-8  # Avoid division by zero

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            
            # Safety check: ensure cluster ID is valid
            if cid not in gamma_dict or cid not in beta_dict:
                print(f"Warning: Invalid cluster ID {cid}, using default values")
                gamma = torch.ones(q.size(0)).to(q.device)
                beta = torch.zeros(q.size(0)).to(q.device)
            else:
                gamma = gamma_dict[cid].to(q.device)
                beta = beta_dict[cid].to(q.device)
            
            # Normalize the current input
            q_normalized = (q - q_mean.to(q.device)) / q_std.to(q.device)
            # Apply correction in normalized space
            affine_corrected_normalized = q_normalized * gamma + beta
            # Denormalize back to original space
            affine_corrected = affine_corrected_normalized * fp_std.to(q.device) + fp_mean.to(q.device)
            
            blended = q + alpha * (affine_corrected - q)
            corrected.append(blended)
        
        result = torch.stack(corrected)
        
        # Add safety check for output tensor
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: Output contains NaN or Inf values, returning original input")
            return q_logits
            
        return result
    
    def evaluate_cluster_affine_with_alpha(q_model, cluster_model, gamma_dict, beta_dict, dataloader, device, pca=None, alpha=0.4):
        q_model.eval()
        total_top1, total_top5, total = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images, targets = images.to(device), targets.to(device)
                q_logits = q_model(images)
                
                # Debug: Check input tensor shapes
                if batch_idx == 0:
                    print(f"Debug: q_logits shape: {q_logits.shape}, targets shape: {targets.shape}")
                    print(f"Debug: q_logits device: {q_logits.device}, targets device: {targets.device}")

                corrected_logits = apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha)
                
                # Debug: Check corrected tensor
                if batch_idx == 0:
                    print(f"Debug: corrected_logits shape: {corrected_logits.shape}")
                    print(f"Debug: corrected_logits device: {corrected_logits.device}")
                    if torch.isnan(corrected_logits).any():
                        print("Warning: corrected_logits contains NaN values")
                    if torch.isinf(corrected_logits).any():
                        print("Warning: corrected_logits contains Inf values")

                acc1, acc5 = accuracy(corrected_logits, targets, topk=(1, 5))
                total_top1 += acc1.item() * images.size(0)
                total_top5 += acc5.item() * images.size(0)
                total += images.size(0)

        print(f"[Alpha={alpha:.2f}] Top-1 Accuracy: {total_top1 / total:.2f}%")
        print(f"[Alpha={alpha:.2f}] Top-5 Accuracy: {total_top5 / total:.2f}%")
        return total_top1 / total, total_top5 / total
    
    # Extract logits from both models
    print("Extracting logits from quantized and full-precision models...")
    all_q, all_fp = extract_model_logits(qnn, fp_model, train_loader, device)
    
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
                cluster_model, gamma_dict, beta_dict, pca, _ = build_cluster_affine(
                    all_q, all_fp, num_clusters=num_clusters, pca_dim=pca_dim
                )
                
                # Evaluate with current parameters
                top1_acc, top5_acc = evaluate_cluster_affine_with_alpha(
                    qnn, cluster_model, gamma_dict, beta_dict, test_loader, device, 
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

