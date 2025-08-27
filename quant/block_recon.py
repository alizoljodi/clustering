import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock, specials_unquantized
from .adaptive_rounding import AdaRoundQuantizer
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params


class CenterMarginLoss(nn.Module):
    """
    Center loss + inter-class margin (repulsion) for improving clusterability.

    L = lambda_center * (1/B) * sum_i || z_i - c_{y_i} ||_2^2
      + lambda_repel  * sum_{k!=j} max(0, m - ||c_k - c_j||_2)^2 / N_pairs

    Args:
        num_classes (int): number of classes (K).
        feat_dim (int): feature dimension (d).
        lambda_center (float): weight for center (within-cluster) term.
        lambda_repel (float): weight for repulsion (between-centers) term.
        margin (float): target minimum distance between any two centers.
        center_lr (float): step size for updating centers (EMA-style).
        normalize (bool): if True, L2-normalize features & centers before loss.
        init_std (float): std for random center initialization.
        device, dtype: optional placement.
    """
    def __init__(self,
                 num_classes: int,
                 feat_dim: int,
                 lambda_center: float = 1.0,
                 lambda_repel: float = 0.1,
                 margin: float = 1.0,
                 center_lr: float = 0.5,
                 normalize: bool = False,
                 init_std: float = 0.01,
                 device=None,
                 dtype=torch.float32):
        super().__init__()
        self.K = int(num_classes)
        self.d = int(feat_dim)
        self.lambda_center = float(lambda_center)
        self.lambda_repel  = float(lambda_repel)
        self.margin = float(margin)
        self.center_lr = float(center_lr)
        self.normalize = bool(normalize)

        # Centers are parameters we update manually (not via the main optimizer).
        centers = torch.randn(self.K, self.d, device=device, dtype=dtype) * init_std
        self.register_buffer("centers", centers)

        # For lazy init (optional): mark classes we have seen at least once
        self.register_buffer("seen_mask", torch.zeros(self.K, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _lazy_init_centers(self, z, y):
        """
        Initialize centers of unseen classes from the current batch means.
        """
        for k in y.unique().tolist():
            k = int(k)
            mask = (y == k)
            if mask.any() and not self.seen_mask[k]:
                self.centers[k] = z[mask].mean(dim=0)
                self.seen_mask[k] = True

    def _update_centers(self, z, y):
        """
        Online center update (like an EMA over batch samples).
        Δc_k = mean(c_k - z_i) for i in class k
        c_k <- c_k - center_lr * Δc_k
        """
        with torch.no_grad():
            for k in y.unique().tolist():
                k = int(k)
                zk = z[y == k]
                if zk.numel() == 0:
                    continue
                ck = self.centers[k]
                delta = (ck - zk).mean(dim=0)
                self.centers[k] = ck - self.center_lr * delta
                self.seen_mask[k] = True

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        """
        Args:
            z: [B, d] features (e.g., penultimate layer or logits—you choose).
            y: [B] int labels in [0, K-1] (true or pseudo-labels).

        Returns:
            total_loss, dict(stats)
        """
        assert z.dim() == 2 and z.size(1) == self.d, "z must be [B, d]"
        assert y.dim() == 1 and y.size(0) == z.size(0), "y must be [B]"

        if self.normalize:
            z = F.normalize(z, dim=1)
            # normalize centers for a fair cosine-space comparison
            centers_norm = F.normalize(self.centers, dim=1)
        else:
            centers_norm = self.centers

        # Lazy init unseen centers from this batch
        self._lazy_init_centers(z, y)

        B = z.size(0)

        # ----- Center (within-cluster) loss -----
        c_y = centers_norm[y]                    # [B, d]
        center_loss = (z - c_y).pow(2).sum(dim=1).mean()  # average over batch
        # Optionally normalize by d (kept simple here):
        # center_loss = center_loss / self.d

        # ----- Repulsion (between-centers) loss -----
        # Compute on centers of classes present in this batch (stable & fast)
        present = y.unique()
        C = centers_norm[present]                # [Kb, d]
        if C.size(0) > 1:
            # pairwise distances
            D = torch.cdist(C, C, p=2)          # [Kb, Kb]
            # mask out diagonal
            infdiag = torch.full_like(D, float('inf'))
            D = torch.where(torch.eye(D.size(0), device=D.device, dtype=torch.bool), infdiag, D)
            # hinge on margin
            repel = torch.clamp(self.margin - D, min=0.0).pow(2)
            # average over k!=j pairs
            repel_loss = repel[repel.isfinite()].mean()
        else:
            repel_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        total = self.lambda_center * center_loss + self.lambda_repel * repel_loss

        # ----- Update centers after computing gradients wrt features -----
        # (Call backward on `total` first in your training loop, then call
        #  `loss_obj.update()` OR set update_mode='post_backward' and call it outside.)
        # Here we update immediately; if you prefer post-backward, split this call.
        self._update_centers(z.detach(), y.detach())

        stats = {
            "center_loss": center_loss.detach(),
            "repel_loss": repel_loss.detach(),
            "num_present_classes": present.numel()
        }
        return total, stats

include = False
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

def block_reconstruction(model: QuantModel, fp_model: QuantModel, block: BaseQuantBlock, fp_block: BaseQuantBlock,
                        cali_data: torch.Tensor, batch_size: int = 32, iters: int = 20000, weight: float = 0.01, 
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5,
                        input_prob: float = 1.0, keep_gpu: bool = True, 
                        lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02,
                        num_clusters: int = 64, pca_dim: int = None, lambda_center: float = 0.1):
    """
    Reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param lamb_r: hyper-parameter for regularization
    :param T: temperature coefficient for KL divergence
    :param bn_lr: learning rate for DC
    :param lamb_c: hyper-parameter for DC
    :param num_clusters: number of clusters for CenterMarginLoss (default: 64)
    :param pca_dim: PCA dimension for clustering (default: None)
    :param lambda_center: weight for center loss (0.0 to 1.0, default: 0.1)
    """

    '''get input and set scale'''
    cached_inps = get_init(model, block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=keep_gpu)
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c)
    set_act_quantize_params(block, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    '''set state'''
    cur_weight, cur_act = True, True
    
    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    block.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=module.org_weight.data)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.act_quantizer.delta is not None:
                module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                a_para += [module.act_quantizer.delta]
            '''set up drop'''
            module.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)

    loss_mode = 'relaxation'
    rec_loss = opt_mode
    
    # Determine feature dimension for CenterMarginLoss
    # Use PCA dimension if specified, otherwise use the output dimension of the block
    if pca_dim is not None:
        feat_dim = pca_dim
    else:
        # Get the output dimension from the block's output shape
        with torch.no_grad():
            sample_input = torch.randn(1, *cached_inps.shape[1:]).to(cached_inps.device)
            sample_output = block(sample_input)
            feat_dim = sample_output.shape[1] if len(sample_output.shape) > 1 else sample_output.shape[0]
    
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T,
                             num_clusters=num_clusters, pca_dim=pca_dim, feat_dim=feat_dim, lambda_center=lambda_center)
    device = 'cuda'
    sz = cached_inps.size(0)
    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = cached_inps[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        output_fp = cached_output[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        
        cur_inp = torch.cat((drop_inp, cur_inp))

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        
        out_all = block(cur_inp)

        '''forward for prediction difference'''
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]
        output = out_quant
        for num, module in enumerate(module_list):
            # for ResNet and RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # for MobileNet and MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)
        err = loss_func(out_drop, cur_out, output, output_fp)

        err.backward(retain_graph=True)
        if w_opt:
            w_opt.step()    
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    for module in block.modules():
        if isinstance(module, QuantModule):
            '''weight '''
            module.weight_quantizer.soft_targets = False
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
            module.trained = True
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True

class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0,
                 num_clusters: int = 64,
                 pca_dim: int = None,
                 feat_dim: int = None,
                 lambda_center: float = 0.1):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.feat_dim = feat_dim
        self.lambda_center = lambda_center

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')
        
        # Initialize CenterMarginLoss for potential future use
        # Note: Currently not used in loss computation as it requires labels/cluster assignments
        if self.feat_dim is not None:
            self.center_loss = CenterMarginLoss(num_classes=self.num_clusters, feat_dim=self.feat_dim,
                               lambda_center=self.lambda_center,
                               lambda_repel=0.1,
                               margin=1.0,
                               center_lr=0.5,
                               normalize=False)
        else:
            self.center_loss = None

    def __call__(self, pred, tgt, output, output_fp):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy, pd_loss is the 
        prediction difference loss.

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param output: prediction from quantized model
        :param output_fp: prediction from FP model
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        # Calculate center loss if available
        center_loss_val = 0.0
        if self.center_loss is not None and self.lambda_center > 0.0:
            # For center loss, we need cluster assignments (labels)
            # Since we don't have them during reconstruction, we'll use a simple approach
            # or skip the center loss component
            try:
                # Use output features as cluster assignments (simple approach)
                # This is a placeholder - in practice you'd want proper cluster assignments
                cluster_assignments = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
                center_loss_val = self.center_loss(output, cluster_assignments)[0]  # Get the loss value
            except Exception as e:
                # If center loss fails, set it to 0
                center_loss_val = 0.0
                if self.count % 500 == 0:
                    print(f"Warning: Center loss computation failed: {e}")

        total_loss = rec_loss + round_loss + pd_loss + self.lambda_center * center_loss_val
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, pd:{:.3f}, round:{:.3f}, center:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(pd_loss), float(round_loss), float(center_loss_val), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
