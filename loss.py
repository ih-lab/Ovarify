import torch
from torch import nn
from utils import *
import segmentation_models_pytorch as smp


class ComboLoss(nn.Module):
    """Combination of BCE and Dice losses
    from: https://arxiv.org/pdf/1805.02798.pdf
    code: https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
    """
    def __init__(self, alpha=0.75, weight=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha # < 0.5 penalizes FP more, > 0.5 penalizes FN more
        self.weight = weight # proportion of BCE loss in overall loss
    
    def forward(self, y_pred, y_true, smooth=1, eps=1e-7):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        y_pred = torch.clamp(y_pred, eps, 1. - eps)
        out = - (self.alpha * ((y_true * torch.log(y_pred)) + ((1 - self.alpha) * (1.0 - y_true) * torch.log(1.0 - y_pred))))
        weighted_ce = out.mean(-1)
        
        return (self.weight * weighted_ce) - ((1 - self.weight) * dice)

class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.8, beta=0.3, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 1
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

def cosh_dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    fin_loss = torch.log((torch.exp(loss) + torch.exp(-loss))/2.0)

    return fin_loss.mean()

def dice_loss(pred, target, smooth = 1e-3):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))).mean()
    
    return loss

def shape_loss(pred, target_sdm, smooth=1e-5): 
    """
    from: https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py#L95
    paper: https://arxiv.org/pdf/1912.03849.pdf
    """
    intersect = torch.sum(pred * target_sdm)
    pred_sum = torch.sum(pred ** 2)
    target_sum = torch.sum(target_sdm ** 2)
    l_product = (intersect + smooth) / (intersect + pred_sum + target_sum + smooth)
    loss = - l_product + torch.norm(pred - target_sdm, 1)/torch.numel(pred)

    return loss

def l2_focal_loss(pred, target, gamma=1, alpha=0.1, beta=0.02, theta=0.01):
    """
    from: https://github.com/PopicLab/cue/blob/master/models/cue_net.py (Cue, Popic et al 2023)
    adapted from: SimplePose, Li et al 2020
    """
    dkt = torch.where(torch.ge(target, theta), pred - alpha, 1 - pred - beta)
    factor = torch.abs(1. - dkt) ** gamma
    lkt = (pred - target) ** 2 * factor
    fl = lkt.sum(dim=(1, 2, 3))

    return fl.mean()


def calc_loss(pred, target, target_sdm, classif, metrics, loss_type, bce_weight=torch.tensor(1), model_type='seg'):
    
    pred_sig = torch.sigmoid(pred)
    
    if model_type == 'seg':
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
        bce = bce_loss(pred, target)
        btl_loss = BinaryTverskyLossV2()
        btl = btl_loss(pred, target)
        dice = dice_loss(pred_sig, target)
        cdl = cosh_dice_loss(pred_sig, target)
        combo_loss = ComboLoss()
        combo = combo_loss(pred_sig, target)
        jaccard_loss = smp.losses.JaccardLoss(mode='binary', from_logits=False, log_loss=False)
        jaccard = jaccard_loss(pred_sig, target)
        shape = (10*shape_loss(pred, target_sdm)) + dice_loss(torch.sigmoid(-1500*pred), target)
        l2_focal = l2_focal_loss(pred_sig, target)
        combo_jac_foc = jaccard_loss(pred_sig, target) + (l2_focal_loss(pred_sig, target))

        metrics['cosh_dice'] += cdl.data.cpu().numpy() * target.size(0) 
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['binary_tversky_v2'] += btl.data.cpu().numpy() * target.size(0)
        metrics['combo'] += combo.data.cpu().numpy() * target.size(0)
        metrics['shape'] += shape.data.cpu().numpy() * target.size(0)
        metrics['jaccard'] += jaccard.data.cpu().numpy() * target.size(0)
        metrics['l2_focal'] += l2_focal.data.cpu().numpy() * target.size(0)
        metrics['combo_jaccard_l2focal'] += combo_jac_foc.data.cpu().numpy() * target.size(0)

    elif model_type == 'class':
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
        bce = bce_loss(pred, classif)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)

    if loss_type == 'cosh_dice':
        return cdl*10
    if loss_type == 'bce':
        return bce
    if loss_type == 'dice':
        return dice
    if loss_type == 'binary_tversky_v2':
        return btl
    if loss_type == 'combo':
        return combo
    if loss_type == 'shape':
        return shape
    if loss_type == 'jaccard':
        return jaccard
    if loss_type == 'l2_focal':
        return l2_focal
    if loss_type =='combo_jaccard_l2focal':
        return combo_jac_foc

    return bce