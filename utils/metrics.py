import torch
import torch.nn.functional as F

def bce_with_logits(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

@torch.no_grad()
def per_channel_iou_from_logits(logits, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum(dim=(0,2,3))
    union = ((pred + target) > 0).float().sum(dim=(0,2,3))
    iou = (inter + eps) / (union + eps)
    return iou
