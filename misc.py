import os
import numpy as np
import random
import torch
from torch import nn
from torchvision import transforms


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class Silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(Silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask] + 1e-7) - torch.log(depth_gt[mask] + 1e-7)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def _compute_depth_errors(gt, pred):
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    rms = (gt - pred) ** 2
    rms = torch.sqrt(rms.mean())

    log_rms = (torch.log(gt) - torch.log(pred)) ** 2
    log_rms = torch.sqrt(log_rms.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


@torch.no_grad()
def compute_depth_metrics(gt, pred, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10):
    pred = pred.squeeze()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze()
    valid_mask = torch.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = torch.zeros(valid_mask.shape, device=valid_mask.device)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = torch.ones(valid_mask.shape)
    valid_mask = torch.logical_and(valid_mask, eval_mask)
    return _compute_depth_errors(gt_depth[valid_mask], pred[valid_mask])


@torch.no_grad()
def normalize_depth_result(value, vmin=None, vmax=None):
    value = value[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return value.unsqueeze(0)


@torch.no_grad()
def compute_segmentation_metrics(mask_gt, mask_est, n_classes):
    mask_est = torch.argmax(mask_est, dim=1, keepdim=True)
    mask_gt, mask_est = mask_gt[mask_gt > 0], mask_est[mask_gt > 0]
    pixAcc = (mask_gt == mask_est).float().mean()

    inter = mask_est * (mask_est == mask_gt)
    area_inter = torch.histogram(inter, bins=n_classes-1, range=(1, n_classes-1))
    area_est = torch.histogram(mask_est, bins=n_classes-1, range=(1, n_classes-1))
    area_gt = torch.histogram(mask_gt, bins=n_classes-1, range=(1, n_classes-1))

    area_union = area_est + area_gt - area_inter
    return pixAcc, area_inter, area_union


ADE20K_PALETTE = [
    (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50), (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255), (230, 230, 230), (4, 250, 7), 
    (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82), (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3), 
    (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), 
    (8, 255, 214), (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), 
    (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), 
    (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), 
    (11, 200, 200), (255 ,82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), 
    (0, 143, 255), (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255), (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), 
    (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20), (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194), 
    (0, 255, 82), (0, 10, 255), (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255), (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0), 
    (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0), (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255), (255, 0, 31), (0, 184, 255), 
    (0, 214, 255), (255, 0, 112), (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160), (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163), 
    (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0), (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0), (10, 190, 212), (214, 255, 0), 
    (0, 204, 255), (20, 0, 255), (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204), (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255), 
    (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255), (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194), (102, 255, 0), (92, 0, 255)
]


def visualize_segmentation_result(mask, dataset):
    # mask: (C, H, W)
    _, H, W = mask.shape

    if dataset == 'ade20k':
        palette = ADE20K_PALETTE
    
    palette = torch.Tensor(palette, device=mask.device)
    colorized = torch.index_select(mask.argmax(0).reshape(-1), 0, palette).reshape(H, W, 3)
    return colorized
