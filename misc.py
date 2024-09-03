import os
import numpy as np
import random
import torch
from torch import nn


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(Silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask] + 1e-7) - torch.log(depth_gt[mask] + 1e-7)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def compute_errors(gt, pred):
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


def compute_metrics(gt, pred, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10):
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
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])
