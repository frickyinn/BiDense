import torch
from torchvision import transforms
import lightning as L

from models import DEPTH_MODEL_DICT
from misc import Silog_loss, compute_metrics


def normalize_result(value, vmin=None, vmax=None):
    value = value[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return value.unsqueeze(0)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class PL_DepthTrainer(L.LightningModule):
    def __init__(
            self, 
            model_type, 
            binary_type, 
            dataset, 
            variance_focus, 
            max_lr, 
            epochs, 
            garg_crop,
            eigen_crop,
            min_depth_eval, 
            max_depth_eval, 
            **kwargs):
        super().__init__()
        self.module = DEPTH_MODEL_DICT[model_type][binary_type](**kwargs)
        self.criterion = Silog_loss(variance_focus)
        
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        image = batch['image']
        depth_gt = batch['depth']

        depth_est = self.module(image)
        if self.hparams.dataset == 'nyu':
            mask = depth_gt > 0.1
        else:
            mask = depth_gt > 1.0
        
        loss = self.criterion(depth_est, depth_gt, mask.bool())
        # silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = compute_metrics(
        #     depth_gt, depth_est, 
        #     garg_crop=self.hparams.garg_crop, eigen_crop=self.hparams.eigen_crop, 
        #     dataset=self.hparams.dataset,
        #     min_depth_eval=self.hparams.min_depth_eval, max_depth_eval=self.hparams.max_depth_eval
        # )
        self.log_dict({
            'train/loss': loss,
            # 'train/d1': d1, 'train/d2': d2, 'train/d3': d3,
            # 'train/abs_rel': abs_rel, 'train/rms': rms, 'train/log10': log10,
            # 'train/silog': silog, 'train/sq_rel': sq_rel, 'train/log_rms': log_rms,
        }, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        depth_gt = batch['depth']

        depth_est = self.module(image)
        if self.hparams.dataset == 'nyu':
            mask = depth_gt > 0.1
        else:
            mask = depth_gt > 1.0
        
        loss = self.criterion(depth_est, depth_gt, mask.bool())
        silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = compute_metrics(
            depth_gt, depth_est, 
            garg_crop=self.hparams.garg_crop, eigen_crop=self.hparams.eigen_crop, 
            dataset=self.hparams.dataset,
            min_depth_eval=self.hparams.min_depth_eval, max_depth_eval=self.hparams.max_depth_eval
        )
        self.log_dict({
            'valid/loss': loss,
            'valid/d1': d1, 'valid/d2': d2, 'valid/d3': d3,
            'valid/abs_rel': abs_rel, 'valid/rms': rms, 'valid/log10': log10,
            'valid/silog': silog, 'valid/sq_rel': sq_rel, 'valid/log_rms': log_rms,
        }, on_epoch=True, sync_dist=True, batch_size=image.size(0))

        self.depth_gt = depth_gt
        self.depth_est = depth_est
        self.image = image

    def test_step(self, batch, batch_idx):
        image = batch['image']
        depth_gt = batch['depth']

        depth_est = self.module(image)
        if self.hparams.dataset == 'nyu':
            mask = depth_gt > 0.1
        else:
            mask = depth_gt > 1.0
        
        loss = self.criterion(depth_est, depth_gt, mask.bool())
        silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = compute_metrics(
            depth_gt, depth_est, 
            garg_crop=self.hparams.garg_crop, eigen_crop=self.hparams.eigen_crop, 
            dataset=self.hparams.dataset,
            min_depth_eval=self.hparams.min_depth_eval, max_depth_eval=self.hparams.max_depth_eval
        )
        self.log_dict({
            'test/loss': loss,
            'test/d1': d1, 'test/d2': d2, 'test/d3': d3,
            'test/abs_rel': abs_rel, 'test/rms': rms, 'test/log10': log10,
            'test/silog': silog, 'test/sq_rel': sq_rel, 'test/log_rms': log_rms,
        }, on_epoch=True, sync_dist=True, batch_size=image.size(0), logger=False)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        self.depth_gt = torch.where(self.depth_gt < 1e-3, self.depth_gt * 0 + 1e3, self.depth_gt)
        self.logger.experiment.add_image('valid_viz/depth_gt', normalize_result(1 / self.depth_gt[0, :, :, :].data), self.global_step)
        self.logger.experiment.add_image('valid_viz/depth_est', normalize_result(1 / self.depth_est[0, :, :, :].data), self.global_step)
        self.logger.experiment.add_image('valid_viz/image', inv_normalize(self.image[0, :, :, :]).data, self.global_step)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.hparams.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.max_lr, steps_per_epoch=1, epochs=self.hparams.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
