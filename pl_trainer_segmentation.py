import torch
import lightning as L

from models import SEGMENTATION_MODEL_DICT
from misc import inv_normalize, compute_segmentation_metrics, visualize_segmentation_result


class PL_SegmentationTrainer(L.LightningModule):
    def __init__(
            self, 
            model_type, 
            binary_type,
            dataset,
            max_lr, 
            epochs, 
            **kwargs):
        super().__init__()
        self.module = SEGMENTATION_MODEL_DICT[model_type][binary_type](**kwargs)
        weigth = None
        if dataset == 'ade20k' or dataset == 'pascal_voc':
            weight = torch.ones(kwargs['num_classes'], dtype=torch.float32)
            weight[-1] = 0
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        
        self.save_hyperparameters()
        self.n_classes = kwargs['num_classes'] - 1 if (dataset == 'ade20k' or dataset == 'pascal_voc') else kwargs['num_classes']
        self.train_total_inter, self.train_total_union = 0, 0
        self.valid_total_inter, self.valid_total_union = 0, 0
        self.test_total_inter, self.test_total_union = 0, 0

    def training_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']
        pred = self.module(image)
        
        loss = self.criterion(pred, mask_gt)
        # pixAcc, area_inter, area_union = compute_segmentation_metrics(mask_gt, pred, self.n_classes)
        # self.train_total_inter += area_inter
        # self.train_total_union += area_union
        self.log_dict({
            'train/loss': loss,
            # 'train/pixAcc': pixAcc,
        }, sync_dist=True, batch_size=image.size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']
        pred = self.module(image)
        
        loss = self.criterion(pred, mask_gt)
        pixAcc, area_inter, area_union = compute_segmentation_metrics(mask_gt, pred, self.n_classes)
        self.valid_total_inter += area_inter
        self.valid_total_union += area_union
        self.log_dict({
            'valid/loss': loss,
            'valid/pixAcc': pixAcc,
        }, on_epoch=True, sync_dist=True, batch_size=image.size(0))

        self.mask_gt = mask_gt
        self.mask_est = pred
        self.image = image

    def test_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']
        pred = self.module(image)
        
        loss = self.criterion(pred, mask_gt)
        pixAcc, area_inter, area_union = compute_segmentation_metrics(mask_gt, pred, self.n_classes)
        self.test_total_inter += area_inter
        self.test_total_union += area_union
        self.log_dict({
            'test/loss': loss,
            'test/pixAcc': pixAcc,
        }, on_epoch=True, sync_dist=True, batch_size=image.size(0))

    def on_train_epoch_end(self):
        # self.train_total_union[self.train_total_union == 0] = torch.inf
        # mIoU = (self.train_total_inter / self.train_total_union).mean()
        # self.log('train/mIoU', mIoU, sync_dist=True)
        # self.train_total_inter, self.train_total_union = 0, 0
        pass

    def on_validation_epoch_end(self):
        self.valid_total_union[self.valid_total_union == 0] = torch.inf
        mIoU = (self.valid_total_inter / self.valid_total_union).mean()
        self.log('valid/mIoU', mIoU, sync_dist=True)
        self.valid_total_inter, self.valid_total_union = 0, 0
        
        self.logger.experiment.add_image('valid_viz/mask_gt', visualize_segmentation_result(self.mask_gt[0, :, :].data, self.hparams.dataset), self.global_step)
        self.logger.experiment.add_image('valid_viz/mask_est', visualize_segmentation_result(self.mask_est[0, :, :, :].argmax(0).data, self.hparams.dataset), self.global_step)
        self.logger.experiment.add_image('valid_viz/image', inv_normalize(self.image[0, :, :, :]).data, self.global_step)

    def on_test_epoch_end(self):
        self.test_total_union[self.test_total_union == 0] = torch.inf
        mIoU = (self.test_total_inter / self.test_total_union).mean()
        self.log('test/mIoU', mIoU, sync_dist=True)
        self.test_total_inter, self.test_total_union = 0, 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.hparams.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.max_lr, steps_per_epoch=1, epochs=self.hparams.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
