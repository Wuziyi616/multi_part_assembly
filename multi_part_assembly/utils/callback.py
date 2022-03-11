import numpy as np
import wandb

import torch

from pytorch_lightning import Callback


class PCAssemblyLogCallback(Callback):
    """Predict part poses and perform visualize the assembly."""

    def __init__(self, sample_num, train_loader, val_loader):
        super().__init__()

        self.sample_num = sample_num
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _sample_assembly(self, trainer, pl_module, dataloader, split):
        num = self.sample_num
        batch = next(iter(dataloader))
        batch = {k: v[:num].to(pl_module.device) for k, v in batch.items()}
        # gt_pcs: [B, N, 6]; pred_pcs: [B, num_samples, N, 6]
        gt_pcs, pred_pcs = pl_module.sample_assembly(batch)
        # offset the GT point clouds to draw in one figure
        for i in range(num):
            gt_pcs[i][:, 0] = gt_pcs[i][:, 0] + 1.5
            for j in range(len(pred_pcs[0])):
                pred_pcs[i][j][:, 0] = pred_pcs[i][j][:, 0] - 1.5 * j
        log_dict = {
            f'{split}_pc_{i}':
            wandb.Object3D(np.concatenate([gt_pcs[i], *pred_pcs[i]], axis=0))
            for i in range(num)
        }
        trainer.logger.experiment.log(log_dict, commit=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        if trainer.logger:
            pl_module.eval()
            self._sample_assembly(trainer, pl_module, self.train_loader,
                                  'train')
            self._sample_assembly(trainer, pl_module, self.val_loader, 'val')
