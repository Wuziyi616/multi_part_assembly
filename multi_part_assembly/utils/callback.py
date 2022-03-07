import numpy as np
import wandb

import torch

from pytorch_lightning import Callback


class PCAssemblyLogCallback(Callback):
    """Predict part poses and perform visualize the assembly."""

    def __init__(self, cfg, val_loader):
        super().__init__()

        self.cfg = cfg
        self.val_loader = val_loader

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        if trainer.logger:
            num = self.cfg.exp.val_sample_vis
            batch = next(iter(self.val_loader))
            batch = {k: v[:num].to(pl_module.device) for k, v in batch.items()}
            pl_module.eval()
            gt_pcs, pred_pcs = pl_module.sample_assembly(batch)  # [num, N, 3]
            gt_pcs += 1.5  # offset the GT point cloud to draw in one figure
            pcs = np.concatenate([gt_pcs, pred_pcs], axis=1)  # [num, 2N, 3]
            log_dict = {
                f'pc_{i}': wandb.Object3D(pc)
                for i, pc in enumerate(pcs)
            }
            trainer.logger.experiment.log(log_dict, commit=True)
