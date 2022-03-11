import torch
import torch.nn as nn
import torch.optim as optim

from multi_part_assembly.models.encoder import build_encoder
from multi_part_assembly.utils.transforms import qtransform
from multi_part_assembly.utils.lr import CosineAnnealingWarmupRestarts

from .network import PNTransformer


class ShapeDiscriminator(nn.Module):

    def __init__(self, encoder_arch, feat_dim):
        super().__init__()

        self.encoder = build_encoder(
            encoder_arch, feat_dim=feat_dim, global_feat=True)
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.encoder(x)  # [B, C]
        pred = self.classifier(feats)  # [B, 1]
        return pred.squeeze(-1)  # [B]


class PNTransformerGAN(PNTransformer):
    """Baseline multi-part assembly model with discriminator.

    Encoder: PointNet extracting per-part global point cloud features
    Correlator: TransformerEncoder perform part interactions
    Predictor: MLP-based pose predictor
    Discriminator: PointNet based classifier
    """

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.discriminator = self._init_discriminator()
        self.d_npoint = self.cfg.model.discriminator_num_points

        # loss configs
        adv_loss = self.cfg.model.discriminator_loss
        assert adv_loss in ['mse', 'ce']
        self.adv_loss_fn = nn.MSELoss() if \
            adv_loss == 'mse' else nn.BCEWithLogitsLoss()

    def _init_discriminator(self):
        discriminator = ShapeDiscriminator(self.cfg.model.discriminator,
                                           self.cfg.model.pc_feat_dim)
        return discriminator

    @staticmethod
    def _sample_points(part_pcs, valids, sample_num):
        """Sample N points from valid parts to produce a shape point cloud.

        Args:
            part_pcs: [B, P, N, 3]
            valids: [B, P], 1 is valid, 0 is padded
            N: int
        """
        B, P, N, _ = part_pcs.shape
        part_pcs = part_pcs.flatten(1, 2)  # [B, P*N, 3]
        # in case `valids` == [1., 1., ..., 1.] (all_ones)
        valids = torch.cat([valids, torch.zeros(B, 1).type_as(valids)], dim=1)
        num_valid_parts = valids.argmin(1)  # find the first `0` in `valids`
        all_idx = torch.stack([
            torch.randperm(num_valid_parts[i] * N)[:sample_num]
            for i in range(B)
        ]).type_as(num_valid_parts)  # [B, num_samples]
        batch_idx = torch.arange(B)[:, None].type_as(all_idx)
        pcs = part_pcs[batch_idx, all_idx]  # [B, num_samples, 3]
        return pcs

    def _loss_function(self, data_dict, pre_pose_feats=None, optimizer_idx=-1):
        """Inner loop for sampling loss computation.

        Besides the translation and rotation loss, also compute the GAN loss.
        """
        if optimizer_idx == -1:  # in eval mode
            assert not self.training
            return super()._loss_function(data_dict, pre_pose_feats)

        batch_size = data_dict['part_pcs'].shape[0]
        if optimizer_idx == 0:  # g step
            loss_dict, out_dict = super()._loss_function(
                data_dict, pre_pose_feats)
            real_pts = out_dict['pred_trans_pts']  # [B, P, N, 3]
            real_pts = self._sample_points(real_pts, data_dict['part_valids'],
                                           self.d_npoint)  # [B, n, 3]
            real_logits = self.discriminator(real_pts)  # [B]
            real = torch.ones(batch_size).type_as(real_pts).detach()
            g_loss = self.adv_loss_fn(real_logits, real)
            loss_dict.update({'g_loss': g_loss})
            return loss_dict, out_dict

        assert optimizer_idx == 1  # d step
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        instance_label = data_dict['instance_label']

        # generate
        forward_dict = {
            'part_pcs': part_pcs,
            'part_valids': valids,
            'instance_label': instance_label,
            'pre_pose_feats': pre_pose_feats,
        }
        with torch.no_grad():
            out_dict = self.forward(forward_dict)

        pred_trans, pred_quat = out_dict['trans'], out_dict['quat']
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']
        pred_pts = qtransform(pred_trans, pred_quat, part_pcs).detach()
        pred_pts = self._sample_points(pred_pts, valids, self.d_npoint)
        gt_pts = qtransform(gt_trans, gt_quat, part_pcs).detach()
        gt_pts = self._sample_points(gt_pts, valids, self.d_npoint)
        real = torch.ones(batch_size).type_as(part_pcs).detach()
        fake = torch.zeros(batch_size).type_as(part_pcs).detach()

        real_loss = self.adv_loss_fn(self.discriminator(gt_pts), real)
        fake_loss = self.adv_loss_fn(self.discriminator(pred_pts), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        return {'d_loss': d_loss}, out_dict

    def loss_function(self, data_dict, optimizer_idx):
        """Wrapper for computing MoN loss.

        We sample predictions for multiple times and return the min one.

        Args:
            data_dict: from dataloader
            optimizer_idx: 0 --> Generator step, 1 --> Discriminator step;
                -1 --> in doing eval
        """
        if optimizer_idx == -1:  # in eval mode
            return super().loss_function(data_dict, optimizer_idx)

        loss_dict = None
        pre_pose_feats = None
        for _ in range(self.sample_iter):
            sample_loss, out_dict = self._loss_function(
                data_dict, pre_pose_feats, optimizer_idx=optimizer_idx)
            pre_pose_feats = out_dict['pre_pose_feats']

            if loss_dict is None:
                loss_dict = {k: [] for k in sample_loss.keys()}
            for k, v in sample_loss.items():
                loss_dict[k].append(v)
        loss_dict = {k: torch.stack(v, dim=0) for k, v in loss_dict.items()}

        if optimizer_idx == 1:  # d step, only `d_loss`
            d_loss = loss_dict['d_loss'].mean()
            return {
                'd_loss': d_loss,
                'loss': d_loss * self.cfg.loss.d_loss_w,
            }

        assert optimizer_idx == 0

        # `g_loss` doesn't involve in MoN loss computation
        g_loss = loss_dict.pop('g_loss').mean()

        # take the min for each data in the batch
        total_loss = 0.
        for k, v in loss_dict.items():
            if 'loss' in k:  # we may log some other metrics in eval, e.g. acc
                total_loss += v * eval(f'self.cfg.loss.{k}_w')  # weighting
        loss_dict['loss'] = total_loss

        # `total_loss` is of shape [sample_iter, B]
        min_idx = total_loss.argmin(0)  # [B]
        B = min_idx.shape[0]
        batch_idx = torch.arange(B).type_as(min_idx)
        loss_dict = {
            k: v[min_idx, batch_idx].mean()
            for k, v in loss_dict.items()
        }

        # add `g_loss`
        loss_dict['g_loss'] = g_loss
        loss_dict['loss'] = loss_dict['loss'] + g_loss * self.cfg.loss.g_loss_w

        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        g_lr = self.cfg.optimizer.g_lr
        d_lr = self.cfg.optimizer.d_lr
        g_opt = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.corr_module.parameters()) +
            list(self.pose_predictor.parameters()),
            lr=g_lr)
        d_opt = optim.Adam(self.discriminator.parameters(), lr=d_lr)

        clip_lr = min(g_lr, d_lr) / 10.  # this is a bit hack
        total_epochs = self.cfg.exp.num_epochs
        warmup_epochs = int(total_epochs * self.cfg.optimizer.warmup_ratio)
        g_scheduler = CosineAnnealingWarmupRestarts(
            g_opt,
            total_epochs,
            max_lr=g_lr,
            min_lr=clip_lr,
            warmup_steps=warmup_epochs)
        d_scheduler = CosineAnnealingWarmupRestarts(
            d_opt,
            total_epochs,
            max_lr=d_lr,
            min_lr=clip_lr,
            warmup_steps=warmup_epochs)

        return (
            [g_opt, d_opt],
            [{
                'scheduler': g_scheduler,
                'interval': 'epoch',
            }, {
                'scheduler': d_scheduler,
                'interval': 'epoch',
            }],
        )
