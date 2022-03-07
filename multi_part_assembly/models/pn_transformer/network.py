import torch
import torch.optim as optim
import pytorch_lightning as pl

import numpy as np
from scipy.optimize import linear_sum_assignment

from multi_part_assembly.models.encoder import build_encoder
from multi_part_assembly.utils.quaternion import qtransform
from multi_part_assembly.utils.chamfer import chamfer_distance
from multi_part_assembly.utils.loss import trans_l2_loss, rot_l2_loss, \
    rot_cosine_loss, rot_points_l2_loss, rot_points_cd_loss, shape_cd_loss

from .transformer import TransformerEncoder
from .regressor import StocasticPoseRegressor


class PNTransformer(pl.LightningModule):
    """Baseline multi-part assembly model.

    Encoder: PointNet extracting per-part global point cloud features
    Correlator: TransformerEncoder perform part interactions
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.max_num_part = self.cfg.data.max_num_part
        self.pc_feat_dim = self.cfg.model.pc_feat_dim

        # loss configs
        self.sample_iter = self.cfg.loss.sample_iter
        self.rot_loss = self.cfg.loss.rot_loss
        self.use_rot_pt_l2_loss = self.cfg.loss.use_rot_pt_l2_loss
        self.use_rot_pt_cd_loss = self.cfg.loss.use_rot_pt_cd_loss
        self.use_transform_pt_cd_loss = self.cfg.loss.use_transform_pt_cd_loss

        self.encoder = self._init_encoder()
        self.corr_module = self._init_corr_module()
        self.pose_predictor = self._init_pose_predictor()

    def _init_encoder(self):
        encoder = build_encoder(
            self.cfg.encoder, feat_dim=self.pc_feat_dim, global_feat=True)
        return encoder

    def _init_corr_module(self):
        corr_module = TransformerEncoder(
            d_model=self.pc_feat_dim,
            num_heads=self.cfg.model.transformer_heads,
            ffn_dim=self.cfg.model.transformer_feat_dim,
            num_layers=self.cfg.model.transformer_layers,
            norm_first=self.cfg.model.transformer_pre_ln,
        )
        return corr_module

    def _init_pose_predictor(self):
        # concat feature, instance_label and noise as input
        pose_predictor = StocasticPoseRegressor(
            feat_dim=self.pc_feat_dim + self.max_num_part,
            noise_dim=self.cfg.model.noise_dim)
        return pose_predictor

    def forward(self, part_pcs, part_valids, instance_label):
        """Forward pass to predict poses for each part.

        Args:
            part_pcs: [B, P, N, 3]
            part_valids: [B, P]
            instance_label: [B, P, P]
        """
        B, P, N, _ = part_pcs.shape
        # shared-weight encoder
        pcs = part_pcs.flatten(0, 1)  # [B*P, N, 3]
        pc_feats = self.encoder(pcs).unflatten(0, (B, P))  # [B, P, C]
        # transformer feature fusion
        pc_feats = self.corr_module(pc_feats, part_valids)  # [B, P, C]
        # MLP predict poses
        instance_label = instance_label.type_as(pc_feats)
        feats = torch.cat([pc_feats, instance_label], dim=-1)  # [B, P, C+P]
        quat, trans = self.pose_predictor(feats)

        pred_dict = {
            'quat': quat,  # [B, P, 4]
            'trans': trans,  # [B, P, 3]
        }
        return pred_dict

    def training_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='train')
        return loss_dict['loss']

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='val')
        return loss_dict

    def validation_epoch_end(self, outputs):
        # avg_loss among all data
        loss_dict = {
            f'val/{k}':
            torch.stack([output[k] for output in outputs]).mean().item()
            for k in outputs[0].keys()
        }
        self.log_dict(loss_dict, sync_dist=True)

    def forward_pass(self, data_dict, mode):
        """Forward pass: loss computation and logging.

        data_dict = {
            'part_pcs': [B, P, N, 3],
            'part_trans': [B, P, 3],
            'part_quat': [B, P, 4],
            'part_valids': [B, P], 1 for valid, 0 for padded,
            'shape_id': int,
            'part_ids': [B, P],
            'instance_label': [B, P, P],
            'match_ids': [B, P],
            'contact_points': [B, P, P, 4],
            'sym': [B, P, 3],
        }
        """
        loss_dict = self.loss_function(data_dict)

        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss_dict.items()}
            self.log_dict(log_dict, logger=True, sync_dist=False)

        return loss_dict

    @torch.no_grad()
    @staticmethod
    def _linear_sum_assignment(pts, trans1, quat1, trans2, quat2):
        """Find the min-cose match between two groups of poses.

        Args:
            pts: [p, N, 3]
            trans1/2: [p, 3]
            quat1/2: [p, 4]

        Returns:
            torch.Tensor x2: [p], [p], matching index
        """
        p, N, _ = pts.shape
        # subsample points for speed-up
        n = 100
        sample_idx = torch.randperm(N)[:n].to(pts.device).long()
        pts = pts[:, sample_idx]  # [p, n, 3]

        pts1 = qtransform(trans1, quat1, pts)
        pts2 = qtransform(trans2, quat2, pts)

        pts1 = pts1.unsqueeze(1).repeat(1, p, 1, 1).view(-1, n, 3)
        pts2 = pts2.unsqueeze(0).repeat(p, 1, 1, 1).view(-1, n, 3)
        dist1, dist2 = chamfer_distance(pts1, pts2)
        dist_mat = (dist1.mean(1) + dist2.mean(1)).view(p, p)
        rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

        rind = torch.from_numpy(rind).type_as(sample_idx)
        cind = torch.from_numpy(cind).type_as(sample_idx)
        return rind, cind

    @torch.no_grad()
    def _match_parts(self, part_pcs, pred_trans, pred_quat, gt_trans, gt_quat,
                     match_ids):
        """Match GT to predctions.

        Args:
            part_pcs: [B, P, N, 3]
            pred/gt_trans: [B, P, 3]
            pred/gt_quat: [B, P, 4]
            match_ids: [B, P], indicator of equivalent parts in the shape, e.g.
                [0, 1, 1, 0, 2, 2, 2, 0, 0], where 0 are padded or unique part,
                two `1` are one group, three `2` are another group of parts.

        Returns:
            GT poses after rearrangement
        """
        max_num_part = self.max_num_part
        match_ids = match_ids.long()

        # iterate over batch
        for ind in range(part_pcs.shape[0]):
            cur_match_ids = match_ids[ind]

            # for each group of parts
            for i in range(1, cur_match_ids.max().item() + 1):
                # find equivalent parts to perform matching
                # if i == 2, then need_to_match_part == [4, 5, 6]
                need_to_match_part = []
                for j in range(max_num_part):
                    if cur_match_ids[j] == i:
                        need_to_match_part.append(j)

                # extract group data and perform matching
                cur_pts = part_pcs[ind, need_to_match_part]
                cur_pred_trans = pred_trans[ind, need_to_match_part]
                cur_pred_quat = pred_quat[ind, need_to_match_part]
                cur_gt_trans = gt_trans[ind, need_to_match_part]
                cur_gt_quat = gt_quat[ind, need_to_match_part]

                _, matched_gt_ids = self._linear_sum_assignment(
                    cur_pts, cur_pred_trans, cur_pred_quat, cur_gt_trans,
                    cur_gt_quat)

                # since row_idx is sorted, we can directly rearrange GT
                gt_trans[ind, need_to_match_part] = \
                    gt_trans[ind, need_to_match_part][matched_gt_ids]
                gt_quat[ind, need_to_match_part] = \
                    gt_quat[ind, need_to_match_part][matched_gt_ids]

        return gt_trans, gt_quat

    def _loss_function(self, data_dict):
        """Predict poses and calculate loss.

        Since there could be several parts that are the same in one shape, we
            need to do Hungarian matching to find the min loss values.

        Returns a dict of loss, each is a [B] shape tensor for later selection.
        See GNN Assembly paper Sec 3.4, the MoN loss is sampling prediction
            several times and select the min one as final loss.
        """
        part_pcs, part_valids = data_dict['part_pcs'], data_dict['part_valids']
        instance_label = data_dict['instance_label']

        # prediction
        out_dict = self.forward(part_pcs, part_valids, instance_label)
        pred_trans, pred_quat = out_dict['quat'], out_dict['trans']

        # matching
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']
        match_ids = data_dict['match_ids']
        gt_trans, gt_quat = self._match_parts(part_pcs, pred_trans, pred_quat,
                                              gt_trans, gt_quat, match_ids)

        # computing loss
        trans_loss = trans_l2_loss(pred_trans, gt_trans, part_valids)
        if self.rot_loss == 'l2':
            rot_loss = rot_l2_loss(pred_quat, gt_quat, part_valids)
        elif self.rot_loss == 'cosine':
            rot_loss = rot_cosine_loss(pred_quat, gt_quat, part_valids)
        else:
            raise NotImplementedError
        loss_dict = {
            'trans_loss': trans_loss,  # [B]
            'rot_loss': rot_loss,  # [B]
        }
        if self.use_rot_pt_l2_loss:
            loss_dict['rot_pt_l2_loss'] = rot_points_l2_loss(
                part_pcs, pred_quat, gt_quat, part_valids)  # [B]
        if self.use_rot_pt_cd_loss:
            loss_dict['rot_pt_cd_loss'] = rot_points_cd_loss(
                part_pcs, pred_quat, gt_quat, part_valids)  # [B]
        if self.use_transform_pt_cd_loss:
            loss_dict['transform_pt_cd_loss'] = shape_cd_loss(
                part_pcs, pred_trans, gt_trans, pred_quat, gt_quat,
                part_valids)  # [B]

        return loss_dict

    def loss_function(self, data_dict):
        """Wrapper for computing MoN loss.

        We sample predictions for multiple times and return the min one.
        """
        loss_dict = None
        for _ in range(self.sample_iter):
            # TODO: can optimize speed by reusing computed features
            # TODO: only StochasticPoseRegressor needs to be sampled
            sample_loss = self._loss_function(data_dict)

            if loss_dict is None:
                loss_dict = {k: [] for k in sample_loss.keys()}
            for k, v in sample_loss.items():
                loss_dict[k].append(v)

        # take the min for each data in the batch
        total_loss = 0.
        loss_dict = {k: torch.stack(v, dim=0) for k, v in loss_dict.items()}
        for k, v in loss_dict.items():
            total_loss += v * eval(f'self.cfg.loss.{k}_w')
        loss_dict['loss'] = total_loss

        # `total_loss` is of shape [sample_iter, B]
        min_idx = total_loss.argmin(0)  # [B]
        B = min_idx.shape[0]
        batch_idx = torch.arange(B).type_as(min_idx)
        loss_dict = {
            k: v[min_idx, batch_idx].mean()
            for k, v in loss_dict.items()
        }

        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay)

        total_epochs = self.cfg.exp.num_epochs
        warmup_epochs = int(total_epochs * self.cfg.optimizer.warmup_ratio)

        def warmup_cosine_decay(epoch):
            """First linear increase then cosine decay to 0."""
            assert epoch <= total_epochs
            if epoch < warmup_epochs:
                factor = epoch / warmup_epochs
            else:
                factor = 1
            factor *= ((np.cos(epoch / total_epochs * np.pi) + 1.) / 2.)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=warmup_cosine_decay)

        return (
            [optimizer],
            [{
                'scheduler': scheduler,
                'interval': 'epoch',
            }],
        )

    def sample_assembly(self, data_dict):
        """Sample assembly for visualization."""
        part_pcs, part_valids = data_dict['part_pcs'], data_dict['part_valids']
        instance_label = data_dict['instance_label']
        out_dict = self.forward(part_pcs, part_valids, instance_label)
        pred_trans, pred_quat = out_dict['quat'], out_dict['trans']
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']

        pred_pcs = qtransform(pred_trans, pred_quat, part_pcs)
        gt_pcs = qtransform(gt_trans, gt_quat, part_pcs)
        B, P, N, _ = part_pcs.shape
        pred_pcs_lst, gt_pcs_lst = [], []
        for i in range(B):
            valid = part_valids[i].bool()  # [P]
            pred, gt = pred_pcs[i], gt_pcs[i]  # [P, N, 3]
            pred = pred[valid].flatten(0, 1).cpu().numpy()
            gt = gt[valid].flatten(0, 1).cpu().numpy()  # [n*N, 3]
            pred_pcs_lst.append(pred)
            gt_pcs_lst.append(gt)
        pred_pcs = np.stack(pred_pcs_lst, axis=0)
        gt_pcs = np.stack(gt_pcs_lst, axis=0)
        return gt_pcs, pred_pcs
