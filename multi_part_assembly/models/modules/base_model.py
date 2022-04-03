import torch
import torch.optim as optim
import pytorch_lightning as pl

import numpy as np
from scipy.optimize import linear_sum_assignment

from multi_part_assembly.utils import qtransform, chamfer_distance
from multi_part_assembly.utils import colorize_part_pc, filter_wd_parameters
from multi_part_assembly.utils import trans_l2_loss, rot_points_cd_loss, \
    shape_cd_loss, calc_part_acc, calc_connectivity_acc, \
    trans_metrics, rot_metrics, rot_l2_loss, rot_cosine_loss
from multi_part_assembly.utils import CosineAnnealingWarmupRestarts


class BaseModel(pl.LightningModule):
    """Base class for multi-part assembly model."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.semantic = (cfg.data.dataset != 'geometry')
        self.max_num_part = self.cfg.data.max_num_part
        self.pc_feat_dim = self.cfg.model.pc_feat_dim

        # loss configs
        self.sample_iter = self.cfg.loss.sample_iter

    def forward(self, data_dict):
        """Forward pass to predict poses for each part."""
        pass

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1):
        loss_dict = self.forward_pass(
            data_dict, mode='train', optimizer_idx=optimizer_idx)
        return loss_dict['loss']

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='val', optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size
        batch_sizes = torch.tensor([
            output.pop('batch_size') for output in outputs
        ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs])
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='test', optimizer_idx=-1)
        return loss_dict

    def test_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size
        batch_sizes = torch.tensor([
            output.pop('batch_size') for output in outputs
        ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'test/{k}': torch.stack([output[k] for output in outputs])
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))
        # TODO: this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss

    def forward_pass(self, data_dict, mode, optimizer_idx):
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
        loss_dict = self.loss_function(data_dict, optimizer_idx=optimizer_idx)

        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss_dict.items()}
            log_dict[f'{mode}/data_time'] = \
                self.trainer.profiler.recorded_durations['get_train_batch'][-1]
            self.log_dict(log_dict, logger=True, sync_dist=False)

        return loss_dict

    @torch.no_grad()
    def _linear_sum_assignment(self, pts, trans1, quat1, trans2, quat2):
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
        # gt to be modified
        new_gt_trans = gt_trans.detach().clone()
        new_gt_quat = gt_quat.detach().clone()

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
                new_gt_trans[ind, need_to_match_part] = \
                    gt_trans[ind, need_to_match_part][matched_gt_ids]
                new_gt_quat[ind, need_to_match_part] = \
                    gt_quat[ind, need_to_match_part][matched_gt_ids]

        return new_gt_trans, new_gt_quat

    def _calc_loss(self, out_dict, data_dict):
        """Calculate loss by matching GT to prediction.

        Also compute evaluation metrics during testing.
        """
        pred_trans, pred_quat = out_dict['trans'], out_dict['quat']

        # matching GT with predictions for lowest loss in semantic assembly
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']
        if self.semantic:
            match_ids = data_dict['match_ids']
            new_trans, new_quat = self._match_parts(part_pcs, pred_trans,
                                                    pred_quat, gt_trans,
                                                    gt_quat, match_ids)
        else:
            new_trans = gt_trans.detach().clone()
            new_quat = gt_quat.detach().clone()

        # computing loss
        trans_loss = trans_l2_loss(pred_trans, new_trans, valids)
        rot_pt_cd_loss = rot_points_cd_loss(part_pcs, pred_quat, new_quat,
                                            valids)
        transform_pt_cd_loss, gt_trans_pts, pred_trans_pts = \
            shape_cd_loss(
                part_pcs,
                pred_trans,
                new_trans,
                pred_quat,
                new_quat,
                valids,
                ret_pts=True)
        loss_dict = {
            'trans_loss': trans_loss,
            'rot_pt_cd_loss': rot_pt_cd_loss,
            'transform_pt_cd_loss': transform_pt_cd_loss,
        }  # all loss are of shape [B]

        # TODO: direct regression loss on quat?
        if self.cfg.loss.rot_loss:
            if self.cfg.loss.rot_loss == 'l2':
                rot_loss = rot_l2_loss(pred_quat, new_quat, valids)
            elif self.cfg.loss.rot_loss == 'cosine':
                rot_loss = rot_cosine_loss(pred_quat, new_quat, valids)
            loss_dict['rot_loss'] = rot_loss

        # some specific evaluation metrics calculated in eval
        if not self.training:
            # part_acc and connectivity_acc in DGL paper
            loss_dict['part_acc'] = calc_part_acc(part_pcs, pred_trans,
                                                  new_trans, pred_quat,
                                                  new_quat, valids)
            if 'contact_points' in data_dict.keys():
                loss_dict['connectivity_acc'] = calc_connectivity_acc(
                    pred_trans, pred_quat, data_dict['contact_points'])
            # mse/rmse/mae of translation and rotation in NSM
            for metric in ['mse', 'rmse', 'mae']:
                loss_dict[f'trans_{metric}'] = trans_metrics(
                    pred_trans, new_trans, valids, metric)
                loss_dict[f'rot_{metric}'] = rot_metrics(
                    pred_quat, new_quat, valids, metric)

        # return some intermediate variables for reusing
        out_dict = {
            'pred_trans': pred_trans,  # [B, P, 3]
            'pred_quat': pred_quat,  # [B, P, 4]
            'gt_trans_pts': gt_trans_pts,  # [B, P, N, 3]
            'pred_trans_pts': pred_trans_pts,  # [B, P, N, 3]
        }

        return loss_dict, out_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss.

        A wrapper for `_calc_loss`, we can do some pre/post-processing here.
        """
        pass

    def loss_function(self, data_dict, optimizer_idx):
        """Wrapper for computing MoN loss.

        We sample predictions for multiple times and return the min one.
        """
        loss_dict = None
        out_dict = {}
        for _ in range(self.sample_iter):
            sample_loss, out_dict = self._loss_function(
                data_dict, out_dict, optimizer_idx=optimizer_idx)

            if loss_dict is None:
                loss_dict = {k: [] for k in sample_loss.keys()}
            for k, v in sample_loss.items():
                loss_dict[k].append(v)

        # take the min for each data in the batch
        total_loss = 0.
        loss_dict = {k: torch.stack(v, dim=0) for k, v in loss_dict.items()}
        for k, v in loss_dict.items():
            # we may log some other metrics in eval, e.g. acc
            # exclude them from loss computation
            if k.endswith('_loss'):
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

        # log the batch_size for avg_loss computation
        if not self.training:
            loss_dict['batch_size'] = B

        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.cfg.optimizer.lr
        wd = self.cfg.optimizer.weight_decay

        if wd > 0.:
            params_dict = filter_wd_parameters(self)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)

        total_epochs = self.cfg.exp.num_epochs
        warmup_epochs = int(total_epochs * self.cfg.optimizer.warmup_ratio)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_epochs,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_epochs)

        return (
            [optimizer],
            [{
                'scheduler': scheduler,
                'interval': 'epoch',
            }],
        )

    @torch.no_grad()
    def sample_assembly(self, data_dict, ret_pcs=True):
        """Sample assembly for visualization."""
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']
        sample_pred_pcs, sample_pred_trans, sample_pred_quat = [], [], []
        for _ in range(self.sample_iter):
            out_dict = self.forward(data_dict)
            pred_trans, pred_quat = out_dict['trans'], out_dict['quat']
            sample_pred_trans.append(pred_trans)
            sample_pred_quat.append(pred_quat)
            if ret_pcs:
                pred_pcs = qtransform(pred_trans, pred_quat, part_pcs)
                sample_pred_pcs.append(pred_pcs)

        # only return transformation for further visualization
        if not ret_pcs:
            data_id = data_dict['data_id']
            out_dict = {
                'data_id': data_id,
                'gt_trans': gt_trans,
                'gt_quat': gt_quat,
                'pred_trans': torch.stack(sample_pred_trans, dim=1),
                'pred_quat': torch.stack(sample_pred_quat, dim=1),
            }
            out_dict = {k: v.cpu().numpy() for k, v in out_dict.items()}
            # to list of dict
            out_dict = [{k: v[i]
                         for k, v in out_dict.items()}
                        for i in range(len(data_id))]
            return out_dict

        # return GT assembly and predicted assembly
        gt_pcs = qtransform(gt_trans, gt_quat, part_pcs)  # [B, P, N, 3]

        colors = np.array(self.cfg.data.colors)
        B = part_pcs.shape[0]
        pred_pcs_lst, gt_pcs_lst = [[] for _ in range(B)], []
        for i in range(self.sample_iter):
            pred_pcs = sample_pred_pcs[i]  # [B, P, N, 3]
            for j in range(B):
                valid = valids[j].bool()  # [P]
                pred = pred_pcs[j][valid].cpu().numpy()  # [p, N, 3]
                pred = colorize_part_pc(pred, colors).reshape(-1, 6)
                pred_pcs_lst[j].append(pred)  # [p*N, 6]
                # only append GT once
                if i == 0:
                    gt = gt_pcs[j][valid].cpu().numpy()
                    gt = colorize_part_pc(gt, colors).reshape(-1, 6)
                    gt_pcs_lst.append(gt)

        return gt_pcs_lst, pred_pcs_lst
