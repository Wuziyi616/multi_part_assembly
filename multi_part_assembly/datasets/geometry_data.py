import os
import random

import trimesh
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader


class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset."""

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        num_points=1000,
        max_num_part=20,
        overfit=-1,
    ):
        # store parameters
        self.data_dir = data_dir
        with open(os.path.join(data_dir, data_fn), 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.num_points = num_points

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    @staticmethod
    def _rotate_pc(pc):
        """pc: [N, 3]"""
        rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

    def _rand_another(self):
        """Randomly load another data when current shape has too much parts."""
        index = random.choice(range(len(self)))
        return self.__getitem__(index)

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()
        if len(mesh_files) > self.max_num_part or len(mesh_files) <= 1:
            raise ValueError

        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))
            for mesh_file in mesh_files
        ]
        pcs = [
            trimesh.sample.sample_surface(mesh, self.num_points)
            for mesh in meshes
        ]
        return np.array(pcs)

    def __getitem__(self, index):
        try:
            pcs = self._get_pcs(self.data_list[index])
        except ValueError:
            return self._rand_another()
        num_parts = pcs.shape[0]
        cur_pts, cur_quat, cur_trans = [], [], []
        for i in range(num_parts):
            pc = pcs[i]
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            cur_pts.append(self._shuffle_pc(pc))
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0))  # [P, N, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0))  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0))  # [P, 3]
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'part_ids': MAX_NUM, useless

            'instance_label': MAX_NUM x 0, useless

        }
        """

        data_dict = {
            'part_pcs': cur_pts,
            'part_quat': cur_quat,
            'part_trans': cur_trans,
        }
        # valid part masks
        valids = np.zeros((self.max_num_part), dtype=np.float32)
        valids[:num_parts] = 1.
        data_dict['part_valids'] = valids

        for key in self.data_keys:
            if key == 'part_ids':
                cur_part_ids = np.arange(num_parts)  # p
                data_dict['part_ids'] = self._pad_data(cur_part_ids)

            elif key == 'instance_label':
                # useless in non-semantic assembly
                # make its last dim 0 so that we concat nothing
                instance_label = np.zeros((self.max_num_part, 0),
                                          dtype=np.float32)
                data_dict['instance_label'] = instance_label

            elif key == 'valid_matrix':
                out = np.zeros((self.max_num_part, self.max_num_part),
                               dtype=np.float32)
                out[:num_parts, :num_parts] = 1.
                data_dict['valid_matrix'] = out

            else:
                raise ValueError(f'ERROR: unknown data {key}!')

        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):
    train_set = GeometryPartDataset(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        num_points=cfg.data.num_points,
        max_num_part=cfg.data.max_num_part,
        overfit=cfg.data.overfit,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.exp.num_workers > 0),
    )

    val_set = GeometryPartDataset(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('val'),
        data_keys=cfg.data.data_keys,
        num_points=cfg.data.num_points,
        max_num_part=cfg.data.max_num_part,
        overfit=cfg.data.overfit,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.exp.num_workers > 0),
    )
    return train_loader, val_loader
