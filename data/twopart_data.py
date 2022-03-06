import os
import numpy as np
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset

from shape_assembly.utils import read_csv

class BaseDataLoader:

    def __init__(
        self, 
        data_root_dir,
        data_csv_file,
        num_points
    ):
        # Data related variables
        self.data_root_dir = data_root_dir
        self.data_list     = read_csv(data_csv_file)

        # Number of points in a point cloud
        self.num_points = num_points

        # Data sampling related variables
        self.sample_idx_list = self.generate_sample_idx()
        self.sample_idx = 0

    def generate_sample_idx(self):
        idx_list = np.arange(len(self.data_list))
        np.random.shuffle(idx_list)
        return idx_list

    def adjust_idx(self):
        self.sample_idx += 1
        if self.sample_idx == len(self.data_list):
            self.sample_idx = 0
            self.sample_idx_list = self.generate_sample_idx()
        return

    def recenter_pc(self, pc):
        centroid = np.mean(pc, axis=1, keepdims=True)
        pc = pc - centroid
        return pc, centroid

    def sample_rotation(self):
        rot_mat = R.random().as_matrix()
        return rot_mat

    def rotate_pc(self, pcs):
        rot_mat = self.sample_rotation()
        pcs = rot_mat @ pcs
        rot_mat_gt = self.inverse_transform(rot_mat)
        return pcs, rot_mat_gt

    def inverse_transform(self, transform):
        return np.linalg.inv(transform)

    def shuffle_pc(self, pc):
        order = np.arange(self.num_points)
        np.random.shuffle(order)
        pc = pc[:,order]
        return pc

class DataFromPointClouds(BaseDataLoader):

    def __init__(
        self,
        data_root_dir,
        data_csv_file,
        num_points
    ):
        super().__init__(
            data_root_dir=data_root_dir,
            data_csv_file=data_csv_file,
            num_points=num_points
        )

    def get_pcs(self):
        # Get sample index
        idx = self.sample_idx_list[self.sample_idx]

        # Get paths
        src_pc_path = os.path.join(self.data_root_dir, self.data_list[idx], 'partA-pc.csv')
        tgt_pc_path = os.path.join(self.data_root_dir, self.data_list[idx], 'partB-pc.csv')

        # Read point clouds
        src_pc = np.genfromtxt(src_pc_path, delimiter=',')[:self.num_points].T
        tgt_pc = np.genfromtxt(tgt_pc_path, delimiter=',')[:self.num_points].T

        # Adjust sample index
        self.adjust_idx()

        return src_pc, tgt_pc

    def get_data(self):
        # Get point clouds
        src_pc, tgt_pc = self.get_pcs() # 3 x 1024

        # Recenter point clouds and get translation ground truths
        src_pc, src_trans_gt = self.recenter_pc(src_pc) # 3 x 1024, 3 x 1
        tgt_pc, tgt_trans_gt = self.recenter_pc(tgt_pc) # 3 x 1024, 3 x 1

        # Get rotation ground truths
        src_pc, src_rot_gt = self.rotate_pc(src_pc) # 3 x 1024, 3 x 3
        tgt_pc, tgt_rot_gt = self.rotate_pc(tgt_pc) # 3 x 1024, 3 x 3

        # Shuffle point clouds
        src_pc = self.shuffle_pc(src_pc)
        tgt_pc = self.shuffle_pc(tgt_pc)

        data_dict = {
            'src_pc': src_pc,             # 3 x 1024
            'tgt_pc': tgt_pc,             # 3 x 1024
            'src_rot_gt': src_rot_gt,     # 3 x 3
            'tgt_rot_gt': tgt_rot_gt,     # 3 x 3
            'src_trans_gt': src_trans_gt, # 3 x 1
            'tgt_trans_gt': tgt_trans_gt, # 3 x 1
        }

        return data_dict

class PartAssemblyDataset(Dataset):

    def __init__(
        self,
        data_root_dir,
        data_csv_file,
        num_points
    ):
        self.dataset = DataFromPointClouds(
            data_root_dir=data_root_dir,
            data_csv_file=data_csv_file,
            num_points=num_points
        )

    def __len__(self):
        return 1000

    def __getitem__(self, _):
        return self.dataset.get_data()
