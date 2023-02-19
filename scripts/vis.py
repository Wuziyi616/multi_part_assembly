import os
import sys
import copy
import argparse
import importlib

import trimesh
import numpy as np
from tqdm import tqdm

import torch

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import trans_rmat_to_pmat, trans_quat_to_pmat, \
    quaternion_to_rmat, save_pc, Rotation3D


@torch.no_grad()
def visualize(cfg):
    # Initialize model
    model = build_model(cfg)
    ckp = torch.load(cfg.exp.weight_file, map_location='cpu')
    model.load_state_dict(ckp['state_dict'])
    model = model.cuda().eval()

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)
    val_dst = val_loader.dataset

    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in tqdm(val_loader):
        batch = {k: v.float().cuda() for k, v in batch.items()}
        out_dict = model(batch)  # trans/rot: [B, P, 3/4/(3, 3)]
        # compute loss to measure the quality of the predictions
        batch['part_rot'] = Rotation3D(
            batch['part_quat'], rot_type='quat').convert(model.rot_type)
        loss_dict, _ = model._calc_loss(out_dict, batch)  # loss is [B]
        # the criterion to cherry-pick examples
        loss = loss_dict['rot_pt_l2_loss'] + loss_dict['trans_mae']
        # convert all the rotations to quaternion for simplicity
        out_dict = {
            'data_id': batch['data_id'].long(),
            'pred_trans': out_dict['trans'],
            'pred_quat': out_dict['rot'].to_quat(),
            'gt_trans': batch['part_trans'],
            'gt_quat': batch['part_rot'].to_quat(),
            'part_valids': batch['part_valids'].long(),
        }
        out_dict = {k: v.cpu().numpy() for k, v in out_dict.items()}
        out_dict_lst = [{k: v[i]
                         for k, v in out_dict.items()}
                        for i in range(loss.shape[0])]
        vis_lst += out_dict_lst
        loss_lst.append(loss.cpu().numpy())
    loss_lst = np.concatenate(loss_lst, axis=0)
    top_idx = np.argsort(loss_lst)[:args.vis]

    # apply the predicted transforms to the original meshes and save them
    save_dir = os.path.join(
        os.path.dirname(cfg.exp.weight_file), 'vis', args.category)
    for rank, idx in enumerate(top_idx):
        out_dict = vis_lst[idx]
        data_id = out_dict['data_id']
        mesh_dir = os.path.join(val_dst.data_dir, val_dst.data_list[data_id])
        mesh_files = os.listdir(mesh_dir)
        mesh_files.sort()
        assert len(mesh_files) == out_dict['part_valids'].sum()
        subfolder_name = f"rank{rank}-{len(mesh_files)}pcs-"\
                         f"{mesh_dir.split('/')[-1]}"
        cur_save_dir = os.path.join(save_dir,
                                    mesh_dir.split('/')[-2], subfolder_name)
        os.makedirs(cur_save_dir, exist_ok=True)
        for i, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            mesh.export(os.path.join(cur_save_dir, mesh_file))
            # R^T (mesh - T) --> init_mesh
            gt_trans, gt_quat = \
                out_dict['gt_trans'][i], out_dict['gt_quat'][i]
            gt_rmat = quaternion_to_rmat(gt_quat)
            init_trans = -(gt_rmat.T @ gt_trans)
            init_rmat = gt_rmat.T
            init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
            init_mesh = mesh.apply_transform(init_pmat)
            init_mesh.export(os.path.join(cur_save_dir, f'input_{mesh_file}'))
            init_pc = trimesh.sample.sample_surface(init_mesh,
                                                    val_dst.num_points)[0]
            save_pc(init_pc,
                    os.path.join(cur_save_dir, f'input_{mesh_file[:-4]}.ply'))
            # predicted pose
            pred_trans, pred_quat = \
                out_dict['pred_trans'][i], out_dict['pred_quat'][i]
            pred_pmat = trans_quat_to_pmat(pred_trans, pred_quat)
            pred_mesh = init_mesh.apply_transform(pred_pmat)
            pred_mesh.export(os.path.join(cur_save_dir, f'pred_{mesh_file}'))
            pred_pc = trimesh.sample.sample_surface(pred_mesh,
                                                    val_dst.num_points)[0]
            save_pc(pred_pc,
                    os.path.join(cur_save_dir, f'pred_{mesh_file[:-4]}.ply'))

    print(f'Saving {len(top_idx)} predictions for visualization...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=-1, help='visualization')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()

    if args.category:
        cfg.data.category = args.category
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part
    if args.weight:
        cfg.exp.weight_file = args.weight
    else:
        assert cfg.exp.weight_file, 'Please provide weight to test'

    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)

    if not args.category:
        args.category = 'all'
    visualize(cfg)
