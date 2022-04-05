import os
import sys
import copy
import argparse
import importlib

import trimesh
import numpy as np
import open3d as o3d

import torch
import pytorch_lightning as pl

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import trans_rmat_to_pmat, trans_quat_to_pmat, \
    quaternion_to_rmat


def test(cfg):
    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)

    all_gpus = list(cfg.exp.gpus)
    trainer = pl.Trainer(
        gpus=all_gpus,
        strategy='dp' if len(all_gpus) > 1 else None,
    )

    trainer.test(model, val_loader, ckpt_path=cfg.exp.weight_file)

    if args.category != 'all':
        return
    # if `args.category` is 'all', we also compute per-category results
    # TODO: currently we hard-code to support Breaking Bad dataset
    all_category = [
        'BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate',
        'Spoon', 'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie',
        'DrinkBottle', 'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot',
        'Vase', 'WineGlass'
    ]
    all_metrics = {
        'rot_rmse': 1.,
        'rot_mae': 1.,
        'trans_rmse': 100.,
        'trans_mae': 100.,
        'transform_pt_cd_loss': 1000.,
        'part_acc': 100.,
    }
    all_results = {metric: [] for metric in all_metrics.keys()}
    for cat in all_category:
        cfg = copy.deepcopy(cfg_backup)
        cfg.data.category = cat
        _, val_loader = build_dataloader(cfg)
        trainer.test(model, val_loader, ckpt_path=cfg.exp.weight_file)
        results = model.test_results
        results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
        for metric in all_metrics.keys():
            all_results[metric].append(results[metric] * all_metrics[metric])
    all_results = {k: np.array(v).round(1) for k, v in all_results.items()}
    # format for latex table
    for metric, result in all_results.items():
        print(f'{metric}:')
        result = result.tolist()
        result.append(np.mean(result).round(1))  # per-category mean
        result = [str(res) for res in result]
        print(' & '.join(result))

    print('Done testing...')


@torch.no_grad()
def visualize(cfg):
    # Initialize model
    model = build_model(cfg).cuda()

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)
    val_dst = val_loader.dataset

    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in val_loader:
        batch = {k: v.float().to(model.device) for k, v in batch.items()}
        out_dict = model(batch)  # trans/quat: [B, P, 3/4]
        loss_dict, _ = model._calc_loss(out_dict, batch)  # each loss is [B]
        # TODO: the criterion to select examples
        loss = loss_dict['trans_mae'] + loss_dict['rot_mae'] / 1000.
        out_dict = {
            'data_id': batch['data_id'].long(),
            'pred_trans': out_dict['trans'],
            'pred_quat': out_dict['quat'],
            'gt_trans': batch['part_trans'],
            'gt_quat': batch['part_quat'],
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
    for idx in top_idx:
        out_dict = vis_lst[idx]
        data_id = out_dict['data_id']
        mesh_dir = os.path.join(val_dst.data_dir, val_dst.data_list[data_id])
        mesh_files = os.listdir(mesh_dir)
        mesh_files.sort()
        assert len(mesh_files) == out_dict['part_valids'].sum()
        cur_save_dir = os.path.join(save_dir, os.path.basename(mesh_dir))
        os.makedirs(cur_save_dir, exist_ok=True)
        for i, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            gt_trans, gt_quat = \
                out_dict['gt_trans'][i], out_dict['gt_trans'][i]
            # R^T (mesh - T) --> init_mesh
            gt_rmat = quaternion_to_rmat(gt_quat)
            init_trans = -(gt_rmat.T @ gt_trans)
            init_rmat = gt_rmat.T
            init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
            init_mesh = mesh.apply_transform(init_pmat)
            # predicted pose
            pred_trans, pred_quat = \
                out_dict['pred_trans'][i], out_dict['pred_trans'][i]
            pred_pmat = trans_quat_to_pmat(pred_trans, pred_quat)
            pred_mesh = init_mesh.apply_transform(pred_pmat)
            # sample point clouds
            init_pc = trimesh.sample.sample_surface(init_mesh,
                                                    val_dst.num_points)[0]
            pred_pc = trimesh.sample.sample_surface(pred_mesh,
                                                    val_dst.num_points)[0]
            # save
            mesh.export(os.path.join(cur_save_dir, mesh_file))
            init_mesh.export(os.path.join(cur_save_dir, f'input_{mesh_file}'))
            pred_mesh.export(os.path.join(cur_save_dir, f'pred_{mesh_file}'))
            o3d.io.write_point_cloud(
                os.path.join(cur_save_dir, f'input_{mesh_file[:-4]}.ply'),
                init_pc)
            o3d.io.write_point_cloud(
                os.path.join(cur_save_dir, f'pred_{mesh_file[:-4]}.ply'),
                pred_pc)

    print(f'Saving {args.vis} predictions for visualization...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--yml_file', required=True, type=str, help='.yml')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=-1, help='visualization')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    cfg.merge_from_file(args.yml_file)

    cfg.exp.gpus = args.gpus
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

    if args.vis > 0:
        if not args.category:
            args.category = 'all'
        visualize(cfg)
    else:
        test(cfg)
