import os
import sys
import copy
import argparse
import importlib
import numpy as np

import pytorch_lightning as pl

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import pickle_dump


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


def visualize(cfg):
    # Initialize model
    model = build_model(cfg).cuda()

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)

    # save some predictions for visualization
    vis_num = args.vis
    vis_lst = []
    for batch in val_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out_dict = model.sample_assembly(batch, ret_pcs=False)
        """list of dicts, each dict contains:
            - data_id: int, index input of dataset.__getitem__
            - gt_trans/quat: [P, 3/4]
            - pred_trans/quat: [P, 3/4]
        """
        vis_lst += out_dict
        if len(vis_lst) >= vis_num:
            break

    # save results
    save_dir = os.path.dirname(cfg.exp.weight_file)
    save_name = os.path.join(save_dir, 'vis.pkl')
    pickle_dump(vis_lst, save_name)

    print(f'Saving {vis_num} predictions for visualization...')


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
        visualize(cfg)
    else:
        test(cfg)
