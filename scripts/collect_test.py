"""Test per-category trained models and aggregate test results."""

import os
import sys
import copy
import argparse
import importlib

import numpy as np

import torch
import pytorch_lightning as pl

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model


def find_last_ckp(ckp_dir):
    """Find the last ckp in a directory."""
    ckp_files = os.listdir(ckp_dir)
    ckp_files = [ckp for ckp in ckp_files if 'model-' in ckp]
    assert len(ckp_files)
    ckp_files = sorted(
        ckp_files, key=lambda x: os.path.getmtime(os.path.join(ckp_dir, x)))
    last_ckp = ckp_files[-1]
    ckp_path = os.path.join(ckp_dir, last_ckp)
    return ckp_path


@torch.no_grad()
def test(cfg):
    # Initialize model
    model = build_model(cfg)

    all_gpus = list(cfg.exp.gpus)
    trainer = pl.Trainer(
        gpus=all_gpus,
        # we should use DP because DDP will duplicate some data
        strategy='dp' if len(all_gpus) > 1 else None,
    )

    # TODO: modify this to fit in the metrics you want to report
    all_metrics = {
        'rot_rmse': 1.,
        'rot_mae': 1.,
        'trans_rmse': 100.,  # presented as \times 1e-2 in the table
        'trans_mae': 100.,  # presented as \times 1e-2 in the table
        'transform_pt_cd_loss': 1000.,  # presented as \times 1e-3 in the table
        'part_acc': 100.,  # presented in % in the table
    }

    # performance on all categories
    if args.train_all:
        all_results = {metric: [] for metric in all_metrics.keys()}
        ckp_suffix = f'{args.ckp_suffix}dup'
        _, val_loader = build_dataloader(cfg)
        for i in range(1, args.num_dup + 1, 1):
            ckp_folder = f'{ckp_suffix}{i}/models'
            try:
                ckp_path = find_last_ckp(ckp_folder)
            except AssertionError:
                continue
            trainer.test(model, val_loader, ckpt_path=ckp_path)
            results = model.test_results
            results = {k[5:]: v.cpu().numpy() for k, v in results.items()}
            for metric in all_metrics.keys():
                all_results[metric].append(results[metric] *
                                           all_metrics[metric])
        # average over `dup` runs
        for metric in all_metrics.keys():
            all_results[metric] = np.mean(all_results[metric]).round(1)
            print(f'{metric}: {all_results[metric]}')
        # format for latex table
        print('\n##############################################')
        print('Results averaged over all categories:')
        result = [str(all_results[metric]) for metric in all_metrics.keys()]
        print(' & '.join(result))

        if not hasattr(cfg.data, 'all_category'):
            return

    # iterate over all categories
    all_category = cfg.data.all_category
    all_results = {
        cat: {metric: []
              for metric in all_metrics.keys()}
        for cat in all_category
    }
    for cat in all_category:
        cfg = copy.deepcopy(cfg_backup)
        cfg.data.category = cat
        _, val_loader = build_dataloader(cfg)
        if not len(val_loader):
            for metric in all_metrics.keys():
                all_results[cat][metric].append(np.nan)
            continue
        # iterate over all dup-trained models
        # 'dup1', 'dup2', 'dup3', ...
        # if the model is trained on all categories together
        # then there is only one weight
        if args.train_all:
            ckp_suffix = f'{args.ckp_suffix}dup'
        # else there is one weight per category
        else:
            ckp_suffix = f'{args.ckp_suffix}{cat}-dup'
        for i in range(1, args.num_dup + 1, 1):
            ckp_folder = f'{ckp_suffix}{i}/models'
            try:
                ckp_path = find_last_ckp(ckp_folder)
            except AssertionError:
                continue
            trainer.test(model, val_loader, ckpt_path=ckp_path)
            results = model.test_results
            results = {k[5:]: v.cpu().numpy() for k, v in results.items()}
            for metric in all_metrics.keys():
                all_results[cat][metric].append(results[metric] *
                                                all_metrics[metric])
    # average over `dup` runs
    for cat in all_category:
        for metric in all_metrics.keys():
            all_results[cat][metric] = np.mean(all_results[cat][metric])
    # format for latex table
    all_results = {
        metric: [all_results[cat][metric] for cat in all_category]
        for metric in all_metrics.keys()
    }
    all_results = {k: np.array(v).round(1) for k, v in all_results.items()}
    # format for latex table
    # per-category results
    print('\n##############################################')
    print('Results per category:')
    for metric, result in all_results.items():
        print(f'{metric}:')
        result = result.tolist()
        result.append(np.nanmean(result).round(1))  # per-category mean
        result = [str(res) for res in result]
        print(' & '.join(result))
        all_results[metric] = result
    # averaged over all categories
    print('\n##############################################')
    print('Results averaged over all categories:')
    all_metric_names = list(all_metrics.keys())
    result = [str(all_results[metric][-1]) for metric in all_metric_names]
    print(' & '.join(all_metric_names))
    print(' & '.join(result))

    print('Done testing...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--num_dup', type=int, default=3)
    parser.add_argument('--ckp_suffix', type=str, required=True)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument(
        '--train_all', action='store_true', help='trained on all categories')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()

    cfg.exp.gpus = args.gpus
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part

    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)

    test(cfg)
