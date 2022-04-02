import os
import sys
import argparse
import importlib

import pytorch_lightning as pl

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import pickle_dump


def test(cfg):
    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)

    trainer = pl.Trainer(gpus=[0])

    trainer.test(model, val_loader, ckpt_path=cfg.exp.weight_file)

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
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=-1, help='visualization')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    cfg.merge_from_file(args.yml_file)

    if args.weight:
        cfg.exp.weight_file = args.weight
    else:
        assert cfg.exp.weight_file, 'Please provide weight to test'

    cfg.freeze()
    print(cfg)

    if args.vis > 0:
        visualize(cfg)
    else:
        test(cfg)
