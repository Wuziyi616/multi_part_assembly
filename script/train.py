import os
import pwd
import argparse
import importlib

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from multi_part_assembly.datasets import build_partnet_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import PCAssemblyLogCallback


def main(cfg):
    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    train_loader, val_loader = build_partnet_dataloader(cfg)

    # Create checkpoint directory
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    exp_name = cfg.exp.name
    cfg_name = os.path.basename(args.yml_file)[:-4]  # remove '.yml'
    ckp_dir = os.path.join(cfg.exp.ckp_dir, exp_name, cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)

    # on clusters, quota is limited
    # soft link temp space for checkpointing
    if not os.path.exists(ckp_dir):
        usr = pwd.getpwuid(os.getuid())[0]
        os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
            usr, SLURM_JOB_ID, ckp_dir))

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-{epoch:03d}',
        monitor='val/loss',
        save_top_k=5,
        mode='min',
    )

    # visualize assembly results
    assembly_callback = PCAssemblyLogCallback(cfg.exp.val_sample_vis,
                                              train_loader, val_loader)

    logger_name = f'{exp_name}-{cfg_name}-{SLURM_JOB_ID}'
    logger = WandbLogger(
        project='Multi-Part-Assembly', name=logger_name, id=logger_name)

    all_gpus = list(cfg.exp.gpus)
    trainer = pl.Trainer(
        logger=logger,
        gpus=all_gpus,
        # TODO: very strange, I still cannot train DDP on Vector...
        # strategy='ddp' if len(all_gpus) > 1 else None,
        strategy='dp' if len(all_gpus) > 1 else None,
        max_epochs=cfg.exp.num_epochs,
        callbacks=[
            LearningRateMonitor('epoch'),
            checkpoint_callback,
            assembly_callback,
        ],
        precision=16 if args.fp16 else 32,  # FP16 training
        benchmark=args.cudnn,  # cudnn benchmark
        gradient_clip_val=cfg.optimizer.clip_grad,  # clip grad norm
        check_val_every_n_epoch=cfg.exp.val_every,
        log_every_n_steps=50,
        profiler='simple',  # training time bottleneck analysis
        # detect_anomaly=True,
    )

    # automatically detect existing checkpoints in case of preemption
    ckp_files = os.listdir(ckp_dir)
    ckp_files = [ckp for ckp in ckp_files if 'model-' in ckp]
    if ckp_files:  # note that this will overwrite `args.weight`
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(ckp_dir, x)))
        last_ckp = ckp_files[-1]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        ckp_path = os.path.join(ckp_dir, last_ckp)
    elif cfg.exp.weight_file:
        ckp_path = cfg.exp.weight_file
    else:
        ckp_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)

    print('Done training...')


def test(cfg):
    assert args.weight or cfg.exp.weight_file, 'Please provide weight to test'
    weight = args.weight if args.weight else cfg.exp.weight_file

    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    _, val_loader = build_partnet_dataloader(cfg)

    trainer = pl.Trainer(gpus=[0])

    trainer.test(model, val_loader, ckpt_path=weight)

    print('Done testing...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--yml_file', required=True, type=str, help='.yml')
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true', help='FP16 training')
    parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
    parser.add_argument('--test', action='store_true', help='test model')
    args = parser.parse_args()

    cfg = importlib.import_module(args.cfg_file)
    cfg = cfg.get_cfg_defaults()
    cfg.merge_from_file(args.yml_file)

    if args.gpus == -1:
        args.gpus = [
            0,
        ]
    cfg.exp.gpus = args.gpus
    if args.weight:
        cfg.exp.weight_file = args.weight

    cfg.freeze()
    print(cfg)

    if args.test:
        test(cfg)
    else:
        main(cfg)
