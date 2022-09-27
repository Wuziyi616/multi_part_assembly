import os
import sys
import pwd
import argparse
import importlib

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import PCAssemblyLogCallback


def main(cfg):
    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    train_loader, val_loader = build_dataloader(cfg)

    # Create checkpoint directory
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    cfg_name = os.path.basename(args.cfg_file)[:-3]  # remove '.py'
    ckp_dir = os.path.join(cfg.exp.ckp_dir, cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)

    # on clusters, quota under user dir is usually limited
    # soft link to save the weights in temp space for checkpointing
    # TODO: modify this if you are not running on clusters
    CHECKPOINT_DIR = '/checkpoint/'  # ''
    if SLURM_JOB_ID and CHECKPOINT_DIR and os.path.isdir(CHECKPOINT_DIR):
        if not os.path.exists(ckp_dir):
            # on my cluster, the temp dir is /checkpoint/$USER/$SLURM_JOB_ID
            # TODO: modify this if your cluster is different
            usr = pwd.getpwuid(os.getuid())[0]
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                usr, SLURM_JOB_ID, ckp_dir))
    else:
        os.makedirs(ckp_dir, exist_ok=True)

    # it's not good to hard-code the wandb id
    # but on preemption clusters, we want the job to resume the same wandb
    # process after resuming training (i.e. drawing the same graph)
    # so we have to keep the same wandb id
    # TODO: modify this if you are not running on preemption clusters
    preemption = True  # False
    if SLURM_JOB_ID and preemption:
        logger_id = logger_name = f'{cfg_name}-{SLURM_JOB_ID}'
    else:
        logger_name = cfg_name
        logger_id = None

    # configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-{epoch:03d}',
        monitor='val/part_acc',
        save_top_k=5,
        mode='max',
    )
    callbacks = [
        LearningRateMonitor('epoch'),
        checkpoint_callback,
    ]
    # visualize assembly results
    if args.vis:
        assembly_callback = PCAssemblyLogCallback(cfg.exp.val_sample_vis,
                                                  train_loader, val_loader)
        callbacks.append(assembly_callback)

    logger = WandbLogger(
        project='Multi-Part-Assembly',
        name=logger_name,
        id=logger_id,
        save_dir=ckp_dir,
    )

    all_gpus = list(cfg.exp.gpus)
    trainer = pl.Trainer(
        logger=logger,
        gpus=all_gpus,
        strategy=parallel_strategy if len(all_gpus) > 1 else None,
        max_epochs=cfg.exp.num_epochs,
        callbacks=callbacks,
        precision=16 if args.fp16 else 32,  # FP16 training
        benchmark=args.cudnn,  # cudnn benchmark
        gradient_clip_val=cfg.optimizer.clip_grad,  # clip grad norm
        check_val_every_n_epoch=cfg.exp.val_every,
        log_every_n_steps=50,
        profiler='simple',  # training time bottleneck analysis
        # detect_anomaly=True,  # for debug
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
        # check if it has trainint states, or just a model weight
        ckp = torch.load(cfg.exp.weight_file, map_location='cpu')
        # if it has, then it's a checkpoint compatible with pl
        if 'state_dict' in ckp.keys():
            ckp_path = cfg.exp.weight_file
        # if it's just a weight, then manually load it to the model
        else:
            ckp_path = None
            model.load_state_dict(ckp)
    else:
        ckp_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)

    print('Done training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true', help='FP16 training')
    parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
    parser.add_argument('--vis', action='store_true', help='visualize results')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()

    # TODO: modify this if you cannot run DDP training, and want to use DP
    parallel_strategy = 'ddp'  # 'dp'
    cfg.exp.gpus = args.gpus
    # manually increase batch_size according to the number of GPUs in DP
    # not necessary in DDP because it's already per-GPU batch size
    if len(cfg.exp.gpus) > 1 and parallel_strategy == 'dp':
        cfg.exp.batch_size *= len(cfg.exp.gpus)
        cfg.exp.num_workers *= len(cfg.exp.gpus)
    if args.category:
        cfg.data.category = args.category
    if args.weight:
        cfg.exp.weight_file = args.weight

    cfg.freeze()
    print(cfg)

    main(cfg)
