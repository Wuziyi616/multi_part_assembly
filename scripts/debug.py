import os
import sys
import importlib

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model


def build(cfg_file):
    sys.path.append(os.path.dirname(cfg_file))
    cfg = importlib.import_module(os.path.basename(cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()

    cfg.freeze()
    print(cfg)

    # Initialize model
    model = build_model(cfg)

    # Initialize dataloaders
    train_loader, val_loader = build_dataloader(cfg)

    return model, train_loader, val_loader, cfg
