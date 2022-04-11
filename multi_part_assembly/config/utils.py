import os
import sys
import importlib


def merge_cfg(base_cfg, base_dir, cfg_lst):
    """Merge a list of sub configs to the base config."""
    for k, v in cfg_lst.items():
        sys.path.append(os.path.join(base_dir, os.path.dirname(v)))
        lib = importlib.import_module(os.path.basename(v[:-3]))
        sub_cfg = lib.get_cfg_defaults()
        if k not in base_cfg:
            base_cfg[k] = sub_cfg
        # don't overwrite
        else:
            for key, value in sub_cfg.items():
                if key not in base_cfg[k]:
                    base_cfg[k][key] = value
    return base_cfg
