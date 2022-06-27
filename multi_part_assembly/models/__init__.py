from .modules import *
from .pn_transformer import PNTransformer, PNTransformerGAN, PNTransformerRefine
from .b_global import GlobalModel
from .b_lstm import LSTMModel
from .dgl import DGLModel
from .rgl_net import RGLNet


def build_model(cfg):
    if cfg.model.name == 'global':
        return GlobalModel(cfg)
    elif cfg.model.name == 'lstm':
        return LSTMModel(cfg)
    elif cfg.model.name == 'dgl':
        return DGLModel(cfg)
    elif cfg.model.name == 'rgl_net':
        return RGLNet(cfg)
    elif cfg.model.name == 'pn_transformer':
        return PNTransformer(cfg)
    elif cfg.model.name == 'pn_transformer_gan':
        return PNTransformerGAN(cfg)
    elif cfg.model.name == 'pn_transformer_refine':
        return PNTransformerRefine(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.model.name} not supported!')
