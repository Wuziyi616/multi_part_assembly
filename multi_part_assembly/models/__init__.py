from .encoder import build_encoder, PointNet, DGCNN, PointNet2SSG, PointNet2MSG
from .pn_transformer import PNTransformer, PNTransformerGAN


def build_model(cfg):
    if cfg.exp.name == 'pn_transformer':
        return PNTransformer(cfg)
    if cfg.exp.name == 'pn_transformer_gan':
        return PNTransformerGAN(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.exp.name} not supported!')
