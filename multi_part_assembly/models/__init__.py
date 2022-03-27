from .modules import build_encoder, PointNet, DGCNN, PointNet2SSG, PointNet2MSG
from .modules import PoseRegressor, StocasticPoseRegressor, BaseModel
from .pn_transformer import PNTransformer, PNTransformerGAN, PNTransformerRefine


def build_model(cfg):
    if cfg.exp.name == 'pn_transformer':
        return PNTransformer(cfg)
    elif cfg.exp.name == 'pn_transformer_gan':
        return PNTransformerGAN(cfg)
    elif cfg.exp.name == 'pn_transformer_refine':
        return PNTransformerRefine(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.exp.name} not supported!')
