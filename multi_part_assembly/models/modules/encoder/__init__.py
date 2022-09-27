from .pointnet import PointNet
from .dgcnn import DGCNN
from .pointnet2 import PointNet2SSG, PointNet2MSG


def build_encoder(arch, feat_dim, global_feat=True, **kwargs):
    if arch == 'pointnet':
        model = PointNet(feat_dim, global_feat=global_feat)
    elif arch == 'dgcnn':
        model = DGCNN(feat_dim, global_feat=global_feat)
    elif 'pointnet2' in arch:
        assert global_feat
        if 'ssg' in arch:
            model = PointNet2SSG(feat_dim)
        elif 'msg' in arch:
            model = PointNet2MSG(feat_dim)
        else:
            raise NotImplementedError(f'{arch} not supported')
    else:
        raise NotImplementedError(f'{arch} is not supported')
    return model
