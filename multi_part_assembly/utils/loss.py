import torch

from .transforms import qrot, qtransform, qeuler, get_sym_point_list
from .chamfer import chamfer_distance


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_l2_loss(trans1, trans2, valids):
    """L2 loss for transformation.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_l2_loss(quat1, quat2, valids):
    """L2 loss for rotation.

    Args:
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    # since quat == -quat
    rot_l2_1 = (quat1 - quat2).pow(2).sum(dim=-1)  # [B, P]
    rot_l2_2 = (quat1 + quat2).pow(2).sum(dim=-1)
    loss_per_data = torch.minimum(rot_l2_1, rot_l2_2)
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_cosine_loss(quat1, quat2, valids):
    """Cosine loss for rotation.

    Args:
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    # since quat == -quat
    loss_per_data = 1. - torch.abs(torch.sum(quat1 * quat2, dim=-1))  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_points_l2_loss(pts, quat1, quat2, valids, ret_pts=False):
    """L2 distance between point clouds transformed by quats.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    pts1 = qrot(quat1, pts)
    pts2 = qrot(quat2, pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def rot_points_cd_loss(pts, quat1, quat2, valids, ret_pts=False):
    """Chamfer distance between point clouds transformed by quats.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    batch_size = pts.shape[0]

    pts1 = qrot(quat1, pts)
    pts2 = qrot(quat2, pts)

    dist1, dist2 = chamfer_distance(pts1.flatten(0, 1), pts2.flatten(0, 1))
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1).type_as(pts)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def shape_cd_loss(pts, trans1, trans2, quat1, quat2, valids, ret_pts=False):
    """Chamfer distance between point clouds after rotation and translation.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the transformed point clouds

    Returns:
        [B], loss per data in the batch
    """
    batch_size = pts.shape[0]
    num_points = pts.shape[2]

    pts1 = qtransform(trans1, quat1, pts)
    pts2 = qtransform(trans2, quat2, pts)

    shape1 = pts1.flatten(1, 2)  # [B, P*N, 3]
    shape2 = pts2.flatten(1, 2)
    # TODO: the padded 0 points may break the chamfer here?
    # TODO: i.e. some points' corresponding points is the padded points
    dist1, dist2 = chamfer_distance(shape1, shape2)  # [B, P*N]

    valids = valids.float().detach()
    valids = valids.unsqueeze(2).repeat(1, 1, num_points).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    # TODO: should use `_valid_mean` instead of directly taking mean?
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def repulsion_cd_loss(part_pcs, valids, thre):
    """Compute Chamfer distance as repulsion loss to push parts away.

    Args:
        part_pcs: [B, P, N, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        thre: float, only penalize cd smaller than this value

    Returns:
        [B], loss per data in the batch
    """
    B, P, N, _ = part_pcs.shape
    pts1 = part_pcs.unsqueeze(2).repeat_interleave(P, dim=2).flatten(0, 2)
    pts2 = part_pcs.unsqueeze(1).repeat_interleave(P, dim=1).flatten(0, 2)
    dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P*P, N]
    cd = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)  # [B*P*P]
    cd = torch.clamp_min(thre - cd.view(B, P, P), min=0.)  # [B, P, P]
    valid_mask = torch.ones(B, P, P).type_as(valids) * \
        valids[:, :, None] * valids[:, None, :]
    loss_per_data = (cd * valid_mask).sum([1, 2]) / valid_mask.sum([1, 2])
    return loss_per_data


def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for transformation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    if metric == 'mse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


def rot_metrics(quat1, quat2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    deg1 = qeuler(quat1, order='zyx', to_degree=True)
    deg2 = qeuler(quat2, order='zyx', to_degree=True)
    # since euler angle has the discontinuity at 180
    # -179 and +179 actually only has an error of 2 degree
    # convert -179 to 181
    deg2_offset = deg2 + 360.
    diff1 = (deg1 - deg2).abs()
    diff2 = (deg1 - deg2_offset).abs()
    deg2 = torch.where(diff1 < diff2, deg2, deg2_offset)
    if metric == 'mse':
        metric_per_data = (deg1 - deg2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (deg1 - deg2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (deg1 - deg2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


def calc_part_acc(pts, trans1, trans2, quat1, quat2, valids):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    B, P, N, _ = pts.shape

    pts1 = qtransform(trans1, quat1, pts)  # [B, P, N, 3]
    pts2 = qtransform(trans2, quat2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P, N]
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc.sum(-1) / (valids == 1).sum(-1)
    return acc


def calc_connectivity_acc(trans, quat, contact_points):
    """Compute the `Connectivity Accuracy` in the paper.

    We transform pre-computed connected point pairs using predicted pose, then
        we compare the distance between them.
    Distance lower than a threshold will be considered as correct.

    Args:
        trans: [B, P, 3]
        quat: [B, P, 4]
        contact_points: [B, P, P, 4], pairwise contact matrix.
            First item is 1 --> two parts are connecting, 0 otherwise.
            Last three items are the contacting point coordinate.

    Returns:
        [B], accuracy per data in the batch
    """
    B, P, _ = trans.shape
    thre = 0.01

    def get_min_l2_dist(points1, points2, trans1, trans2, quat1, quat2):
        """Compute the min L2 distance between two set of points."""
        # points1/2: [num_contact, num_symmetry, 3]
        # trans/quat: [num_contact, 3/4]
        points1 = qtransform(trans1, quat1, points1)
        points2 = qtransform(trans2, quat2, points2)
        dist = ((points1[:, :, None] - points2[:, None, :])**2).sum(-1)
        return dist.min(-1)[0].min(-1)[0]  # [num_contact]

    # find all contact points
    mask = (contact_points[..., 0] == 1)  # [B, P, P]
    # points1 = contact_points[mask][..., 1:]
    # TODO: more efficient way of getting paired contact points?
    points1, points2, trans1, trans2, quat1, quat2 = [], [], [], [], [], []
    for b in range(B):
        for i in range(P):
            for j in range(P):
                if mask[b, i, j]:
                    points1.append(contact_points[b, i, j, 1:])
                    points2.append(contact_points[b, j, i, 1:])
                    trans1.append(trans[b, i])
                    trans2.append(trans[b, j])
                    quat1.append(quat[b, i])
                    quat2.append(quat[b, j])
    points1 = torch.stack(points1, dim=0)  # [n, 3]
    points2 = torch.stack(points2, dim=0)  # [n, 3]
    # [n, 3/4], corresponding translation and rotation
    trans1, trans2 = torch.stack(trans1, dim=0), torch.stack(trans2, dim=0)
    quat1, quat2 = torch.stack(quat1, dim=0), torch.stack(quat2, dim=0)
    points1 = torch.stack(get_sym_point_list(points1), dim=1)  # [n, sym, 3]
    points2 = torch.stack(get_sym_point_list(points2), dim=1)  # [n, sym, 3]
    dist = get_min_l2_dist(points1, points2, trans1, trans2, quat1, quat2)
    acc = (dist < thre).sum().float() / float(dist.numel())

    # the official code is doing avg per-contact_point acc (not per-shape)
    # so we tile the `acc` to [B]
    acc = torch.ones(B).type_as(trans) * acc
    return acc
