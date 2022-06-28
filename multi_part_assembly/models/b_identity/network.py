from multi_part_assembly.models import BaseModel


class IdentityModel(BaseModel):
    """Trivial model that always returns identity transformation."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict should contains:
                - part_pcs: [B, P, N, 3]
        """
        part_pcs = data_dict['part_pcs']
        B, P = part_pcs.shape[:2]
        zero_pose = self.zero_pose.repeat(B, P, 1).type_as(part_pcs)
        rot = self._wrap_rotation(zero_pose[..., :-3])
        trans = zero_pose[..., -3:]

        pred_dict = {
            'rot': rot,  # [B, P, 4/(3, 3)], Rotation3D
            'trans': trans,  # [B, P, 3]
            'pre_pose_feats': None,  # useless
        }
        return pred_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss.

        Since there could be several parts that are the same in one shape, we
            need to do Hungarian matching to find the min loss values.

        Args:
            data_dict: the data loaded from dataloader
            pre_pose_feats: because the stochasticity is only in the final pose
                regressor, we can reuse all the computed features before

        Returns a dict of loss, each is a [B] shape tensor for later selection.
        See GNN Assembly paper Sec 3.4, the MoN loss is sampling prediction
            several times and select the min one as final loss.
            Also returns computed features before pose regressing for reusing.
        """
        forward_dict = {
            'part_pcs': data_dict['part_pcs'],
        }

        # prediction
        out_dict = self.forward(forward_dict)

        # loss computation
        loss_dict, out_dict = self._calc_loss(out_dict, data_dict)

        return loss_dict, out_dict

    def load_state_dict(self, *args, **kwargs):
        pass
