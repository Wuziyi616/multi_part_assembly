import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (idx + idx_base).view(-1)

    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  
                                        # -> (batch_size*num_points, num_dims) 
                                        #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    """DGCNN feature extractor.
    Input point clouds [B, 3, N], output per-point features [B, C, N].
    """

    def __init__(self, feat_dim):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(feat_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """x: [B, C, N]"""
        batch_size, num_dims, num_points = x.size()                 # x:      batch x   3 x num of points
        x      = get_graph_feature(x)                               # x:      batch x   6 x num of points x 20

        x1     = self.conv1(x)                                      # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]                    # x1_max: batch x  64 x num of points x 1

        x2     = self.conv2(x1)                                     # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]                    # x2_max: batch x  64 x num of points x 1

        x3     = self.conv3(x2)                                     # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]                    # x3_max: batch x 128 x num of points x 1

        x4     = self.conv4(x3)                                     # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]                    # x4_max: batch x 256 x num of points x 1
 
        x_max  = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1) # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)        # point feat:  batch x feat_dim x num of points

        return point_feat
