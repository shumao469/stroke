from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import farthest_point_sample, index_points, knn_group

class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (SA) module with FPS + KNN grouping + shared MLP."""
    def __init__(self, npoint: int, k: int, in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_c = in_channel
        for out_c in mlp:
            self.mlp_convs.append(nn.Conv2d(last_c, out_c, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_c))
            last_c = out_c

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """Forward SA.

        xyz: (B, N, 3)
        points: (B, N, D) or None

        Returns:
          new_xyz: (B, npoint, 3)
          new_points: (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)              # (B, npoint, 3)

        idx = knn_group(xyz, new_xyz, self.k)             # (B, npoint, k)
        grouped_xyz = index_points(xyz, idx)              # (B, npoint, k, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, idx)    # (B, npoint, k, D)
            new_group = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_group = grouped_xyz_norm

        # (B, npoint, k, C) -> (B, C, npoint, k)
        new_group = new_group.permute(0, 3, 1, 2).contiguous()

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_group = F.relu(bn(conv(new_group)))

        # max pool over k neighbors -> (B, mlp[-1], npoint)
        new_points = torch.max(new_group, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B, npoint, mlp[-1])
        return new_xyz, new_points

class DBSPointNetPP(nn.Module):
    """PointNet++ for DBS efficacy prediction (regression + optional classification head)."""
    def __init__(self, in_channels: int = 4, n_classes: int = 1):
        super().__init__()
        # in_channels includes xyz(3) + features(?), but SA modules receive (xyz_norm+features)
        # SA input channel = 3 (xyz_norm) + (in_channels-3) feature dims
        feat_dim = max(in_channels - 3, 0)

        # SetAbstractionMsg 1: three radii branches in paper figure -> simplified as 3 MLP branches on same group
        # Here we keep three SA modules with different mlps and concatenate, matching DBS_PointNetPP.py structure.
        self.sa1_1 = SetAbstraction(npoint=2048, k=32, in_channel=3 + feat_dim, mlp=[64, 64, 128])
        self.sa1_2 = SetAbstraction(npoint=2048, k=32, in_channel=3 + feat_dim, mlp=[128, 128, 256])
        self.sa1_3 = SetAbstraction(npoint=2048, k=32, in_channel=3 + feat_dim, mlp=[128, 196, 256])

        self.sa2_1 = SetAbstraction(npoint=512, k=32, in_channel=3 + (128+256+256), mlp=[128, 128, 256])
        self.sa2_2 = SetAbstraction(npoint=512, k=32, in_channel=3 + (128+256+256), mlp=[256, 256, 512])
        self.sa2_3 = SetAbstraction(npoint=512, k=32, in_channel=3 + (128+256+256), mlp=[256, 384, 512])

        # SA3: global feature projection
        self.sa3_1 = nn.Sequential(nn.Conv1d(1280, 512, 1), nn.BatchNorm1d(512), nn.ReLU(True))
        self.sa3_2 = nn.Sequential(nn.Conv1d(512, 786, 1), nn.BatchNorm1d(786), nn.ReLU(True))
        self.sa3_3 = nn.Sequential(nn.Conv1d(786, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(True))

        # Heads
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.3))

        self.reg_head = nn.Linear(256, 1)
        self.cls_head = nn.Linear(256, n_classes) if n_classes > 0 else None

    def forward(self, x: torch.Tensor):
        """x: (B, N, C) with xyz first 3 columns."""
        xyz = x[:, :, :3]
        pts = x[:, :, 3:] if x.shape[-1] > 3 else None

        l1_xyz, l1_1 = self.sa1_1(xyz, pts)
        _,      l1_2 = self.sa1_2(xyz, pts)
        _,      l1_3 = self.sa1_3(xyz, pts)
        l1_points = torch.cat([l1_1, l1_2, l1_3], dim=-1)  # (B, 2048, 640)

        l2_xyz, l2_1 = self.sa2_1(l1_xyz, l1_points)
        _,      l2_2 = self.sa2_2(l1_xyz, l1_points)
        _,      l2_3 = self.sa2_3(l1_xyz, l1_points)
        l2_points = torch.cat([l2_1, l2_2, l2_3], dim=-1)  # (B, 512, 1280)

        # (B, 512, 1280) -> (B, 1280, 512)
        x3 = l2_points.permute(0, 2, 1).contiguous()
        x3 = self.sa3_1(x3)
        x3 = self.sa3_2(x3)
        x3 = self.sa3_3(x3)

        # Global max pool over points
        x_global = torch.max(x3, dim=-1)[0]  # (B, 1024)

        feat = self.fc2(self.fc1(x_global))
        pred_reg = self.reg_head(feat).squeeze(-1)
        pred_cls = self.cls_head(feat).squeeze(-1) if self.cls_head is not None else None
        return pred_reg, pred_cls
