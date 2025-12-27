#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dbs_pointnetpp.py
# -*- coding: utf-8 -*-
"""
PointNet++-style 3D point cloud deep learning for personalized DBS efficacy prediction.

依赖:
    pip install numpy torch scikit-learn
"""

import os
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


# =========================
# 1. 数据集定义
# =========================

class DBSPointCloudDataset(Dataset):
    """
    读取每个 .npz 文件中的点云和标签:
        - points: (N, C)
        - y_reg: scalar, regression target (UPDRS improvement)
        - y_cls: scalar, 0/1, responder vs non-responder
    """
    def __init__(self, data_root, num_points=20000, normalize=True):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        assert len(self.files) > 0, f"No .npz files found under {data_root}"
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        data = np.load(f)

        points = data["points"].astype(np.float32)  # (N, C)
        y_reg = np.float32(data["y_reg"])
        y_cls = np.int64(data["y_cls"])

        # 随机下采样到固定点数 (或重复采样)
        N = points.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            # 不足时有放回重复采样
            extra = np.random.choice(N, self.num_points - N, replace=True)
            choice = np.concatenate([np.arange(N), extra], axis=0)
        points = points[choice, :]

        # 简单归一化: 几何坐标减均值/除标准差
        if self.normalize:
            xyz = points[:, :3]
            xyz_mean = xyz.mean(axis=0, keepdims=True)
            xyz_std = xyz.std(axis=0, keepdims=True) + 1e-6
            points[:, :3] = (xyz - xyz_mean) / xyz_std

            # 电场幅值可以做 log1p 或标准化视情况而定
            # 这里做一个简单的标准化
            feat = points[:, 3:]
            feat_mean = feat.mean(axis=0, keepdims=True)
            feat_std = feat.std(axis=0, keepdims=True) + 1e-6
            points[:, 3:] = (feat - feat_mean) / feat_std

        points = torch.from_numpy(points)   # (N, C)
        y_reg = torch.tensor(y_reg)
        y_cls = torch.tensor(y_cls)

        return points, y_reg, y_cls


# =========================
# 2. PointNet++ 相关工具
# =========================

def farthest_point_sample(x, npoint):
    """
    最远点采样 (Farthest Point Sampling)
    输入:
        x: (B, N, 3)  只用几何坐标做采样
    输出:
        centroids_idx: (B, npoint) 采样点在原始点云中的索引
    """
    device = x.device
    B, N, _ = x.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = x[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
    根据索引取点
    points: (B, N, C)
    idx: (B, S) 或 (B, S, K)
    输出 shapes 对应
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(idx.shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_group(xyz, points, npoint, k):
    """
    简化版 PointNet++: 使用 FPS + kNN 分组 (无半径限制)
    输入:
        xyz: (B, N, 3)
        points: (B, N, C)
    输出:
        new_xyz: (B, npoint, 3)
        new_points: (B, npoint, k, C+3)
    """
    B, N, C_xyz = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)     # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)             # (B, npoint, 3)

    # 计算每个采样点到所有点的距离, 取最近 k 个
    # dist: (B, npoint, N)
    dist = torch.cdist(new_xyz, xyz)    # 计算欧氏距离
    _, group_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)  # (B, npoint, k)

    grouped_xyz = index_points(xyz, group_idx)       # (B, npoint, k, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)   # 相对坐标 (局部坐标系)

    if points is not None:
        grouped_points = index_points(points, group_idx)    # (B, npoint, k, C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, k, C+3)
    else:
        new_points = grouped_xyz_norm   # 只有几何信息

    return new_xyz, new_points


class SetAbstraction(nn.Module):
    """
    简化版 PointNet++ Set Abstraction (SSG: single-scale grouping)
    """
    def __init__(self, npoint, k, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.k = k

        layers = []
        last_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv2d(last_c, out_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        xyz:    (B, N, 3)
        points: (B, N, C) 或 None
        输出:
            new_xyz:    (B, npoint, 3)
            new_points: (B, npoint, C_out)
        """
        new_xyz, new_points = knn_group(xyz, points, self.npoint, self.k)
        # new_points: (B, npoint, k, in_channels)
        B, S, K, C_in = new_points.shape

        # 转置成 (B, C_in, S, K) 方便 1×1 卷积
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)   # (B, C_out, S, K)
        new_points = torch.max(new_points, dim=-1)[0]  # (B, C_out, S)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B, S, C_out)

        return new_xyz, new_points


# =========================
# 3. DBS PointNet++ 模型
# =========================

class DBSPointNetPP(nn.Module):
    def __init__(self, in_channels, use_regression=True, use_classification=True):
        super().__init__()
        self.use_regression = use_regression
        self.use_classification = use_classification

        # 三层 SA，参数可根据你自己的点数调
        self.sa1 = SetAbstraction(
            npoint=1024, k=32,
            in_channels=in_channels + 3,      # (局部坐标3 + features)
            mlp_channels=[64, 64, 128]
        )
        self.sa2 = SetAbstraction(
            npoint=256, k=32,
            in_channels=128 + 3,
            mlp_channels=[128, 128, 256]
        )
        self.sa3 = SetAbstraction(
            npoint=64, k=32,
            in_channels=256 + 3,
            mlp_channels=[256, 512, 1024]
        )

        # 全局池化后, 再接回归头/分类头
        feat_dim = 1024

        if use_regression:
            self.reg_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1)
            )

        if use_classification:
            self.cls_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1)    # 输出 logit, 后面再 sigmoid
            )

    def forward(self, x):
        """
        x: (B, N, C) 全部特征, 前 3 维是 xyz
        """
        xyz = x[:, :, :3]
        points = x[:, :, 3:]

        # SA 层
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # 全局 maxpool
        global_feat = torch.max(l3_points, dim=1)[0]  # (B, 1024)

        out_reg = None
        out_cls = None

        if self.use_regression:
            out_reg = self.reg_head(global_feat).squeeze(-1)  # (B,)

        if self.use_classification:
            logits = self.cls_head(global_feat).squeeze(-1)   # (B,)
            out_cls = logits

        return out_reg, out_cls


# =========================
# 4. 训练 & AUC 评估
# =========================

def train_one_epoch(model, loader, optimizer, device, lambda_reg=0.5):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for points, y_reg, y_cls in loader:
        points = points.to(device)          # (B, N, C)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device).float()

        optimizer.zero_grad()
        pred_reg, pred_logits = model(points)

        loss = 0.0

        if pred_logits is not None:
            cls_loss = bce(pred_logits, y_cls)
            loss = loss + cls_loss
            total_cls_loss += cls_loss.item() * points.size(0)

        if pred_reg is not None:
            reg_loss = l1_loss(pred_reg, y_reg)
            loss = loss + lambda_reg * reg_loss
            total_reg_loss += reg_loss.item() * points.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)

    n = len(loader.dataset)
    return (total_loss / n,
            total_reg_loss / n if total_reg_loss > 0 else 0.0,
            total_cls_loss / n if total_cls_loss > 0 else 0.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []
    y_reg_true = []
    y_reg_pred = []

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0

    for points, y_reg, y_cls in loader:
        points = points.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device).float()

        pred_reg, pred_logits = model(points)

        if pred_logits is not None:
            cls_loss = bce(pred_logits, y_cls)
            total_cls_loss += cls_loss.item() * points.size(0)
            prob = torch.sigmoid(pred_logits)
            y_true.append(y_cls.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

        if pred_reg is not None:
            reg_loss = l1_loss(pred_reg, y_reg)
            total_reg_loss += reg_loss.item() * points.size(0)
            y_reg_true.append(y_reg.cpu().numpy())
            y_reg_pred.append(pred_reg.cpu().numpy())

        total_samples += points.size(0)

    metrics = {}

    if len(y_true) > 0:
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)

        metrics["cls_loss"] = total_cls_loss / total_samples
        metrics["auc_roc"] = auc_roc
        metrics["auc_pr"] = auc_pr
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["prec"] = prec
        metrics["rec"] = rec

    if len(y_reg_true) > 0:
        y_reg_true = np.concatenate(y_reg_true, axis=0)
        y_reg_pred = np.concatenate(y_reg_pred, axis=0)
        mae = np.mean(np.abs(y_reg_true - y_reg_pred))
        metrics["reg_loss"] = total_reg_loss / total_samples
        metrics["mae"] = mae

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to .npz files")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_points", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.5, help="Weight of regression loss")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 数据集
    tmp_dataset = DBSPointCloudDataset(args.data_root, num_points=args.num_points)
    in_channels = tmp_dataset[0][0].shape[1] - 3
    print(f"Detected point feature dim (C) = {in_channels}")

    # train/val split
    total_len = len(tmp_dataset)
    val_len = int(total_len * args.val_ratio)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(tmp_dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2) 模型
    model = DBSPointNetPP(in_channels=in_channels, use_regression=True, use_classification=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_auc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_reg_loss, train_cls_loss = train_one_epoch(
            model, train_loader, optimizer, device, lambda_reg=args.lambda_reg
        )

        metrics = evaluate(model, val_loader, device)
        auc_roc = metrics.get("auc_roc", 0.0)
        auc_pr = metrics.get("auc_pr", 0.0)
        val_reg = metrics.get("reg_loss", 0.0)
        val_cls = metrics.get("cls_loss", 0.0)

        print(f"Epoch {epoch:03d}: "
              f"TrainLoss={train_loss:.4f} (reg={train_reg_loss:.4f}, cls={train_cls_loss:.4f}) | "
              f"ValLoss(reg={val_reg:.4f}, cls={val_cls:.4f}) | "
              f"AUC ROC={auc_roc:.4f}, AUC PR={auc_pr:.4f}")

        # 以 AUC ROC 作为 early stopping / 模型选择指标
        if auc_roc > best_auc:
            best_auc = auc_roc
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr
            }

    if best_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        save_path = os.path.join("checkpoints", f"dbs_pointnetpp_best_auc_{best_state['auc_roc']:.4f}.pth")
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path}, epoch={best_state['epoch']}, "
              f"AUC ROC={best_state['auc_roc']:.4f}, AUC PR={best_state['auc_pr']:.4f}")


if __name__ == "__main__":
    main()


# In[ ]:




