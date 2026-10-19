#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DBS PointNet++ 完整算法 Pipeline 演示 (Digital Twin Demo)
基于论文: PointNet++-based 3D point cloud deep learning for personalized DBS efficacy prediction
描述: 此脚本模拟从数据生成、预处理、模型训练到最终预测的全过程。
"""

import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time

# ==========================================
# 1. 配置与环境设置
# ==========================================
CONFIG = {
    'data_dir': './mock_dbs_data',  # 模拟数据存储路径
    'num_patients': 20,             # 模拟患者数量
    'points_per_sample': 4096,      # 采样点数 (论文中约227k, 演示用4096加速)
    'epochs': 5,                    # 演示训练轮数
    'batch_size': 4,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"🚀 启动 DBS 疗效预测算法 Pipeline...")
print(f"⚙️  运行设备: {CONFIG['device']}")

# ==========================================
# 2. 模拟数据生成器 (Data Simulation)
# ==========================================
def generate_mock_data():
    """
    模拟生成 .npz 文件。
    每个文件代表一个患者的样本，包含:
    - points: (N, 8) -> [x, y, z, tissue_id, e_field_mag, ex, ey, ez]
    - y_reg: UPDRS 改善率 (0-100)
    - y_cls: 是否响应 (0 或 1)
    """
    if os.path.exists(CONFIG['data_dir']):
        shutil.rmtree(CONFIG['data_dir'])
    os.makedirs(CONFIG['data_dir'])
    
    print(f"\n[Step 1] 正在生成模拟临床数据 ({CONFIG['num_patients']} 例)...")
    
    for i in tqdm(range(CONFIG['num_patients']), desc="生成患者数据"):
        # 1. 模拟空间坐标 (x, y, z) - 简单的高斯分布模拟脑区
        coords = np.random.randn(CONFIG['points_per_sample'], 3).astype(np.float32) * 10
        
        # 2. 模拟组织标签 (Tissue ID: 0=STN, 1=GPi, 2=White Matter)
        tissue_id = np.random.randint(0, 3, (CONFIG['points_per_sample'], 1)).astype(np.float32)
        
        # 3. 模拟电场 (Magnitude + Vector)
        # 假设电极中心在原点，电场随距离衰减
        dist = np.linalg.norm(coords, axis=1, keepdims=True)
        e_mag = 1000 / (dist + 1) + np.random.normal(0, 10, (CONFIG['points_per_sample'], 1))
        e_vec = coords / (dist + 1e-6) * e_mag # 简化的径向电场
        
        # 合并特征: [x, y, z, tissue, mag, vec_x, vec_y, vec_z]
        points = np.concatenate([coords, tissue_id, e_mag, e_vec], axis=1).astype(np.float32)
        
        # 4. 模拟临床结果
        # 假设靠近原点(电极)且电场强度适中的改善更好
        score = np.mean(e_mag) * 0.1 + np.random.normal(0, 5)
        y_reg = np.clip(score, 0, 100).astype(np.float32)
        y_cls = 1 if y_reg > 50 else 0
        
        np.savez(os.path.join(CONFIG['data_dir'], f'patient_{i:03d}.npz'), 
                 points=points, y_reg=y_reg, y_cls=y_cls)
    
    print(f"✅ 数据生成完毕: {CONFIG['data_dir']}")

# ==========================================
# 3. 数据集加载类 (Dataset)
# ==========================================
class DBSDataset(Dataset):
    def __init__(self, data_root, num_points=4096):
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"].astype(np.float32)
        y_reg = torch.tensor(data["y_reg"], dtype=torch.float32)
        y_cls = torch.tensor(data["y_cls"], dtype=torch.float32)

        # 简单采样
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        
        # 归一化 XYZ
        xyz = points[:, :3]
        xyz -= np.mean(xyz, axis=0)
        xyz /= (np.std(xyz, axis=0) + 1e-6)
        points[:, :3] = xyz
        
        return torch.from_numpy(points), y_reg, y_cls

# ==========================================
# 4. PointNet++ 核心模块 (简化版)
# ==========================================
def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS)"""
    B, N, C = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球状查询寻找邻居"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.cdist(new_xyz, xyz) ** 2
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C).to(xyz.device)
            grouped_points = torch.cat([xyz, points], dim=2).view(B, -1, 1, N) if points is not None else xyz.view(B, -1, 1, N)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)

            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
            else:
                new_points = grouped_xyz_norm

            new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
            grouped_points = new_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class DBSPointNetPP(nn.Module):
    def __init__(self, num_classes=1, normal_channel=True):
        super(DBSPointNetPP, self).__init__()
        in_channel = 8 if normal_channel else 3 # 3(coords) + 5(features)
        self.normal_channel = normal_channel
        
        # 模拟论文中的三层层级结构
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # 多任务输出头 (回归 + 分类)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.reg_head = nn.Linear(256, 1)     # UPDRS 预测
        self.cls_head = nn.Linear(256, 1)     # Responder 概率

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
            
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        
        reg_out = self.reg_head(x)
        cls_out = torch.sigmoid(self.cls_head(x))
        return reg_out, cls_out

# ==========================================
# 5. 执行逻辑 (Execution Logic)
# ==========================================
def main():
    # 1. 生成数据
    generate_mock_data()
    
    # 2. 准备数据集
    print(f"\n[Step 2] 正在加载 Dataset 和 DataLoader...")
    dataset = DBSDataset(CONFIG['data_dir'], num_points=CONFIG['points_per_sample'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    print(f"   - 训练集样本: {len(train_dataset)}")
    print(f"   - 验证集样本: {len(val_dataset)}")
    
    # 3. 初始化模型
    print(f"\n[Step 3] 初始化 PointNet++ 模型架构...")
    model = DBSPointNetPP(normal_channel=True).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion_reg = nn.L1Loss() # 论文提到的回归误差
    criterion_cls = nn.BCELoss()
    
    print(f"   - 模型已加载到: {CONFIG['device']}")
    
    # 4. 训练循环
    print(f"\n[Step 4] 开始训练 ({CONFIG['epochs']} Epochs)...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for points, y_reg, y_cls in pbar:
            points = points.transpose(2, 1).to(CONFIG['device']) # [B, C, N]
            y_reg = y_reg.unsqueeze(1).to(CONFIG['device'])
            y_cls = y_cls.unsqueeze(1).to(CONFIG['device'])
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(points)
            
            loss_reg = criterion_reg(pred_reg, y_reg)
            loss_cls = criterion_cls(pred_cls, y_cls)
            
            # 论文中的 Loss 组合 (Classification + Regression)
            total_loss = loss_reg + 0.5 * loss_cls
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
            
        print(f"   -> Epoch {epoch+1} 完成. Avg Loss: {train_loss/len(train_loader):.4f}")
        time.sleep(0.5) # 模拟一下停顿
    
    # 5. 推理演示
    print(f"\n[Step 5] 模拟新患者疗效预测 (Inference)...")
    model.eval()
    sample_points, true_reg, true_cls = val_dataset[0]
    sample_input = sample_points.unsqueeze(0).transpose(2, 1).to(CONFIG['device'])
    
    with torch.no_grad():
        start_time = time.time()
        pred_reg, pred_cls = model(sample_input)
        infer_time = time.time() - start_time
        
    print(f"--------------------------------------------------")
    print(f"📋 患者 ID: PATIENT_TEST_001")
    print(f"⏱️  推理耗时: {infer_time*1000:.2f} ms")
    print(f"🎯 真实 UPDRS 改善: {true_reg.item():.2f}")
    print(f"🤖 预测 UPDRS 改善: {pred_reg.item():.2f}")
    print(f"⚖️  预测误差: {abs(pred_reg.item() - true_reg.item()):.2f}")
    print(f"🔮 是否推荐手术 (Responder Prob): {pred_cls.item():.2%}")
    print(f"--------------------------------------------------")
    
    print("\n✅ 演示结束。这就是 PointNet++ DBS 疗效预测系统的完整工作流。")

if __name__ == '__main__':
    main()