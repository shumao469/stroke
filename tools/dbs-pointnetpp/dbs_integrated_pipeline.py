#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DBS 整合算法 Pipeline (Integrated Pipeline)
功能: 
1. 模拟/调用 Lead-DBS (MATLAB) 进行电极定位与电场仿真
2. 将 MATLAB 结果转换为 3D 点云
3. 使用修复后的 PointNet++ 进行疗效预测

修复说明:
- 修正了 PointNetSetAbstraction 中输入通道数不匹配导致的 RuntimeError (8 vs 11 channels)
- 修正了 group_all (SA3层) 的维度变换逻辑
- 移除了导致 SyntaxError 的非代码文本
"""

import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import sys

# 尝试导入 MATLAB 引擎 (如果未安装则使用 Mock 模式)
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

# ==========================================
# 1. 配置与环境
# ==========================================
CONFIG = {
    'raw_data_dir': './raw_dicom_data',      # 原始 MRI/CT 存放处
    'leaddbs_out_dir': './leaddbs_output',   # Lead-DBS 处理结果 (.mat)
    'pointcloud_dir': './processed_pcd',     # 转换后的点云数据 (.npz)
    'num_patients': 5,                       # 演示用患者数量
    'points_per_sample': 2048,               # 采样点数
    'batch_size': 2,
    'epochs': 3,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"🚀 启动 DBS 整合算法 Pipeline (Lead-DBS + PointNet++)...")
print(f"⚙️  MATLAB 引擎状态: {'✅ 已加载' if MATLAB_AVAILABLE else '⚠️ 未检测到 (将使用仿真模式)'}")
print(f"⚙️  AI 运行设备: {CONFIG['device']}")

# ==========================================
# 2. Lead-DBS 接口层 (MATLAB Integration)
# ==========================================
class LeadDBSAdapter:
    """
    负责与 Lead-DBS (MATLAB) 交互的适配器
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def run_preprocessing(self, patient_id):
        """
        调用 Lead-DBS 流程:
        1. DICOM -> NIfTI
        2. Coregistration & Normalization
        3. Electrode Reconstruction (PaCER)
        4. VTA/E-Field Simulation (SimBio/FieldTrip)
        """
        print(f"\n[Lead-DBS] 正在处理患者: {patient_id} ...")
        
        if MATLAB_AVAILABLE:
            # 真实调用逻辑 (示例伪代码)
            # eng = matlab.engine.start_matlab()
            # eng.lead_dbs_run(patient_id, nargout=0)
            pass
        else:
            # === 仿真模式 (Mock Execution) ===
            steps = [
                "1. Converting DICOM to NIfTI...",
                "2. Normalizing to MNI Space (ANTs)...",
                "3. Reconstructing Electrodes (PaCER)...",
                "4. Simulating Electric Field (FEM/SimBio)..."
            ]
            for step in steps:
                time.sleep(0.3) # 模拟耗时
                print(f"   >> {step}")
            
            # 生成模拟的 .mat 结果文件
            self._generate_mock_mat_result(patient_id)
            print(f"   ✅ Lead-DBS 处理完成. 结果已保存.")

    def _generate_mock_mat_result(self, patient_id):
        """生成模拟的 ea_reconstruction.mat 数据"""
        # 这里实际上保存为 .npz 来模拟 mat 的内容，方便 python 读取
        # 在真实场景中，这里是 Lead-DBS 生成的 .mat 文件
        mat_path = os.path.join(self.output_dir, f'{patient_id}_result.npz')
        
        # 模拟电场和解剖数据
        N = CONFIG['points_per_sample'] * 2 # 原始数据通常点更多
        coords = np.random.randn(N, 3).astype(np.float32) * 10
        e_field = np.random.rand(N, 1).astype(np.float32) * 500 # V/m
        anatomy = np.random.randint(0, 5, (N, 1)).astype(np.float32) # 0=STN, 1=GPi...
        
        # 模拟临床分数
        updrs_score = np.random.uniform(10, 80)
        
        np.savez(mat_path, coords=coords, e_field=e_field, anatomy=anatomy, updrs=updrs_score)

# ==========================================
# 3. 数据转换层 (Converter)
# ==========================================
def convert_mat_to_pointcloud():
    """将 Lead-DBS 的输出转换为 PointNet++ 输入格式"""
    if os.path.exists(CONFIG['pointcloud_dir']):
        shutil.rmtree(CONFIG['pointcloud_dir'])
    os.makedirs(CONFIG['pointcloud_dir'])
    
    mat_files = glob.glob(os.path.join(CONFIG['leaddbs_out_dir'], "*.npz"))
    print(f"\n[Converter] 正在将 Lead-DBS 结果转换为点云格式...")
    
    for f in tqdm(mat_files, desc="Converting"):
        data = np.load(f)
        patient_id = os.path.basename(f).split('_')[0]
        
        # 提取特征: [x, y, z, anatomy, e_field, vec_x, vec_y, vec_z]
        # 这里简单模拟 vector
        vec = data['coords'] / (np.linalg.norm(data['coords'], axis=1, keepdims=True) + 1e-6)
        
        features = np.concatenate([
            data['coords'], 
            data['anatomy'], 
            data['e_field'], 
            vec
        ], axis=1).astype(np.float32) # Shape (N, 8)
        
        # 临床标签
        y_reg = data['updrs']
        y_cls = 1 if y_reg > 40 else 0
        
        np.savez(os.path.join(CONFIG['pointcloud_dir'], f'{patient_id}.npz'),
                 points=features, y_reg=y_reg, y_cls=y_cls)

# ==========================================
# 4. 修复后的 PointNet++ 网络
# ==========================================
def farthest_point_sample(xyz, npoint):
    """FPS 采样"""
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
    """Ball Query"""
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
        self.group_all = group_all

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1) # [B, 3, N]
        if points is not None:
            points = points.permute(0, 2, 1) # [B, D, N]

        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C).to(xyz.device)
            # 组合几何坐标和特征
            if points is not None:
                grouped_points = torch.cat([xyz, points], dim=2) # [B, N, 3+D]
            else:
                grouped_points = xyz
            
            # [B, N, 3+D] -> [B, 3+D, N] -> [B, 3+D, N, 1]
            # 维度含义: Batch, Channels, nsample(N), npoint(1)
            grouped_points = grouped_points.permute(0, 2, 1).unsqueeze(3)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) 
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C) # [B, npoint, nsample, 3]

            if points is not None:
                grouped_points = index_points(points, idx) # [B, npoint, nsample, D]
                # 关键修复点: 拼接后的维度是 D + 3
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
            else:
                new_points = grouped_xyz_norm

            new_points = new_points.permute(0, 3, 2, 1) # [B, D+3, nsample, npoint]
            grouped_points = new_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class DBSPointNetPP(nn.Module):
    def __init__(self, normal_channel=True):
        super(DBSPointNetPP, self).__init__()
        
        # 原始特征维度: 8 (3 coords + 5 feats) 或 3 (only coords)
        input_feature_dim = 8 if normal_channel else 3
        
        # === 关键修复 ===
        # SetAbstraction 会将局部坐标(3)拼接到输入特征上。
        # SA1 输入: input_feature_dim (8) + 3 = 11
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=input_feature_dim + 3, # FIX: 8+3=11
                                          mlp=[64, 64, 128], group_all=False)
        
        # SA2 输入: SA1输出(128) + 3 = 131
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                          in_channel=128 + 3, # FIX: 128+3=131
                                          mlp=[128, 128, 256], group_all=False)
        
        # SA3 输入: SA2输出(256) + 3 = 259
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                          in_channel=256 + 3, 
                                          mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.reg_head = nn.Linear(256, 1)
        self.cls_head = nn.Linear(256, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        # PointNet++ SetAbstraction 期望输入格式:
        # xyz: [B, 3, N] (坐标)
        # points: [B, D, N] (额外特征)
        
        l0_xyz = xyz[:, :3, :] # 前3维是坐标
        l0_points = xyz        # 这里的 points 包含了坐标本身作为特征
            
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
# 5. Dataset 定义
# ==========================================
class DBSDataset(Dataset):
    def __init__(self, data_root, num_points=2048):
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"].astype(np.float32) # (N, 8)
        y_reg = torch.tensor(data["y_reg"], dtype=torch.float32)
        y_cls = torch.tensor(data["y_cls"], dtype=torch.float32)

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        
        # 归一化
        xyz = points[:, :3]
        xyz -= np.mean(xyz, axis=0)
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz /= (max_dist + 1e-6)
        points[:, :3] = xyz
        
        return torch.from_numpy(points), y_reg, y_cls

# ==========================================
# 6. 主程序 Loop
# ==========================================
def main():
    # --- Step 1: Lead-DBS 预处理 ---
    print(f"\n======== Phase 1: Lead-DBS 影像处理 ========")
    dbs_runner = LeadDBSAdapter(CONFIG['leaddbs_out_dir'])
    
    # 模拟对每个患者进行处理
    patient_ids = [f'PD_PATIENT_{i:03d}' for i in range(CONFIG['num_patients'])]
    for pid in patient_ids:
        dbs_runner.run_preprocessing(pid)
        
    # --- Step 2: 格式转换 ---
    print(f"\n======== Phase 2: 点云生成 (Voxel to Point Cloud) ========")
    convert_mat_to_pointcloud()
    
    # --- Step 3: AI 训练 ---
    print(f"\n======== Phase 3: AI 模型训练 (PointNet++) ========")
    dataset = DBSDataset(CONFIG['pointcloud_dir'], num_points=CONFIG['points_per_sample'])
    
    # 避免样本过少导致报错
    if len(dataset) < 2:
        print("样本过少，复制样本以演示训练...")
        dataset.files = dataset.files * 4 
        
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = DBSPointNetPP(normal_channel=True).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion_reg = nn.L1Loss()
    criterion_cls = nn.BCELoss()
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for points, y_reg, y_cls in pbar:
            # 输入: [B, N, C] -> Transpose -> [B, C, N]
            points = points.transpose(2, 1).to(CONFIG['device'])
            y_reg = y_reg.unsqueeze(1).to(CONFIG['device'])
            y_cls = y_cls.unsqueeze(1).to(CONFIG['device'])
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(points)
            
            loss = criterion_reg(pred_reg, y_reg) + 0.5 * criterion_cls(pred_cls, y_cls)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    print(f"\n✅ 训练完成! 模型已就绪.")
    
    # --- Step 4: 模拟预测 ---
    print(f"\n======== Phase 4: 临床疗效预测演示 ========")
    model.eval()
    test_sample = dataset[0][0].unsqueeze(0).transpose(2, 1).to(CONFIG['device']) # [1, C, N]
    
    with torch.no_grad():
        pred_reg, pred_cls = model(test_sample)
        
    print(f"输入: 患者 {patient_ids[0]} 的电场点云数据")
    print(f"预测 UPDRS 改善值: {pred_reg.item():.2f}")
    print(f"预测 响应概率: {pred_cls.item()*100:.2f}%")
    print(f"结论: {'建议手术/调参' if pred_cls.item()>0.5 else '建议重新规划'}")

if __name__ == '__main__':
    main()