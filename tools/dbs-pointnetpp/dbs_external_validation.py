#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DBS 整合算法 Pipeline (TRIPOD-AI Compliant Version)
====================================================
功能: 
1. [模拟/调用] Lead-DBS (MATLAB) 进行电极定位与电场仿真
2. [数据工程] 将 MATLAB 结果转换为 3D 点云，并标记中心来源 (Center ID)
3. [核心算法] 使用 PointNet++ 进行疗效预测 (已修复通道数 Bug)
4. [TRIPOD-AI] 执行严格的临床验证：
   - 严防数据泄漏: Leave-Center-Out Cross Validation
   - 模型校准: Calibration Curve & Brier Score
   - 临床决策: Decision Curve Analysis (DCA)
   - 可复现性: 随机种子锁定与环境记录

引用标准: TRIPOD+AI 2024 Reporting Guidelines
"""

import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# FIX: calibration_curve is in sklearn.calibration
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import time
import sys
import random
import platform

# 尝试导入 MATLAB 引擎
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

# ==========================================
# 1. 全局配置与可复现性设置 (TRIPOD-AI Item: Reproducibility)
# ==========================================
CONFIG = {
    'raw_data_dir': './raw_dicom_data',
    'leaddbs_out_dir': './leaddbs_output',
    'pointcloud_dir': './processed_pcd',
    
    # 模拟多中心数据 (模拟 TRIPOD-AI 要求的外部验证)
    'centers': ['Center_A', 'Center_B', 'Center_C'], # A,B 用于训练，C 用于外部验证
    'samples_per_center': 4, # 演示用少量样本
    
    'points_per_sample': 2048,
    'batch_size': 4,
    'epochs': 5,
    'lr': 0.001,
    'seed': 42, # 锁定随机种子
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def set_reproducibility(seed):
    """锁定所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"🔒 [Reproducibility] Random Seed set to {seed}")
    print(f"ℹ️  [System] Python {platform.python_version()} | PyTorch {torch.__version__} | Device: {CONFIG['device']}")

# ==========================================
# 2. Lead-DBS 接口层 (MATLAB Integration)
# ==========================================
class LeadDBSAdapter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def run_preprocessing(self, patient_id, center_id):
        """调用 Lead-DBS 流程"""
        # print(f"   [Lead-DBS] Processing {patient_id} from {center_id}...") # 简化输出
        if not MATLAB_AVAILABLE:
            # === 仿真模式 ===
            # 生成模拟结果
            self._generate_mock_mat_result(patient_id, center_id)

    def _generate_mock_mat_result(self, patient_id, center_id):
        """生成包含中心信息的模拟数据"""
        mat_path = os.path.join(self.output_dir, f'{patient_id}_{center_id}_result.npz')
        N = CONFIG['points_per_sample'] * 2
        coords = np.random.randn(N, 3).astype(np.float32) * 10
        e_field = np.random.rand(N, 1).astype(np.float32) * 500
        anatomy = np.random.randint(0, 5, (N, 1)).astype(np.float32)
        
        # 模拟不同中心的数据分布差异 (Data Shift Simulation)
        bias = 0
        if center_id == 'Center_C': # 外部中心稍有不同
            bias = 5 
            
        updrs_score = np.random.uniform(10 + bias, 80 + bias)
        
        np.savez(mat_path, coords=coords, e_field=e_field, anatomy=anatomy, 
                 updrs=updrs_score, center=center_id)

# ==========================================
# 3. 数据转换与加载 (TRIPOD-AI Item: Data Leakage Control)
# ==========================================
def convert_mat_to_pointcloud():
    """转换数据并保留 Center ID 以支持留一验证"""
    if os.path.exists(CONFIG['pointcloud_dir']):
        shutil.rmtree(CONFIG['pointcloud_dir'])
    os.makedirs(CONFIG['pointcloud_dir'])
    
    mat_files = glob.glob(os.path.join(CONFIG['leaddbs_out_dir'], "*.npz"))
    
    for f in tqdm(mat_files, desc="[Data Eng] ETL Processing"):
        data = np.load(f)
        filename = os.path.basename(f)
        patient_id = filename.split('_result')[0]
        center_id = str(data['center'])
        
        vec = data['coords'] / (np.linalg.norm(data['coords'], axis=1, keepdims=True) + 1e-6)
        features = np.concatenate([data['coords'], data['anatomy'], data['e_field'], vec], axis=1).astype(np.float32)
        
        y_reg = data['updrs']
        y_cls = 1 if y_reg > 40 else 0 # 响应者阈值
        
        # 保存时带上中心信息
        np.savez(os.path.join(CONFIG['pointcloud_dir'], f'{patient_id}.npz'),
                 points=features, y_reg=y_reg, y_cls=y_cls, center=center_id)

class DBSDataset(Dataset):
    def __init__(self, file_list, num_points=2048):
        self.files = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"].astype(np.float32)
        y_reg = torch.tensor(data["y_reg"], dtype=torch.float32)
        y_cls = torch.tensor(data["y_cls"], dtype=torch.float32)
        
        # 采样与归一化
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        xyz = points[:, :3]
        xyz -= np.mean(xyz, axis=0)
        xyz /= (np.max(np.sqrt(np.sum(xyz**2, axis=1))) + 1e-6)
        points[:, :3] = xyz
        
        return torch.from_numpy(points), y_reg, y_cls

def get_split_by_center(data_dir, val_center='Center_C'):
    """
    实现 Leave-Center-Out 数据划分
    Train: Center A, B
    External Validation: Center C
    """
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    train_files = []
    val_files = []
    
    for f in all_files:
        data = np.load(f)
        center = str(data['center'])
        if center == val_center:
            val_files.append(f)
        else:
            train_files.append(f)
            
    return train_files, val_files

# ==========================================
# 4. PointNet++ 网络 (已修复通道数问题)
# ==========================================
# ... (辅助函数 farthest_point_sample, index_points, query_ball_point 保持不变, 此处省略以节省篇幅) ...
def farthest_point_sample(xyz, npoint):
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
        xyz = xyz.permute(0, 2, 1)
        if points is not None: points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C).to(xyz.device)
            grouped_points = torch.cat([xyz, points], dim=2).view(B, -1, 1, N) if points is not None else xyz.view(B, -1, 1, N)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) 
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
            else:
                new_points = grouped_xyz_norm
            new_points = new_points.permute(0, 3, 2, 1)
            grouped_points = new_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class DBSPointNetPP(nn.Module):
    def __init__(self):
        super(DBSPointNetPP, self).__init__()
        # SA1 Input: 8 (Features) + 3 (Coords) = 11
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=259, mlp=[256, 512, 1024], group_all=True)
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
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz
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
# 5. TRIPOD-AI 评估模块 (Calibration & DCA)
# ==========================================
class TripodAIEvaluator:
    @staticmethod
    def assess_calibration(y_true, y_prob, n_bins=5):
        """
        计算 Brier Score 并打印校准表
        (Brier Score 越低越好，0为完美)
        """
        brier = brier_score_loss(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        print(f"\n   📈 [Calibration] Brier Score: {brier:.4f} (Ideal: 0.0)")
        print(f"   📊 [Calibration Table]")
        print(f"      Mean Predicted Prob  |  Fraction of Positives")
        print(f"      ---------------------------------------------")
        for p_pred, p_true in zip(prob_pred, prob_true):
            print(f"      {p_pred:.4f}               |  {p_true:.4f}")
            
    @staticmethod
    def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.1, 1.0, 0.1)):
        """
        计算 Net Benefit 用于决策曲线分析 (DCA)
        Net Benefit = (TPR * prevalence) - (FPR * (1 - prevalence) * (thresh / (1 - thresh)))
        """
        print(f"\n   ⚖️  [Decision Curve Analysis (DCA)]")
        print(f"      Threshold  |  Net Benefit (Model)  |  Net Benefit (Treat All)")
        print(f"      ------------------------------------------------------------")
        
        n = len(y_true)
        prevalence = np.sum(y_true) / n
        
        for thresh in thresholds:
            # Model Metrics
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            
            # Treat All Metrics
            tp_all = np.sum(y_true == 1)
            fp_all = np.sum(y_true == 0)
            net_benefit_all = (tp_all / n) - (fp_all / n) * (thresh / (1 - thresh))
            
            print(f"      {thresh:.2f}       |  {max(0, net_benefit):.4f}               |  {max(0, net_benefit_all):.4f}")

# ==========================================
# 6. 主程序 Loop (The Movie)
# ==========================================
def main():
    print(f"\n🎬 TRIPOD-AI Pipeline 演示启动...\n")
    time.sleep(1)
    
    # --- Step 0: Reproducibility ---
    print(f"======== Phase 0: Reproducibility Setup ========")
    set_reproducibility(CONFIG['seed'])
    time.sleep(1)

    # --- Step 1: Lead-DBS Preprocessing (Simulation) ---
    print(f"\n======== Phase 1: Multi-Center Simulation (Lead-DBS) ========")
    dbs_runner = LeadDBSAdapter(CONFIG['leaddbs_out_dir'])
    
    for center in CONFIG['centers']:
        print(f"🏥 Generating Data for {center}...")
        for i in range(CONFIG['samples_per_center']):
            pid = f'Pat_{center}_{i:02d}'
            dbs_runner.run_preprocessing(pid, center)
        time.sleep(0.5)

    # --- Step 2: Data Engineering ---
    print(f"\n======== Phase 2: Point Cloud Transformation ========")
    convert_mat_to_pointcloud()
    
    # --- Step 3: Train/Test Split (Leave-Center-Out) ---
    print(f"\n======== Phase 3: Study Design (Leave-Center-Out) ========")
    train_files, val_files = get_split_by_center(CONFIG['pointcloud_dir'], val_center='Center_C')
    print(f"📚 Internal Training Set (Center A+B): {len(train_files)} samples")
    print(f"🛡️  External Validation Set (Center C): {len(val_files)} samples")
    print(f"⚠️  Status: Strict Separation Enforced (No Data Leakage)")
    time.sleep(1)
    
    # --- Step 4: AI Training ---
    print(f"\n======== Phase 4: Model Training (PointNet++) ========")
    train_ds = DBSDataset(train_files, num_points=CONFIG['points_per_sample'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = DBSPointNetPP().to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion_reg = nn.L1Loss()
    criterion_cls = nn.BCELoss()
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for points, y_reg, y_cls in pbar:
            points = points.transpose(2, 1).to(CONFIG['device'])
            y_reg, y_cls = y_reg.unsqueeze(1).to(CONFIG['device']), y_cls.unsqueeze(1).to(CONFIG['device'])
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(points)
            loss = criterion_reg(pred_reg, y_reg) + 0.5 * criterion_cls(pred_cls, y_cls)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        time.sleep(0.2)
            
    # --- Step 5: External Validation (TRIPOD-AI Core) ---
    print(f"\n======== Phase 5: External Validation (TRIPOD-AI Report) ========")
    val_ds = DBSDataset(val_files, num_points=CONFIG['points_per_sample'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model.eval()
    y_true_cls, y_prob_cls = [], []
    
    print("🔍 Running Inference on External Center C...")
    with torch.no_grad():
        for points, y_reg, y_cls in val_loader:
            points = points.transpose(2, 1).to(CONFIG['device'])
            _, pred_cls = model(points)
            y_true_cls.extend(y_cls.cpu().numpy())
            y_prob_cls.extend(pred_cls.cpu().numpy())
    
    y_true_cls = np.array(y_true_cls)
    y_prob_cls = np.array(y_prob_cls)
    
    # Metrics
    auc = roc_auc_score(y_true_cls, y_prob_cls)
    print(f"\n📝 [Discrimination] External AUC: {auc:.4f}")
    
    # Calibration & DCA
    TripodAIEvaluator.assess_calibration(y_true_cls, y_prob_cls)
    time.sleep(1)
    TripodAIEvaluator.decision_curve_analysis(y_true_cls, y_prob_cls)
    
    print(f"\n✅ Pipeline Completed Successfully.")

if __name__ == '__main__':
    main()