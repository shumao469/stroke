#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DBS 整合算法完整 Pipeline (TRIPOD-AI Compliant & Bilingual)
====================================================================
Title: PointNet++ based DBS Efficacy Prediction Framework
Version: 2.0 (Integrated)

此脚本整合了以下功能模块：
1. [Simulation] Lead-DBS 电极定位与电场仿真接口 (Mock/Real)
2. [Data Eng]   3D 点云数据生成与多中心标记 (Center Labeling)
3. [Model]      修复版 PointNet++ 网络 (修正了通道数不匹配问题)
4. [Validation] TRIPOD-AI 临床验证标准：
   - 严防数据泄漏 (Leave-Center-Out Split)
   - 模型校准 (Calibration Curve & Brier Score)
   - 决策曲线分析 (Decision Curve Analysis, DCA)

输出格式: 中英文双语日志，适配 SciLim 等科研展示工具。
Output: Bilingual logs (Chinese/English) suitable for scientific display.
"""

import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# 0. 全局配置与工具 (Global Config & Utils)
# ==========================================
CONFIG = {
    'raw_data_dir': './raw_dicom_data',
    'leaddbs_out_dir': './leaddbs_output',
    'pointcloud_dir': './processed_pcd',
    
    # 模拟多中心数据 (Simulate Multi-center Data)
    'centers': ['Center_A', 'Center_B', 'Center_C'], # A,B -> Train, C -> External Val
    'samples_per_center': 5,
    
    'points_per_sample': 2048,
    'batch_size': 4,
    'epochs': 5,
    'lr': 0.001,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class BilingualLogger:
    """双语日志助手 / Bilingual Log Helper"""
    @staticmethod
    def info(cn, en):
        print(f"[INFO] {cn} | {en}")

    @staticmethod
    def section(cn, en):
        print(f"\n{'='*20}\n[PHASE] {cn}\n        {en}\n{'='*20}")

    @staticmethod
    def metric(name, value, note=""):
        print(f"   📊 {name}: {value} {note}")

def set_reproducibility(seed):
    """锁定随机种子 / Lock random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    BilingualLogger.info(f"随机种子已锁定: {seed}", f"Random Seed set to {seed}")
    BilingualLogger.info(f"运行环境: Python {platform.python_version()} | PyTorch {torch.__version__}", 
                         f"Env: Python {platform.python_version()} | PyTorch {torch.__version__}")

# ==========================================
# 1. Lead-DBS 接口层 (Lead-DBS Adapter)
# ==========================================
class LeadDBSAdapter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def run_preprocessing(self, patient_id, center_id):
        """调用或模拟 Lead-DBS 处理流程"""
        if not MATLAB_AVAILABLE:
            self._generate_mock_mat_result(patient_id, center_id)

    def _generate_mock_mat_result(self, patient_id, center_id):
        """
        生成模拟的 .mat/.npz 数据
        Generate mock clinical data with center-specific distribution shifts
        """
        mat_path = os.path.join(self.output_dir, f'{patient_id}_{center_id}_result.npz')
        N = CONFIG['points_per_sample'] * 2
        
        # 1. 模拟坐标与解剖结构
        coords = np.random.randn(N, 3).astype(np.float32) * 10
        anatomy = np.random.randint(0, 5, (N, 1)).astype(np.float32)
        
        # 2. 模拟电场 (E-Field)
        e_field = np.random.rand(N, 1).astype(np.float32) * 500
        
        # 3. 模拟不同中心的数据偏移 (Data Shift for Center C)
        bias = 0
        if center_id == 'Center_C': 
            bias = 10  # 外部中心评分偏高
            
        updrs_score = np.clip(np.random.normal(40 + bias, 15), 0, 100)
        
        np.savez(mat_path, coords=coords, e_field=e_field, anatomy=anatomy, 
                 updrs=updrs_score, center=center_id)

# ==========================================
# 2. 数据转换与加载 (Data Engineering)
# ==========================================
def convert_mat_to_pointcloud():
    """
    ETL流程：从 Lead-DBS 输出转换为 PointNet++ 输入
    ETL: Lead-DBS Output -> PointNet++ Input (.npz)
    """
    if os.path.exists(CONFIG['pointcloud_dir']):
        shutil.rmtree(CONFIG['pointcloud_dir'])
    os.makedirs(CONFIG['pointcloud_dir'])
    
    mat_files = glob.glob(os.path.join(CONFIG['leaddbs_out_dir'], "*.npz"))
    
    for f in tqdm(mat_files, desc="[ETL] Processing"):
        data = np.load(f)
        filename = os.path.basename(f)
        patient_id = filename.split('_result')[0]
        center_id = str(data['center'])
        
        # 特征工程: 坐标 + 解剖ID + 电场强度 + 归一化向量
        # Feature: [x, y, z, anatomy, e_field, vec_x, vec_y, vec_z]
        vec = data['coords'] / (np.linalg.norm(data['coords'], axis=1, keepdims=True) + 1e-6)
        features = np.concatenate([data['coords'], data['anatomy'], data['e_field'], vec], axis=1).astype(np.float32)
        
        y_reg = data['updrs']
        y_cls = 1 if y_reg > 50 else 0 # 设定改善 > 50% 为 Responder
        
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
        
        # 采样 (Sampling)
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        
        # 归一化 (Normalization)
        xyz = points[:, :3]
        xyz -= np.mean(xyz, axis=0)
        xyz /= (np.max(np.sqrt(np.sum(xyz**2, axis=1))) + 1e-6)
        points[:, :3] = xyz
        
        return torch.from_numpy(points), y_reg, y_cls

def get_split_by_center(data_dir, val_center='Center_C'):
    """
    中心留一法划分 (Leave-Center-Out Split)
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
# 3. PointNet++ 网络模型 (Model Architecture)
# ==========================================
def farthest_point_sample(xyz, npoint):
    """最远点采样 FPS"""
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
    """点索引 Indexing"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球邻域查询 Ball Query"""
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
                # FIX: 通道数拼接修正
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
        # Input Channel Calculation:
        # Original features (8) = 3(coords) + 1(anatomy) + 1(E) + 3(vec)
        # SA1 Input = 8 (Features) + 3 (Local Coords from SA) = 11
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=259, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.reg_head = nn.Linear(256, 1) # 回归: UPDRS
        self.cls_head = nn.Linear(256, 1) # 分类: Responder

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
# 4. TRIPOD-AI 评估模块 (Validation Metrics)
# ==========================================
class TripodAIEvaluator:
    @staticmethod
    def assess_calibration(y_true, y_prob, n_bins=5):
        """模型校准评估"""
        brier = brier_score_loss(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        BilingualLogger.info(f"Brier Score (校准度): {brier:.4f} (理想值 0.0)", 
                             f"Brier Score: {brier:.4f} (Ideal: 0.0)")
        
        print(f"   📊 [Calibration Table]")
        print(f"      Mean Predicted Prob  |  Fraction of Positives")
        print(f"      ---------------------------------------------")
        for p_pred, p_true in zip(prob_pred, prob_true):
            print(f"      {p_pred:.4f}               |  {p_true:.4f}")
            
    @staticmethod
    def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.1, 1.0, 0.1)):
        """临床决策曲线分析 (DCA)"""
        print(f"\n   ⚖️  [Decision Curve Analysis (DCA)]")
        print(f"      Threshold  |  Net Benefit (Model)  |  Net Benefit (Treat All)")
        print(f"      ------------------------------------------------------------")
        
        n = len(y_true)
        # prevalence = np.sum(y_true) / n # 实际患病率
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            # Net Benefit Formula
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            
            tp_all = np.sum(y_true == 1)
            fp_all = np.sum(y_true == 0)
            net_benefit_all = (tp_all / n) - (fp_all / n) * (thresh / (1 - thresh))
            
            print(f"      {thresh:.2f}       |  {max(0, net_benefit):.4f}               |  {max(0, net_benefit_all):.4f}")

# ==========================================
# 5. 主程序 (Main Pipeline)
# ==========================================
def main():
    BilingualLogger.section("初始化与环境配置", "Initialization & Setup")
    set_reproducibility(CONFIG['seed'])

    # --- Phase 1: Lead-DBS Simulation ---
    BilingualLogger.section("Lead-DBS 多中心仿真", "Multi-Center Lead-DBS Simulation")
    dbs_runner = LeadDBSAdapter(CONFIG['leaddbs_out_dir'])
    for center in CONFIG['centers']:
        print(f"   🏥 Generating Data for {center}...")
        for i in range(CONFIG['samples_per_center']):
            pid = f'Pat_{center}_{i:02d}'
            dbs_runner.run_preprocessing(pid, center)
    
    # --- Phase 2: Data Engineering ---
    BilingualLogger.section("点云数据转换与融合", "Point Cloud Transformation & Fusion")
    convert_mat_to_pointcloud()

    # --- Phase 3: Study Design ---
    BilingualLogger.section("实验设计: 中心留一验证", "Study Design: Leave-Center-Out Split")
    train_files, val_files = get_split_by_center(CONFIG['pointcloud_dir'], val_center='Center_C')
    BilingualLogger.info(f"内部训练集 (Center A+B): {len(train_files)} 例", 
                         f"Internal Training Set: {len(train_files)} samples")
    BilingualLogger.info(f"外部验证集 (Center C): {len(val_files)} 例", 
                         f"External Validation Set: {len(val_files)} samples")
    
    # --- Phase 4: AI Training ---
    BilingualLogger.section("模型训练 (PointNet++)", "AI Model Training")
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
    
    # --- Phase 5: External Validation ---
    BilingualLogger.section("外部验证与 TRIPOD-AI 报告", "External Validation & TRIPOD-AI Report")
    val_ds = DBSDataset(val_files, num_points=CONFIG['points_per_sample'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model.eval()
    y_true_cls, y_prob_cls = [], []
    y_true_reg, y_pred_reg = [], []
    
    with torch.no_grad():
        for points, y_reg, y_cls in val_loader:
            points = points.transpose(2, 1).to(CONFIG['device'])
            pred_reg, pred_cls = model(points)
            
            y_true_cls.extend(y_cls.cpu().numpy())
            y_prob_cls.extend(pred_cls.cpu().numpy())
            y_true_reg.extend(y_reg.cpu().numpy())
            y_pred_reg.extend(pred_reg.cpu().numpy())
            
    y_true_cls = np.array(y_true_cls)
    y_prob_cls = np.array(y_prob_cls)
    
    # 1. Discrimination (AUC)
    if len(np.unique(y_true_cls)) > 1:
        auc = roc_auc_score(y_true_cls, y_prob_cls)
        BilingualLogger.metric("外部 AUC (External AUC)", f"{auc:.4f}")
    else:
        print("   ⚠️  [Warning] Only one class present in validation set, skipping AUC.")

    # 2. Calibration
    TripodAIEvaluator.assess_calibration(y_true_cls, y_prob_cls)
    
    # 3. Decision Curve Analysis
    TripodAIEvaluator.decision_curve_analysis(y_true_cls, y_prob_cls)
    
    BilingualLogger.info("流程执行完毕。", "Pipeline Completed Successfully.")

if __name__ == '__main__':
    main()