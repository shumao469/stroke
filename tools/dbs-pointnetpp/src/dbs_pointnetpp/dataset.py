from __future__ import annotations
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DBSPointCloudDataset(Dataset):
    """Dataset reading per-patient point clouds stored as .npz.

    Expected keys in each .npz:
      - points: (N, C) float32
          First 3 columns are xyz coordinates. Remaining columns are point-wise attributes
          (e.g., E-field magnitude, gradients, directional features, etc.).
      - y_reg: scalar float
          regression target (e.g., MDS-UPDRS III improvement / change)
      - y_cls: scalar int (0/1)
          responder classification label (optional)

    Notes
    -----
    The loader performs random sampling to a fixed number of points per patient.
    Basic z-score normalization is applied separately to xyz and feature channels (optional).
    """

    def __init__(self, data_root: str, num_points: int = 20000, normalize: bool = True):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found under {data_root}")
        self.num_points = int(num_points)
        self.normalize = bool(normalize)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        f = self.files[idx]
        data = np.load(f)

        points = data["points"].astype(np.float32)  # (N, C)
        y_reg = np.float32(data["y_reg"]) if "y_reg" in data else np.float32(0.0)
        y_cls = np.int64(data["y_cls"]) if "y_cls" in data else np.int64(-1)

        N = points.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            extra = np.random.choice(N, self.num_points - N, replace=True)
            choice = np.concatenate([np.arange(N), extra], axis=0)
        points = points[choice]

        if self.normalize:
            xyz = points[:, :3]
            xyz_mean = xyz.mean(axis=0, keepdims=True)
            xyz_std = xyz.std(axis=0, keepdims=True) + 1e-6
            points[:, :3] = (xyz - xyz_mean) / xyz_std

            if points.shape[1] > 3:
                feat = points[:, 3:]
                feat_mean = feat.mean(axis=0, keepdims=True)
                feat_std = feat.std(axis=0, keepdims=True) + 1e-6
                points[:, 3:] = (feat - feat_mean) / feat_std

        points = torch.from_numpy(points)  # (N, C)
        y_reg = torch.tensor(y_reg, dtype=torch.float32)
        y_cls = torch.tensor(y_cls, dtype=torch.long)
        return points, y_reg, y_cls
