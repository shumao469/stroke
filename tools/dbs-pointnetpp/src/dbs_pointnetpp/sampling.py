from __future__ import annotations
import torch

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling (FPS).

    Parameters
    ----------
    xyz : (B, N, 3) float tensor
    npoint : int
        number of points to sample

    Returns
    -------
    centroids : (B, npoint) long tensor
        indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices.

    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(idx.shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def knn_group(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """KNN grouping indices using squared Euclidean distance.

    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    returns idx: (B, S, k)
    """
    dist = torch.cdist(new_xyz, xyz, p=2) ** 2  # (B, S, N)
    idx = dist.topk(k, dim=-1, largest=False)[1]  # (B, S, k)
    return idx
