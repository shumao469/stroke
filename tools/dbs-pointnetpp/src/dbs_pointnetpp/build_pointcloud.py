from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from sklearn.neighbors import KDTree
from scipy.interpolate import Rbf

def rbf_interpolate_field(
    sim_nodes_xyz: np.ndarray,
    sim_nodes_E: np.ndarray,
    query_xyz: np.ndarray,
    function: str = "multiquadric",
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """RBF interpolation for mapping FEM electric-field values onto dense anatomy points.

    This mirrors the paper's concept: interpolate FEM node values to query points.

    Parameters
    ----------
    sim_nodes_xyz : (M,3)
    sim_nodes_E : (M,) or (M,1)
    query_xyz : (N,3)
    function : RBF kernel, e.g., 'multiquadric', 'gaussian', 'linear'
    epsilon : optional shape parameter

    Returns
    -------
    E_query : (N,)
    """
    sim_nodes_E = np.asarray(sim_nodes_E).reshape(-1)
    rbf = Rbf(sim_nodes_xyz[:,0], sim_nodes_xyz[:,1], sim_nodes_xyz[:,2], sim_nodes_E,
              function=function, epsilon=epsilon)
    return rbf(query_xyz[:,0], query_xyz[:,1], query_xyz[:,2]).astype(np.float32)

def fuse_pointcloud_kdtree(
    anatomy_xyz: np.ndarray,
    electrode_xyz: np.ndarray,
    efield_xyz: np.ndarray,
    efield_val: np.ndarray,
    k: int = 32,
) -> np.ndarray:
    """Fuse anatomy surface points, electrode surface points, and E-field nodes into a unified point cloud.

    Strategy (paper-aligned, simplified):
      1) stack geometry (anatomy + electrode) as query points
      2) build KDTree on efield_xyz and attach local neighborhood E statistics (mean/std/max)
      3) output points (N, 3 + F), where F includes E_mean/E_std/E_max

    Notes:
      - The paper reports KD-tree nearest-neighbor fusion (k=32) + FPS.
      - Here we implement a practical feature attachment step; downstream FPS is done during training.

    Returns
    -------
    points : (N, 6) float32
        [x,y,z, E_mean, E_std, E_max]
    """
    anatomy_xyz = np.asarray(anatomy_xyz, dtype=np.float32)
    electrode_xyz = np.asarray(electrode_xyz, dtype=np.float32)
    efield_xyz = np.asarray(efield_xyz, dtype=np.float32)
    efield_val = np.asarray(efield_val, dtype=np.float32).reshape(-1)

    query = np.vstack([anatomy_xyz, electrode_xyz]).astype(np.float32)

    tree = KDTree(efield_xyz)
    d, ind = tree.query(query, k=min(k, len(efield_xyz)))
    neigh_E = efield_val[ind]  # (N, k)

    E_mean = neigh_E.mean(axis=1)
    E_std = neigh_E.std(axis=1)
    E_max = neigh_E.max(axis=1)

    feats = np.stack([E_mean, E_std, E_max], axis=1).astype(np.float32)
    points = np.hstack([query, feats]).astype(np.float32)
    return points
