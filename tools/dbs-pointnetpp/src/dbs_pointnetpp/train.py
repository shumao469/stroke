from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error, mean_squared_error

def train_one_epoch(model, loader, optimizer, device, lambda_reg: float = 0.5):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for points, y_reg, y_cls in loader:
        points = points.to(device)
        y_reg = y_reg.to(device)
        y_cls_f = y_cls.to(device).float()

        optimizer.zero_grad()
        pred_reg, pred_logits = model(points)

        loss = 0.0
        if pred_logits is not None and torch.any(y_cls >= 0):
            cls_loss = bce(pred_logits, y_cls_f)
            loss = loss + cls_loss
            total_cls_loss += float(cls_loss.item()) * points.size(0)

        if pred_reg is not None:
            reg_loss = l1(pred_reg, y_reg)
            loss = loss + lambda_reg * reg_loss
            total_reg_loss += float(reg_loss.item()) * points.size(0)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * points.size(0)
        n += points.size(0)

    return {
        "loss": total_loss / max(n, 1),
        "reg_l1": total_reg_loss / max(n, 1),
        "cls_bce": total_cls_loss / max(n, 1),
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_cls, y_prob_cls = [], []
    y_true_reg, y_pred_reg = [], []
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    n = 0

    for points, y_reg, y_cls in loader:
        points = points.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        pred_reg, pred_logits = model(points)

        if pred_reg is not None:
            reg_loss_sum += float(l1(pred_reg, y_reg).item()) * points.size(0)
            y_true_reg.append(y_reg.detach().cpu().numpy())
            y_pred_reg.append(pred_reg.detach().cpu().numpy())

        if pred_logits is not None and torch.any(y_cls >= 0):
            y_cls_f = y_cls.float()
            cls_loss_sum += float(bce(pred_logits, y_cls_f).item()) * points.size(0)
            y_true_cls.append(y_cls.detach().cpu().numpy())
            y_prob_cls.append(torch.sigmoid(pred_logits).detach().cpu().numpy())

        n += points.size(0)

    metrics = {
        "reg_l1": reg_loss_sum / max(n, 1),
        "cls_bce": cls_loss_sum / max(n, 1),
    }

    if len(y_true_reg) > 0:
        yt = np.concatenate(y_true_reg, axis=0)
        yp = np.concatenate(y_pred_reg, axis=0)
        metrics.update({
            "r2": float(r2_score(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        })

    if len(y_true_cls) > 0:
        yt = np.concatenate(y_true_cls, axis=0)
        yp = np.concatenate(y_prob_cls, axis=0)
        try:
            metrics["auc_roc"] = float(roc_auc_score(yt, yp))
            metrics["auc_pr"] = float(average_precision_score(yt, yp))
        except Exception:
            metrics["auc_roc"] = float("nan")
            metrics["auc_pr"] = float("nan")

    return metrics
