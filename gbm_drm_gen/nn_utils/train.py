import os, time, csv
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .model import DRMNet

# Simple CSV logger
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()
    def log(self, row):
        self._writer.writerow(row)
        self._file.flush()
    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

def laplacian_smoothness(yhat_flat, n_out, n_in, weight=0.0):
    if weight <= 0:
        # return a zero tensor on the same device/dtype
        return yhat_flat.sum() * 0.0
    B, L = yhat_flat.shape
    y2d = yhat_flat.view(B, n_out, n_in)
    dy = y2d[:, 1:, :] - y2d[:, :-1, :]
    dx = y2d[:, :, 1:] - y2d[:, :, :-1]
    return weight * (dx.pow(2).mean() + dy.pow(2).mean())

def train_one_epoch(model, loader, opt, device, n_out, n_in, lam_smooth=0.0, progress=True):
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    n = 0

    it = loader
    if progress:
        try:
            it = tqdm(loader, desc="train", total=len(loader), leave=False)
        except TypeError:
            it = tqdm(loader, desc="train", leave=False)

    for batch in it:
        angles = batch["angles"].to(device)
        normals = batch["normals"].to(device)
        y = batch["y"].to(device)               # log1p target
        det_id = batch.get("det_id", None)
        if det_id is not None:
            det_id = det_id.to(device)

        opt.zero_grad()
        yhat = model(angles, normals, det_id)
        loss = mse(yhat, y) + laplacian_smoothness(yhat, n_out, n_in, lam_smooth)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = angles.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)

@torch.no_grad()
def eval_one_epoch(model, loader, device, n_out, n_in, lam_smooth=0.0, progress=True):
    model.eval()
    mse = nn.MSELoss()
    total = 0.0
    n = 0

    it = loader
    if progress:
        # len(loader) is available for standard DataLoader; if not, tqdm still works without total
        try:
            it = tqdm(loader, desc="val", total=len(loader), leave=False)
        except TypeError:
            it = tqdm(loader, desc="val", leave=False)

    for batch in it:
        angles = batch["angles"].to(device)
        normals = batch["normals"].to(device)
        y = batch["y"].to(device)  # log1p target
        det_id = batch.get("det_id", None)
        if det_id is not None:
            det_id = det_id.to(device)

        yhat = model(angles, normals, det_id)
        loss = mse(yhat, y) + laplacian_smoothness(yhat, n_out, n_in, lam_smooth)

        bs = angles.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


@torch.no_grad()
def eval_metrics_linear(model, loader, device, progress=True):
    import torch.nn.functional as F
    model.eval()
    n_samples = 0
    rmse_sum = 0.0
    rel_mae_list = []
    cos_list = []

    it = loader
    if progress:
        try:
            it = tqdm(loader, desc="val-metrics", total=len(loader), leave=False)
        except TypeError:
            it = tqdm(loader, desc="val-metrics", leave=False)

    for batch in it:
        angles = batch["angles"].to(device)
        normals = batch["normals"].to(device)
        y_log = batch["y"].to(device)  # log1p target
        det_id = batch.get("det_id", None)
        if det_id is not None:
            det_id = det_id.to(device)

        yhat_log = model(angles, normals, det_id)
        y = torch.expm1(y_log).clamp_min_(0)
        yhat = torch.expm1(yhat_log).clamp_min_(0)

        # RMSE per-sample, average over samples
        rmse = torch.sqrt(torch.mean((yhat - y) ** 2, dim=1))  # [B]
        rmse_sum += float(rmse.sum().item())

        # Relative MAE per-sample: median over entries (robust)
        denom = (y.abs() + 1e-12)
        rel_mae = torch.median((yhat - y).abs() / denom, dim=1).values  # [B]
        rel_mae_list.append(rel_mae.cpu().numpy())

        # Cosine similarity per-sample
        y_n = F.normalize(y.view(y.size(0), -1), dim=1)
        yhat_n = F.normalize(yhat.view(yhat.size(0), -1), dim=1)
        cos = torch.sum(y_n * yhat_n, dim=1)
        cos_list.append(cos.cpu().numpy())

        n_samples += angles.size(0)

    rel_mae_arr = np.concatenate(rel_mae_list) if rel_mae_list else np.array([np.nan])
    cos_arr = np.concatenate(cos_list) if cos_list else np.array([np.nan])

    return {
        "rmse_mean": rmse_sum / max(1, n_samples),
        "rel_mae_median": float(np.nanmedian(rel_mae_arr)),
        "cos_sim_mean": float(np.nanmean(cos_arr)),
        "n_samples": int(n_samples),
    }

def fit_model(h5_path, out_len, n_out, n_in, save_path,
              use_embedding=False, num_det=12,
              hidden=(128, 256, 16), emb_dim=8, low_rank_k=None,
              batch_size=128, lr=1e-3, epochs=30, val_frac=0.1,
              lam_smooth=0.0, num_workers=0, device=None, seed=0,
              subset_idx=None, subset_by_name=None,
              log_dir=None, log_csv=None, patience=None, tb=False):
    from .datasets import H5DRMDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    full_ds = H5DRMDataset(h5_path, use_log_target=True,
                       subset_idx=subset_idx, subset_by_name=subset_by_name,
                       preload=True)

    N = len(full_ds)
    N_val = max(1, int(val_frac * N))
    N_trn = N - N_val
    train_ds, val_ds = random_split(full_ds, [N_trn, N_val],
                                    generator=torch.Generator().manual_seed(seed))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = DRMNet(out_len=out_len, use_embedding=use_embedding, num_det=num_det,
                   emb_dim=emb_dim, hidden=hidden, use_log_target=True,
                   low_rank_k=low_rank_k).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Logging
    csv_logger = CSVLogger(log_csv, ["epoch","train_mse_log1p","val_mse_log1p","lr",
                                     "val_rmse","val_rel_mae_median","val_cos_sim"]) if log_csv else None
    tb_writer = None
    if tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir or "runs/drmnet")
        except Exception:
            tb_writer = None

    best_val = math.inf
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        trn = train_one_epoch(model, train_loader, opt, device, n_out, n_in, lam_smooth, progress=True)
        val = eval_one_epoch(model, val_loader, device, n_out, n_in, lam_smooth, progress=True)
        extra = eval_metrics_linear(model, val_loader, device, progress=True)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        dt = time.time() - t0

        print(f"Epoch {ep:03d} | train MSE(log1p): {trn:.6f} | val MSE(log1p): {val:.6f} | "
              f"val RMSE: {extra['rmse_mean']:.4f} | val relMAE_med: {extra['rel_mae_median']:.4f} | "
              f"val cos: {extra['cos_sim_mean']:.4f} | lr={lr_now:.2e} | {dt:.1f}s")

        if csv_logger:
            csv_logger.log({
                "epoch": ep,
                "train_mse_log1p": trn,
                "val_mse_log1p": val,
                "lr": lr_now,
                "val_rmse": extra["rmse_mean"],
                "val_rel_mae_median": extra["rel_mae_median"],
                "val_cos_sim": extra["cos_sim_mean"],
            })
        if tb_writer:
            tb_writer.add_scalar("loss/train_mse_log1p", trn, ep)
            tb_writer.add_scalar("loss/val_mse_log1p", val, ep)
            tb_writer.add_scalar("val/rmse_linear", extra["rmse_mean"], ep)
            tb_writer.add_scalar("val/rel_mae_median", extra["rel_mae_median"], ep)
            tb_writer.add_scalar("val/cos_sim", extra["cos_sim_mean"], ep)
            tb_writer.add_scalar("lr", lr_now, ep)

        improved = val < best_val
        if improved:
            best_val = val
            torch.save({
                "model": model.state_dict(),
                "cfg": {
                    "out_len": out_len, "n_out": n_out, "n_in": n_in,
                    "use_embedding": use_embedding, "num_det": num_det,
                    "hidden": tuple(hidden), "emb_dim": int(emb_dim),
                    # Optional extras:
                    # "use_log_target": True,
                    # "low_rank_k": int(low_rank_k) if low_rank_k is not None else 0,
                },
            }, save_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if patience is not None and bad_epochs >= patience:
            print(f"Early stopping at epoch {ep} (no val improvement for {patience} epochs)")
            break

    if csv_logger: csv_logger.close()
    if tb_writer: 
        try: tb_writer.close()
        except Exception: pass

    print(f"Saved best to {save_path} (val={best_val:.6f})")
    return save_path