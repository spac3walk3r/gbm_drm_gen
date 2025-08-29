import os, time, csv
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .model import DRMNet

# -----------------------
# Simple CSV logger
# -----------------------
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

# -----------------------
# Regularizer (optional)
# -----------------------
def laplacian_smoothness(yhat_flat, n_out, n_in, weight=0.0):
    if weight <= 0:
        return yhat_flat.sum() * 0.0
    B, L = yhat_flat.shape
    y2d = yhat_flat.view(B, n_out, n_in)
    dy = y2d[:, 1:, :] - y2d[:, :-1, :]
    dx = y2d[:, :, 1:] - y2d[:, :, :-1]
    return weight * (dx.pow(2).mean() + dy.pow(2).mean())

# -----------------------
# Transforms and metrics
# -----------------------
DEFAULT_LOG10_EPS = 1e-12

def invert_transform(y_trans: torch.Tensor, mode: str = "log1p", eps: float = DEFAULT_LOG10_EPS):
    if mode == "log10":
        return torch.pow(10.0, y_trans) - eps
    elif mode == "log1p":
        return torch.expm1(y_trans)
    else:
        return y_trans

def rel_err_stats(yhat_lin: torch.Tensor, ytrue_lin: torch.Tensor, thresh: float = 1e-10):
    mask = (ytrue_lin > thresh)
    if not torch.any(mask):
        return 0.0, 0.0, 0.0, 0.0
    rel = torch.abs(yhat_lin[mask] - ytrue_lin[mask]) / torch.clamp(ytrue_lin[mask], min=thresh)
    rel_np = rel.detach().cpu().numpy()
    return float(rel_np.mean()), float(np.median(rel_np)), float(np.quantile(rel_np, 0.9)), float(np.quantile(rel_np, 0.99))

# -----------------------
# Train / eval one epoch
# -----------------------
def train_one_epoch(model, loader, opt, device, n_out, n_in, lam_smooth=0.0, progress=True):
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    n = 0

    it = tqdm(loader, desc="train", total=len(loader), leave=False) if progress else loader

    for batch in it:
        # batch contains: x (9 features) and/or angles/normals/phys, and y (transformed), optional det_id
        y = batch["y"].to(device)
        # move all tensor-like batch entries to device for safety
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        opt.zero_grad()
        yhat = model(batch=batch)
        loss = mse(yhat, y) + laplacian_smoothness(yhat, n_out, n_in, lam_smooth)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)

@torch.no_grad()
def eval_one_epoch(model, loader, device, n_out, n_in, lam_smooth=0.0, progress=True):
    model.eval()
    mse = nn.MSELoss()
    total = 0.0
    n = 0

    it = tqdm(loader, desc="val", total=len(loader), leave=False) if progress else loader

    for batch in it:
        y = batch["y"].to(device)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        yhat = model(batch=batch)
        loss = mse(yhat, y) + laplacian_smoothness(yhat, n_out, n_in, lam_smooth)

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)

@torch.no_grad()
def eval_metrics_linear(model, loader, device, target_transform="log1p", log10_eps=DEFAULT_LOG10_EPS, progress=True):
    import torch.nn.functional as F
    model.eval()
    n_samples = 0
    rmse_sum = 0.0
    rel_mean_sum = 0.0
    rel_med_list = []
    rel_p90_list = []
    rel_p99 = 0.0
    cos_list = []

    it = tqdm(loader, desc="val-metrics", total=len(loader), leave=False) if progress else loader

    for batch in it:
        y_trans = batch["y"].to(device)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        yhat_trans = model(batch=batch)

        # invert transform to linear scale
        y_lin = invert_transform(y_trans, mode=target_transform, eps=log10_eps).clamp_min_(0)
        yhat_lin = invert_transform(yhat_trans, mode=target_transform, eps=log10_eps).clamp_min_(0)

        # RMSE per-sample, then averaged
        rmse = torch.sqrt(torch.mean((yhat_lin - y_lin) ** 2, dim=1))  # [B]
        rmse_sum += float(rmse.sum().item())

        # Relative error stats (mean/median/p90/p99)
        m, med, p90, p99 = rel_err_stats(yhat_lin, y_lin)
        rel_mean_sum += m * y_lin.size(0)
        rel_med_list.append(np.array([med] * y_lin.size(0)))
        rel_p90_list.append(np.array([p90] * y_lin.size(0)))
        rel_p99 = max(rel_p99, p99)

        # Cosine similarity (normalized vectors)
        y_n = F.normalize(y_lin, dim=1)
        yhat_n = F.normalize(yhat_lin, dim=1)
        cos = torch.sum(y_n * yhat_n, dim=1)
        cos_list.append(cos.detach().cpu().numpy())

        n_samples += y_lin.size(0)

    rel_med_arr = np.concatenate(rel_med_list) if rel_med_list else np.array([np.nan])
    rel_p90_arr = np.concatenate(rel_p90_list) if rel_p90_list else np.array([np.nan])
    cos_arr = np.concatenate(cos_list) if cos_list else np.array([np.nan])

    return {
        "rmse_mean": rmse_sum / max(1, n_samples),
        "rel_mean": rel_mean_sum / max(1, n_samples),
        "rel_median": float(np.nanmedian(rel_med_arr)),
        "rel_p90": float(np.nanmedian(rel_p90_arr)),
        "rel_p99": float(rel_p99),
        "cos_sim_mean": float(np.nanmean(cos_arr)),
        "n_samples": int(n_samples),
    }

# -----------------------
# Main fit function
# -----------------------
def fit_model(h5_path, out_len, n_out, n_in, save_path,
              use_embedding=False, num_det=12,
              hidden=(128, 256, 16), emb_dim=8, low_rank_k=None,
              batch_size=128, lr=1e-3, epochs=30, val_frac=0.1,
              lam_smooth=0.0, num_workers=0, device=None, seed=0,
              subset_idx=None, subset_by_name=None,
              log_dir=None, log_csv=None, patience=None, tb=False,
              target_transform="log1p", log10_eps=DEFAULT_LOG10_EPS):
    """
    target_transform: 'log1p' (default) or 'log10'
    log10_eps: epsilon used when target_transform='log10'
    """
    from .datasets import H5DRMDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    # Dataset: preload for speed, use selected target transform
    full_ds = H5DRMDataset(
        h5_path,
        target_transform=target_transform,
        log_eps=log10_eps,
        subset_idx=subset_idx,
        subset_by_name=subset_by_name,
        preload=True,
        return_concat=True,
        return_split=True,
    )

    N = len(full_ds)
    N_val = max(1, int(val_frac * N))
    N_trn = N - N_val
    train_ds, val_ds = random_split(full_ds, [N_trn, N_val],
                                    generator=torch.Generator().manual_seed(seed))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = DRMNet(out_len=out_len, use_embedding=use_embedding, num_det=num_det,
                   emb_dim=emb_dim, hidden=hidden, use_log_target=(target_transform in ("log1p","log10")),
                   low_rank_k=low_rank_k).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Logging
    label = f"MSE({target_transform})"
    csv_fields = ["epoch", f"train_{label}", f"val_{label}", "lr",
                  "val_rmse", "val_rel_mean", "val_rel_median", "val_rel_p90", "val_rel_p99", "val_cos_sim"]
    csv_logger = CSVLogger(log_csv, csv_fields) if log_csv else None

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
        extra = eval_metrics_linear(model, val_loader, device, target_transform=target_transform, log10_eps=log10_eps, progress=True)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        dt = time.time() - t0

        print(f"Epoch {ep:03d} | train {label}: {trn:.6f} | val {label}: {val:.6f} | "
              f"val RMSE: {extra['rmse_mean']:.4f} | val rel(mean/med/p90/p99): "
              f"{extra['rel_mean']:.3f}/{extra['rel_median']:.3f}/{extra['rel_p90']:.3f}/{extra['rel_p99']:.3f} | "
              f"val cos: {extra['cos_sim_mean']:.4f} | lr={lr_now:.2e} | {dt:.1f}s")

        if csv_logger:
            csv_logger.log({
                "epoch": ep,
                f"train_{label}": trn,
                f"val_{label}": val,
                "lr": lr_now,
                "val_rmse": extra["rmse_mean"],
                "val_rel_mean": extra["rel_mean"],
                "val_rel_median": extra["rel_median"],
                "val_rel_p90": extra["rel_p90"],
                "val_rel_p99": extra["rel_p99"],
                "val_cos_sim": extra["cos_sim_mean"],
            })
        if tb_writer:
            tb_writer.add_scalar(f"loss/train_{label}", trn, ep)
            tb_writer.add_scalar(f"loss/val_{label}", val, ep)
            tb_writer.add_scalar("val/rmse_linear", extra["rmse_mean"], ep)
            tb_writer.add_scalar("val/rel_mean", extra["rel_mean"], ep)
            tb_writer.add_scalar("val/rel_median", extra["rel_median"], ep)
            tb_writer.add_scalar("val/rel_p90", extra["rel_p90"], ep)
            tb_writer.add_scalar("val/rel_p99", extra["rel_p99"], ep)
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
                    "target_transform": target_transform, "log10_eps": float(log10_eps),
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
        try:
            tb_writer.close()
        except Exception:
            pass

    print(f"Saved best to {save_path} (val={best_val:.6f})")
    return save_path
