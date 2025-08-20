import torch
from .model import DRMNet

@torch.no_grad()
def load_model(ckpt_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    hidden = tuple(cfg.get("hidden", (256, 512, 512)))
    emb_dim = int(cfg.get("emb_dim", 8))
    model = DRMNet(out_len=cfg["out_len"],
                   use_embedding=cfg["use_embedding"],
                   num_det=cfg["num_det"],
                   emb_dim=emb_dim,
                   hidden=hidden)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, device

@torch.no_grad()
def predict_flat(ckpt_path, angles_deg, normals, det_id=None, device=None):
    import numpy as np
    model, cfg, device = load_model(ckpt_path, device)
    a = torch.as_tensor(angles_deg, dtype=torch.float32, device=device).view(-1, 4)
    n = torch.as_tensor(normals,    dtype=torch.float32, device=device).view(-1, 3)
    if cfg["use_embedding"]:
        if det_id is None:
            raise ValueError("This model expects det_id for embedding")
        d = torch.as_tensor(det_id, dtype=torch.long, device=device).view(-1)
        yhat = model(a, n, d)
    else:
        yhat = model(a, n, None)
    drm = torch.expm1(yhat).clamp_min_(0.0)
    return drm.cpu().numpy()

@torch.no_grad()
def predict_matrix(ckpt_path, angles_deg, normals, det_id=None, device=None):
    drm_flat = predict_flat(ckpt_path, angles_deg, normals, det_id, device)
    model, cfg, _ = load_model(ckpt_path, device)
    return drm_flat.reshape(-1, cfg["n_out"], cfg["n_in"])