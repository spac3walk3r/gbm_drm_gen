import numpy as np
import torch
from .model import DRMNet
from .threads import configure_threads
configure_threads(n_intra=16, n_interop=1)

def _infer_hidden_from_state(sd):
    hidden = []
    in_dim = None
    out_len = None
    idx = 0
    while True:
        key = f"mlp.{idx}.weight"
        if key not in sd:
            break
        w = sd[key]
        outd, ind = w.shape
        if idx == 0:
            in_dim = int(ind)
        next_key = f"mlp.{idx+2}.weight"
        if next_key in sd:
            hidden.append(int(outd))
        else:
            out_len = int(outd)
        idx += 2
    return tuple(hidden), out_len

def _invert_transform_np(y_trans, mode="log1p", eps=1e-12):
    if mode == "log10":
        return np.power(10.0, y_trans) - eps
    elif mode == "log1p":
        return np.expm1(y_trans)
    else:
        return y_trans

@torch.no_grad()
def load_model(ckpt_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    sd  = ckpt.get("model", ckpt)

    # Prefer hidden from cfg; fall back to inference
    hidden_from_sd, out_len_inferred = _infer_hidden_from_state(sd)
    hidden = tuple(cfg.get("hidden", ())) or hidden_from_sd
    emb_dim = int(cfg.get("emb_dim", 8))
    out_len = int(cfg.get("out_len", out_len_inferred))
    use_embedding = bool(cfg.get("use_embedding", False))
    num_det = int(cfg.get("num_det", 12))

    target_transform = cfg.get("target_transform", "log1p")
    log10_eps = float(cfg.get("log10_eps", 1e-12))

    model = DRMNet(out_len=out_len,
                   use_embedding=use_embedding,
                   num_det=num_det,
                   emb_dim=emb_dim,
                   hidden=hidden,
                   use_log_target=(target_transform in ("log1p", "log10")),
                   low_rank_k=None)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg, device, target_transform, log10_eps

@torch.no_grad()
def predict_flat(ckpt_path, angles_deg, normals, det_id=None, device=None):
    """
    angles_deg: [..., 4] [src_az, src_el, geo_az, geo_el] in deg
    normals:    [..., 3] detector normal unit vector
    det_id:     optional integer(s) if the model uses embedding
    Returns:    numpy array [..., out_len] on linear scale
    """
    model, cfg, device, target_transform, log10_eps = load_model(ckpt_path, device)
    a = torch.as_tensor(angles_deg, dtype=torch.float32, device=device).view(-1, 4)
    n = torch.as_tensor(normals,    dtype=torch.float32, device=device).view(-1, 3)
    if cfg.get("use_embedding", False):
        if det_id is None:
            raise ValueError("This model expects det_id for embedding")
        d = torch.as_tensor(det_id, dtype=torch.long, device=device).view(-1)
        y_trans = model(angles_deg=a, normals=n, det_id=d)
    else:
        y_trans = model(angles_deg=a, normals=n, det_id=None)
    y_trans_np = y_trans.detach().cpu().numpy()
    drm_flat = _invert_transform_np(y_trans_np, mode=target_transform, eps=log10_eps)
    drm_flat[drm_flat < 0] = 0.0
    return drm_flat

@torch.no_grad()
def predict_matrix(ckpt_path, angles_deg, normals, det_id=None, device=None):
    drm_flat = predict_flat(ckpt_path, angles_deg, normals, det_id, device)
    _, cfg, _, _, _ = load_model(ckpt_path, device)
    n_out = int(cfg.get("n_out", 140))
    n_in  = int(cfg.get("n_in", 128))
    return drm_flat.reshape(-1, n_out, n_in)