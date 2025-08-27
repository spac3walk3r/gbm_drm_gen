import os, threading, numpy as np
def _lazy_torch():
    import torch
    return torch

DET_NAME_TO_ID = {f"NAI_{i:02d}": i for i in range(12)}
DET_NAME_TO_ID.update({"BGO_00": 12, "BGO_01": 13})

DET_ORIENT_DEG = {
    "NAI_00": (45.89, 20.58), "NAI_01": (45.11, 45.31), "NAI_02": (58.44, 90.21),
    "NAI_03": (314.87, 45.24), "NAI_04": (303.15, 90.27), "NAI_05": (3.35, 89.97),
    "NAI_06": (224.93, 20.43), "NAI_07": (224.62, 46.18), "NAI_08": (236.61, 89.97),
    "NAI_09": (135.19, 45.55), "NAI_10": (123.73, 90.42), "NAI_11": (183.74, 90.32),
    "BGO_00": (0.00, 90.00), "BGO_01": (180.00, 90.00),
}
def _azzen_to_unitvec(az_deg, zen_deg):
    az = np.deg2rad(az_deg); zen = np.deg2rad(zen_deg)
    nx = np.sin(zen) * np.cos(az); ny = np.sin(zen) * np.sin(az); nz = np.cos(zen)
    return np.array([nx, ny, nz], dtype=np.float32)
DET_NORMAL = {k: _azzen_to_unitvec(*v) for k, v in DET_ORIENT_DEG.items()}

def _infer_mlp_from_state(sd):
    # Infer first Linear input dim, hidden sizes, and out_len from mlp.*.weight tensors
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
        # If there is another linear after this, treat this as a hidden layer
        next_key = f"mlp.{idx+2}.weight"
        if next_key in sd:
            hidden.append(int(outd))
        else:
            out_len = int(outd)  # final layer
        idx += 2
    if in_dim is None or out_len is None:
        raise RuntimeError("Cannot infer MLP layout from state_dict")
    return in_dim, tuple(hidden), out_len

class ModelRegistry:
    def __init__(self, models_dir=None, device="cpu"):
        self._dir = models_dir or os.environ.get("GBM_DRM_GEN_NN_DIR", ".")
        self._dev = device
        self._torch = None
        self._models = {}
        self._cfgs = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _load_one(self, path):
        t = self._torch
        ckpt = t.load(path, map_location=self._dev)
        # Allow both styles: {"model": state_dict, "cfg": {...}} or plain state_dict
        cfg = ckpt.get("cfg", {})
        sd = ckpt.get("model", ckpt)

        # Build DRMNet to match the checkpoint exactly
        from gbm_drm_gen.nn_utils.model import DRMNet

        # Infer hidden layout and out_len from the tensors (e.g., (128, 256, 16) and 17920)
        in_dim_ckpt, hidden_ckpt, out_len_ckpt = _infer_mlp_from_state(sd)

        # Embedding info from cfg
        use_embedding = bool(cfg.get("use_embedding", False))
        emb_dim = int(cfg.get("emb_dim", 0))
        num_det = int(cfg.get("num_det", 12))

        # Construct model
        model = DRMNet(
            out_len=out_len_ckpt,
            use_embedding=use_embedding,
            num_det=num_det,
            emb_dim=emb_dim,
            hidden=hidden_ckpt,
            use_log_target=bool(cfg.get("use_log_target", True)),
            low_rank_k=cfg.get("low_rank_k", None),
        )
        model.load_state_dict(sd, strict=True)
        model.to(self._dev).eval()

        # Normalize cfg so downstream code sees the true values
        cfg = dict(cfg)  # copy
        cfg["out_len"] = out_len_ckpt
        cfg["hidden"] = hidden_ckpt

        # Keep n_in/n_out if present; they are used to reshape the flat vector
        # If missing, try to keep current ones; otherwise, you can set them here.

        return model, cfg

    def _lazy_load(self):
        with self._lock:
            if self._loaded:
                return
            self._torch = _lazy_torch()
            # NaI (single model for all 12)
            p = os.path.join(self._dir, "drmnet_nai.pt")
            if os.path.exists(p):
                self._models["NAI"], self._cfgs["NAI"] = self._load_one(p)
            # BGO_00
            p = os.path.join(self._dir, "drmnet_bgo00.pt")
            if os.path.exists(p):
                self._models["BGO_00"], self._cfgs["BGO_00"] = self._load_one(p)
            # BGO_01
            p = os.path.join(self._dir, "drmnet_bgo01.pt")
            if os.path.exists(p):
                self._models["BGO_01"], self._cfgs["BGO_01"] = self._load_one(p)
            self._loaded = True

    def predict(self, det_name, src_az, src_el, geo_az, geo_el):
        # Note: DRMNet expects angles in degrees (it converts to trig internally).
        self._lazy_load()
        t = self._torch
        key = "NAI" if det_name.startswith("NAI_") else det_name
        if key not in self._models:
            raise RuntimeError(f"No NN model loaded for {det_name} (key {key}) in {self._dir}")
        model, cfg = self._models[key], self._cfgs[key]

        angles = t.tensor([[src_az, src_el, geo_az, geo_el]], dtype=t.float32, device=self._dev)
        nx, ny, nz = DET_NORMAL[det_name]
        normals = t.tensor([[nx, ny, nz]], dtype=t.float32, device=self._dev)

        with t.no_grad():
            if cfg.get("use_embedding", False):
                det_id = t.tensor([DET_NAME_TO_ID[det_name]], dtype=t.long, device=self._dev)
                yhat_log = model(angles, normals, det_id)
            else:
                yhat_log = model(angles, normals, None)
            drm = t.expm1(yhat_log).clamp_min_(0.0)

            # Reshape using cfg-provided n_out/n_in (your checkpoint had n_out=128, n_in=140)
            n_out = int(cfg.get("n_out", 128))
            n_in = int(cfg.get("n_in", 140))
            drm = drm.view(n_out, n_in).cpu().numpy()

        return drm  # (n_out, n_in)
