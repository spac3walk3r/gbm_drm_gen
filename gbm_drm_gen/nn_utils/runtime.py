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
        cfg = ckpt["cfg"]
        from gbm_drm_gen.nn_utils.model import DRMNet
        model = DRMNet(out_len=cfg["out_len"],
                       use_embedding=cfg["use_embedding"],
                       num_det=cfg["num_det"],
                       emb_dim=int(cfg.get("emb_dim", 8)),
                       hidden=tuple(cfg.get("hidden", (256,512,512))),
                       use_log_target=True,
                       low_rank_k=cfg.get("low_rank_k", None))
        model.load_state_dict(ckpt["model"])
        model.to(self._dev).eval()
        return model, cfg

    def _lazy_load(self):
        with self._lock:
            if self._loaded: return
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
            if cfg["use_embedding"]:
                det_id = t.tensor([DET_NAME_TO_ID[det_name]], dtype=t.long, device=self._dev)
                yhat_log = model(angles, normals, det_id)
            else:
                yhat_log = model(angles, normals, None)
            drm = t.expm1(yhat_log).clamp_min_(0.0)
            drm = drm.view(cfg["n_out"], cfg["n_in"]).cpu().numpy()
        return drm  # (n_out, n_in)
