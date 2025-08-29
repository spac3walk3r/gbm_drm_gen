import math
import torch
import torch.nn as nn

def angles_to_trig(angles_deg: torch.Tensor) -> torch.Tensor:
    """
    angles_deg: [B, 4] = [src_az, src_el, geo_az, geo_el] in degrees
    Returns [B, 8]: [sin(src_az), cos(src_az), sin(src_el), cos(src_el),
                     sin(geo_az), cos(geo_az), sin(geo_el), cos(geo_el)]
    """
    deg2rad = math.pi / 180.0
    a = angles_deg * deg2rad
    sa = torch.sin(a); ca = torch.cos(a)
    return torch.stack([sa[:, 0], ca[:, 0],
                        sa[:, 1], ca[:, 1],
                        sa[:, 2], ca[:, 2],
                        sa[:, 3], ca[:, 3]], dim=1)

def unitvec_from_az_el(az_deg: torch.Tensor, el_deg: torch.Tensor) -> torch.Tensor:
    """
    az_deg, el_deg: [B]
    Returns unit vectors [B, 3] in spacecraft frame for those angles.
    """
    deg2rad = math.pi / 180.0
    az = az_deg * deg2rad
    el = el_deg * deg2rad
    x = torch.cos(el) * torch.cos(az)
    y = torch.cos(el) * torch.sin(az)
    z = torch.sin(el)
    return torch.stack([x, y, z], dim=1)

class DRMNet(nn.Module):
    """
    MLP mapping geometry features to flattened DRM.

    Input features used internally:
      - angles (sin/cos): 8 from [src_az, src_el, geo_az, geo_el]
      - detector normal: 3 = [nx, ny, nz]
      - physics scalars: 2 = [cos_to_nadir, cos_offaxis]
    Total base features = 13.
    Optional: detector embedding (e.g., for NaI multi-detector model)

    out_len: flattened DRM length (e.g., 17920 for 140 x 128 CSPEC)
    """

    def __init__(self,
                 out_len: int,
                 use_embedding: bool = False,
                 num_det: int = 12,
                 emb_dim: int = 8,
                 hidden=(128, 256, 16),
                 use_log_target: bool = True,
                 low_rank_k: int = None):
        super().__init__()
        self.use_embedding = use_embedding
        self.use_log_target = use_log_target

        # Base input: 8 (angles trig) + 3 (normals) + 2 (phys)
        base_in_dim = 8 + 3 + 2

        if use_embedding:
            self.det_emb = nn.Embedding(num_det, emb_dim)
            in_dim = base_in_dim + emb_dim
        else:
            self.det_emb = None
            in_dim = base_in_dim

        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        if low_rank_k is None:
            layers += [nn.Linear(last, out_len)]
        else:
            layers += [nn.Linear(last, low_rank_k), nn.ReLU(),
                       nn.Linear(low_rank_k, out_len)]
        self.mlp = nn.Sequential(*layers)

    def _prepare_from_batch(self, batch):
        """
        Accepts a batch dict from H5DRMDataset:
          - preferred: batch["x"] with 9 features:
              [src_az, src_el, geo_az, geo_el, nx, ny, nz, cos_to_nadir, cos_offaxis]
          - or legacy: batch["angles"], batch["normals"], optional batch["phys"]
        Returns (angles_deg[B,4], normals[B,3], phys[B,2], det_id or None)
        """
        det_id = batch.get("det_id", None)

        if "x" in batch:
            x = batch["x"]
            angles_deg = x[:, 0:4]
            normals = x[:, 4:7]
            # if phys provided in 'x', take it; otherwise compute
            phys = x[:, 7:9] if x.shape[1] >= 9 else None
        else:
            angles_deg = batch["angles"]            # [B,4]
            normals = batch["normals"]              # [B,3]
            phys = batch.get("phys", None)          # [B,2] or None

        return angles_deg, normals, phys, det_id

    def _prepare_from_args(self, angles_deg=None, normals=None, phys=None, x=None, det_id=None):
        """
        Accepts either:
          - angles_deg [B,4], normals [B,3], optional phys [B,2]
          - or x [B,9] containing angles, normals, phys in that order
        Returns (angles_deg, normals, phys, det_id)
        """
        if x is not None:
            angles_deg = x[:, 0:4]
            normals = x[:, 4:7]
            phys = x[:, 7:9] if x.shape[1] >= 9 else phys
        if angles_deg is None or normals is None:
            raise ValueError("Provide either x[...9] or (angles_deg[...4] and normals[...3]).")
        return angles_deg, normals, phys, det_id

    def forward(self, angles_deg=None, normals=None, det_id=None, phys=None, x=None, batch=None):
        """
        Flexible forward:
          - forward(batch=batch) where batch from H5DRMDataset (preferred)
          - forward(x=...) with 9 features
          - forward(angles_deg=..., normals=..., phys=..., det_id=...)
        """
        if batch is not None:
            angles_deg, normals, phys, det_id = self._prepare_from_batch(batch)
        else:
            angles_deg, normals, phys, det_id = self._prepare_from_args(
                angles_deg=angles_deg, normals=normals, phys=phys, x=x, det_id=det_id
            )

        # Build physics-aware features if not provided
        if phys is None:
            # Compute source and nadir unit vectors from angles
            src = unitvec_from_az_el(angles_deg[:, 0], angles_deg[:, 1])   # [B,3]
            nad = unitvec_from_az_el(angles_deg[:, 2], angles_deg[:, 3])   # [B,3]
            cos_to_nadir = torch.sum(src * nad, dim=1, keepdim=True)       # [B,1]
            cos_offaxis = torch.sum(src * normals, dim=1, keepdim=True)    # [B,1]
            phys = torch.cat([cos_to_nadir, cos_offaxis], dim=1)           # [B,2]

        # Angles trig-encoding
        ang_trig = angles_to_trig(angles_deg)          # [B,8]

        # Concatenate base features: [B, 8+3+2] = [B, 13]
        feat = torch.cat([ang_trig, normals, phys], dim=1)

        # Optional detector embedding
        if self.use_embedding:
            if det_id is None:
                raise ValueError("use_embedding=True but det_id is None")
            emb = self.det_emb(det_id)                 # [B, emb_dim]
            feat = torch.cat([feat, emb], dim=1)

        y = self.mlp(feat)                             # [B, out_len]
        # Output is unbounded (you train on transformed targets, e.g., log10 or log1p)
        return y
