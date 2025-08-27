import math
import torch
import torch.nn as nn

def angles_to_trig(angles_deg):
    """
    angles_deg: [B, 4] = [src_az, src_el, geo_az, geo_el] in degrees
    Returns [B, 8]: sin/cos for each angle (wrap-around safe).
    """
    deg2rad = math.pi / 180.0
    a = angles_deg * deg2rad
    sa = torch.sin(a); ca = torch.cos(a)
    # interleave sin/cos pairs in fixed order
    return torch.stack([sa[:,0], ca[:,0], sa[:,1], ca[:,1], sa[:,2], ca[:,2], sa[:,3], ca[:,3]], dim=1)

class DRMNet(nn.Module):
    """
    MLP mapping [angles (sin/cos), normals, optional det embedding] -> flattened DRM.
    - For NaI: use_embedding=True, num_det=12
    - For BGO models (one per detector): use_embedding=False
    """
    def __init__(self, out_len, use_embedding=False, num_det=12, emb_dim=8,
                 hidden=(128, 256, 16), use_log_target=True, low_rank_k=None):
                 
        super().__init__()
        self.use_embedding = use_embedding
        self.use_log_target = use_log_target

        if use_embedding:
            self.det_emb = nn.Embedding(num_det, emb_dim)
            in_dim = 8 + 3 + emb_dim  # angles(8) + normals(3) + embedding
        else:
            self.det_emb = None
            in_dim = 8 + 3

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

    def forward(self, angles_deg, normals, det_id=None):
        x = angles_to_trig(angles_deg)          # [B, 8]
        x = torch.cat([x, normals], dim=1)      # [B, 11]
        if self.use_embedding:
            x = torch.cat([x, self.det_emb(det_id)], dim=1)
        y = self.mlp(x)                         # [B, out_len]
        # When training on log1p target, keep linear output (unbounded).
        # If you ever switch to linear target, you could ReLU here to enforce non-negativity.
        return y
