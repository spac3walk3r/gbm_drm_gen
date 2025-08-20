import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5DRMDataset(Dataset):
    """
    HDF5 dataset for GBM DRM training.

    Expects datasets:
      - inputs: [N, 7] = [src_az, src_el, geo_az, geo_el, nx, ny, nz]
      - drms:   [N, L] flattened DRM
      - det_id: [N] (int), optional (present in your files)
      - det_name: [N] (UTF-8 strings), optional (present in your files)

    Options:
      - use_log_target: if True, returns y = log1p(drm)
      - subset_idx: list/array of indices to select
      - subset_by_name: list of detector names to select (e.g., ["BGO_00"])
    """
    def __init__(self, h5_path, use_log_target=True, subset_idx=None, subset_by_name=None):
        self.h5_path = h5_path
        self.use_log_target = use_log_target

        with h5py.File(self.h5_path, "r") as f:
            N = f["inputs"].shape[0]
            if subset_by_name is not None and "det_name" in f:
                names = f["det_name"][:]
                # decode bytes to str if needed
                names = np.array([n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in names], dtype=object)
                mask = np.isin(names, subset_by_name)
                idxs = np.flatnonzero(mask)
            elif subset_idx is not None:
                idxs = np.asarray(subset_idx, dtype=np.int64)
            else:
                idxs = np.arange(N, dtype=np.int64)

            self.idxs = idxs
            self._flat_len = f["drms"].shape[1]

    @property
    def flat_len(self):
        return self._flat_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = int(self.idxs[i])
        with h5py.File(self.h5_path, "r") as f:
            X = f["inputs"][idx]    # [7]
            Y = f["drms"][idx]      # [L]
            det_id = f["det_id"][idx] if "det_id" in f else None

        angles = torch.tensor(X[:4], dtype=torch.float32)    # [src_az, src_el, geo_az, geo_el] in deg
        normals = torch.tensor(X[4:7], dtype=torch.float32)  # [nx, ny, nz]
        y = torch.tensor(Y, dtype=torch.float32)             # [L]
        if self.use_log_target:
            y = torch.log1p(y)

        out = {"angles": angles, "normals": normals, "y": y}
        if det_id is not None:
            out["det_id"] = torch.tensor(det_id, dtype=torch.long)
        return out