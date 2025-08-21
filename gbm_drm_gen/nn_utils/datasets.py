import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5DRMDataset(Dataset):
    def __init__(self, h5_path, use_log_target=True, subset_idx=None, subset_by_name=None, preload=False):
        self.h5_path = h5_path
        self.use_log_target = use_log_target
        self.preload = preload

        f = h5py.File(self.h5_path, "r")
        inputs = f["inputs"]
        drms = f["drms"]
        self._flat_len = drms.shape[1]

        # Build selection mapping (indices into the HDF5 datasets)
        if subset_by_name is not None and "det_name" in f:
            names = f["det_name"][:]
            names = np.array([n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in names], dtype=object)
            mask = np.isin(names, subset_by_name)
            idxs = np.flatnonzero(mask).astype(np.int64)
        elif subset_idx is not None:
            idxs = np.asarray(subset_idx, dtype=np.int64)
        else:
            idxs = np.arange(inputs.shape[0], dtype=np.int64)
        self.idxs = idxs  # length M

        if preload:
            # Load once and convert once to torch
            X = inputs[self.idxs]                      # numpy [M, 7]
            Y = drms[self.idxs]                        # numpy [M, L]
            self._inputs_t = torch.from_numpy(X).float()     # [M, 7]
            self._y_t = torch.from_numpy(Y).float()          # [M, L]
            if self.use_log_target:
                self._y_t = torch.log1p(self._y_t)           # precompute once
            self._det_id_t = (torch.from_numpy(f["det_id"][self.idxs]).long()
                              if "det_id" in f else None)
            # Close file and clear HDF5 handles
            f.close()
            self._f = None
            self._inputs = None
            self._drms = None
            self._det_id = None
        else:
            # Keep file open; minimal per-sample conversion
            self._f = f
            self._inputs = inputs
            self._drms = drms
            self._det_id = f["det_id"] if "det_id" in f else None
            self._inputs_t = None
            self._y_t = None
            self._det_id_t = None

    @property
    def flat_len(self):
        return self._flat_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        # i is always 0..len(self)-1 (random_split/Subsets hand these to the base dataset)
        if self.preload:
            # Directly index prebuilt tensors; DO NOT remap via self.idxs here
            angles = self._inputs_t[i, :4].contiguous()    # [4]
            normals = self._inputs_t[i, 4:7].contiguous()  # [3]
            y = self._y_t[i].contiguous()                  # [L]
            out = {"angles": angles, "normals": normals, "y": y}
            if self._det_id_t is not None:
                out["det_id"] = self._det_id_t[i]
            return out
        else:
            # Map i -> original HDF5 index, then convert once
            ii = int(self.idxs[i])
            X = self._inputs[ii]                # numpy [7]
            Y = self._drms[ii]                  # numpy [L]
            det_id = self._det_id[ii] if self._det_id is not None else None
            angles = torch.from_numpy(X[:4]).float()
            normals = torch.from_numpy(X[4:7]).float()
            y = torch.from_numpy(Y).float()
            if self.use_log_target:
                y = torch.log1p(y)
            out = {"angles": angles, "normals": normals, "y": y}
            if det_id is not None:
                out["det_id"] = torch.as_tensor(det_id, dtype=torch.long)
            return out

    def __del__(self):
        try:
            if getattr(self, "_f", None) is not None:
                self._f.close()
        except Exception:
            pass
