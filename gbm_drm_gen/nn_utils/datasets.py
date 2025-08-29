import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5DRMDataset(Dataset):
    def __init__(self,
                 h5_path,
                 target_transform="log10",   # 'log10' (default), 'log1p', or 'none'
                 log_eps=1e-12,
                 subset_idx=None,
                 subset_by_name=None,
                 preload=False,
                 return_concat=True,         # return 'x' (all 9 features)
                 return_split=True):         # return 'angles','normals','phys' too (legacy support)
        """
        h5_path: path to HDF5 file with datasets:
          - inputs: [N, 9] =
              [src_az_deg, src_el_deg, geo_az_deg, geo_el_deg,
               det_nx, det_ny, det_nz, cos_to_nadir, cos_offaxis]
          - drms:   [N, L] flattened DRM (float32)
          - det_id: [N]    optional int detector IDs
          - det_name: [N]  optional detector names (bytes or str)

        target_transform:
          - 'log10': y -> log10(y + log_eps)
          - 'log1p': y -> log(1 + y)
          - 'none' : y unchanged
        log_eps: epsilon for 'log10' stabilization

        subset_idx / subset_by_name: optional filtering by det_id or det_name (same behavior as before)
        preload: if True, loads the selected subset into RAM (faster iteration, more memory)
        return_concat: include key 'x' with all 9 input features
        return_split:  include legacy keys: 'angles'[:4], 'normals'[4:7], 'phys'[7:9]
        """
        self.h5_path = h5_path
        self.target_transform = target_transform
        self.log_eps = float(log_eps)
        self.preload = preload
        self.return_concat = bool(return_concat)
        self.return_split = bool(return_split)

        f = h5py.File(self.h5_path, "r")
        inputs = f["inputs"]
        drms = f["drms"]
        self._flat_len = int(drms.shape[1])

        # Build selection mapping (indices into the HDF5 datasets)
        def _decode_names(arr):
            return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr], dtype=object)

        long_to_short = {
            "NAI_00": "n0", "NAI_01": "n1", "NAI_02": "n2", "NAI_03": "n3",
            "NAI_04": "n4", "NAI_05": "n5", "NAI_06": "n6", "NAI_07": "n7",
            "NAI_08": "n8", "NAI_09": "n9", "NAI_10": "na", "NAI_11": "nb",
            "BGO_00": "b0", "BGO_01": "b1",
        }
        short_to_long = {v: k for k, v in long_to_short.items()}

        if subset_idx is not None:
            if subset_by_name:
                if "det_name" not in f:
                    f.close()
                    raise ValueError("subset_by_name=True but 'det_name' dataset not found in HDF5.")
                names = _decode_names(f["det_name"][:])
                targets = set()
                for t in subset_idx:
                    t = t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
                    targets.add(t)
                    if t in long_to_short:
                        targets.add(long_to_short[t])
                    if t in short_to_long:
                        targets.add(short_to_long[t])
                names_short = np.array([long_to_short.get(nm, nm) for nm in names], dtype=object)
                mask = np.isin(names, list(targets)) | np.isin(names_short, list(targets))
                idxs = np.flatnonzero(mask).astype(np.int64)
            else:
                if "det_id" in f:
                    det_ids = np.asarray(f["det_id"][:], dtype=np.int64)
                    wanted = np.asarray(subset_idx, dtype=np.int64)
                    mask = np.isin(det_ids, wanted)
                    idxs = np.flatnonzero(mask).astype(np.int64)
                else:
                    idxs = np.asarray(subset_idx, dtype=np.int64)
        else:
            idxs = np.arange(inputs.shape[0], dtype=np.int64)

        if idxs.size == 0:
            f.close()
            raise ValueError("Empty dataset after subset filtering; check subset_idx/subset_by_name and HDF5 content.")

        self.idxs = idxs  # length M

        # Helper: apply target transform
        def _transform_y(y_np):
            if self.target_transform == "log10":
                return np.log10(y_np + self.log_eps, dtype=np.float32)
            elif self.target_transform == "log1p":
                return np.log1p(y_np).astype(np.float32)
            else:
                return y_np.astype(np.float32)

        if preload:
            X = inputs[self.idxs]                                # [M, 9]
            Y = drms[self.idxs]                                  # [M, L]
            self._inputs_t = torch.from_numpy(X).float()         # [M, 9]
            Yt = _transform_y(Y)
            self._y_t = torch.from_numpy(Yt).float()             # [M, L]
            self._det_id_t = (torch.from_numpy(f["det_id"][self.idxs]).long()
                              if "det_id" in f else None)
            f.close()
            self._f = None
            self._inputs = None
            self._drms = None
            self._det_id = None
        else:
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

    @property
    def input_dim(self):
        return 9

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        if self.preload:
            X = self._inputs_t[i].contiguous()      # [9]
            y = self._y_t[i].contiguous()           # [L]
            out = {"y": y}
            if self.return_concat:
                out["x"] = X
            if self.return_split:
                out["angles"] = X[:4].contiguous()  # [src_az, src_el, geo_az, geo_el]
                out["normals"] = X[4:7].contiguous()
                out["phys"] = X[7:9].contiguous()   # [cos_to_nadir, cos_offaxis]
            if self._det_id_t is not None:
                out["det_id"] = self._det_id_t[i]
            return out
        else:
            ii = int(self.idxs[i])
            X = self._inputs[ii]                    # numpy [9]
            Y = self._drms[ii]                      # numpy [L]
            y_np = Y  # will transform below
            if self.target_transform == "log10":
                y_t = torch.from_numpy(np.log10(y_np + self.log_eps, dtype=np.float32))
            elif self.target_transform == "log1p":
                y_t = torch.from_numpy(np.log1p(y_np).astype(np.float32))
            else:
                y_t = torch.from_numpy(y_np.astype(np.float32))
            out = {"y": y_t}
            X_t = torch.from_numpy(X.astype(np.float32))
            if self.return_concat:
                out["x"] = X_t
            if self.return_split:
                out["angles"] = X_t[:4]
                out["normals"] = X_t[4:7]
                out["phys"] = X_t[7:9]
            if self._det_id is not None:
                det_id = self._det_id[ii]
                out["det_id"] = torch.as_tensor(det_id, dtype=torch.long)
            return out

    def __del__(self):
        try:
            if getattr(self, "_f", None) is not None:
                self._f.close()
        except Exception:
            pass