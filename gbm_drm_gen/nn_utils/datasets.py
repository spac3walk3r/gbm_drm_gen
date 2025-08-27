import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5DRMDataset(Dataset):
    def __init__(self, h5_path, use_log_target=True, subset_idx=None, subset_by_name=None, preload=False):
        """
        h5_path: path to HDF5 file with datasets:
          - inputs: [N, 7] = [src_az, src_el, geo_az, geo_el, nx, ny, nz] (degrees for angles)
          - drms:   [N, L] flattened DRM (float32)
          - det_id: [N]    optional int detector IDs
          - det_name: [N]  optional detector names (bytes or str)
        use_log_target: if True, targets are transformed with log1p
        subset_idx: selector values. If subset_by_name is True, this is an iterable of names
                    (e.g., ("BGO_00",) or ("b0",)). If False/None, this is an iterable of det_id integers
                    (e.g., (12,) or (13,)). If None, no filtering is applied.
        subset_by_name: boolean flag. If True, filter by det_name; if False/None, filter by det_id (or treat
                        subset_idx as row indices if det_id is absent).
        preload: if True, load and convert the entire subset into memory once (faster iteration, higher RAM).
        """
        self.h5_path = h5_path
        self.use_log_target = use_log_target
        self.preload = preload

        f = h5py.File(self.h5_path, "r")
        inputs = f["inputs"]
        drms = f["drms"]
        self._flat_len = drms.shape[1]

        # Build selection mapping (indices into the HDF5 datasets)
        def _decode_names(arr):
            return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr], dtype=object)

        # Map long to short codes (and inverse)
        long_to_short = {
            "NAI_00": "n0", "NAI_01": "n1", "NAI_02": "n2", "NAI_03": "n3",
            "NAI_04": "n4", "NAI_05": "n5", "NAI_06": "n6", "NAI_07": "n7",
            "NAI_08": "n8", "NAI_09": "n9", "NAI_10": "na", "NAI_11": "nb",
            "BGO_00": "b0", "BGO_01": "b1",
        }
        short_to_long = {v: k for k, v in long_to_short.items()}

        if subset_idx is not None:
            if subset_by_name:
                # Filter by detector name; accept both long and short forms
                if "det_name" not in f:
                    raise ValueError("subset_by_name=True but 'det_name' dataset not found in HDF5.")
                names = _decode_names(f["det_name"][:])  # array of Python strings
                # Build a set of target names including both long and short equivalents
                targets = set()
                for t in subset_idx:
                    t = t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
                    targets.add(t)
                    if t in long_to_short:
                        targets.add(long_to_short[t])
                    if t in short_to_long:
                        targets.add(short_to_long[t])
                # Build an array of short-form names for all rows to check both forms
                names_short = np.array([long_to_short.get(nm, nm) for nm in names], dtype=object)
                mask = np.isin(names, list(targets)) | np.isin(names_short, list(targets))
                idxs = np.flatnonzero(mask).astype(np.int64)
            else:
                # Filter by det_id values (e.g., 12 for BGO_00, 13 for BGO_01)
                if "det_id" in f:
                    det_ids = np.asarray(f["det_id"][:], dtype=np.int64)
                    wanted = np.asarray(subset_idx, dtype=np.int64)
                    mask = np.isin(det_ids, wanted)
                    idxs = np.flatnonzero(mask).astype(np.int64)
                else:
                    # Fallback: treat subset_idx as direct row indices if det_id is absent
                    idxs = np.asarray(subset_idx, dtype=np.int64)
        else:
            idxs = np.arange(inputs.shape[0], dtype=np.int64)

        if idxs.size == 0:
            f.close()
            raise ValueError("Empty dataset after subset filtering; check subset_idx/subset_by_name and HDF5 content.")

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
            angles = self._inputs_t[i, :4].contiguous()    # [4]
            normals = self._inputs_t[i, 4:7].contiguous()  # [3]
            y = self._y_t[i].contiguous()                  # [L]
            out = {"angles": angles, "normals": normals, "y": y}
            if self._det_id_t is not None:
                out["det_id"] = self._det_id_t[i]
            return out
        else:
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