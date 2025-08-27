#!/usr/bin/env python3
"""
High-throughput NN DRM training dataset generator.

- Single multiprocessing Pool across all detectors and all samples
- Per-process DRM generator cache built in a Pool initializer (works with 'spawn')
- Streams results to HDF5 in batches (append), avoiding large RAM spikes
- Samples only visible geometries (source above Earth limb by a fixed angle)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import h5py
from multiprocessing import cpu_count, get_context
from tqdm import tqdm

# Visibility control
ENFORCE_VISIBILITY = True
LIMB_ANGLE_DEG = 23.8

# Detector orientation (az, zen) degrees (detector normal in SC frame)
det_orient_deg = {
    "NAI_00": (45.89, 20.58), "NAI_01": (45.11, 45.31), "NAI_02": (58.44, 90.21),
    "NAI_03": (314.87, 45.24), "NAI_04": (303.15, 90.27), "NAI_05": (3.35, 89.97),
    "NAI_06": (224.93, 20.43), "NAI_07": (224.62, 46.18), "NAI_08": (236.61, 89.97),
    "NAI_09": (135.19, 45.55), "NAI_10": (123.73, 90.42), "NAI_11": (183.74, 90.32),
    "BGO_00": (0.00, 90.00), "BGO_01": (180.00, 90.00),
}
DET_NAME_TO_ID = {f"NAI_{i:02d}": i for i in range(12)}
DET_NAME_TO_ID.update({"BGO_00": 12, "BGO_01": 13})

def azzen_to_unitvec(az_deg, zen_deg):
    az = np.radians(az_deg); zen = np.radians(zen_deg)
    nx = np.sin(zen) * np.cos(az); ny = np.sin(zen) * np.sin(az); nz = np.cos(zen)
    return np.array([nx, ny, nz], dtype=np.float32)

det_orient_vec = {det: azzen_to_unitvec(*angles) for det, angles in det_orient_deg.items()}

def unitvec_from_az_el(az, el):
    azr = np.radians(az); elr = np.radians(el)
    x = np.cos(elr) * np.cos(azr)
    y = np.cos(elr) * np.sin(azr)
    z = np.sin(elr)
    return np.stack([x, y, z], axis=0)

def enforce_visibility(src_az, src_el, geo_az, geo_el, limb_angle_deg=23.8):
    s = unitvec_from_az_el(src_az, src_el)
    e = unitvec_from_az_el(geo_az, geo_el)
    dot = np.clip(np.sum(s * e, axis=0), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return ang >= limb_angle_deg

def sample_coords(n_samples,
                  limb_emphasis=True,
                  rare_fraction=0.05,
                  seed=None,
                  enforce_vis=True,
                  limb_angle_deg=23.8):
    rng = np.random.default_rng(seed)
    sa_all, se_all, ga_all, ge_all = [], [], [], []
    need = n_samples
    batch = max(4 * n_samples, 1000)
    while need > 0:
        sa = rng.uniform(0, 360, batch)
        u = rng.uniform(-1, 1, batch)
        se = np.degrees(np.arcsin(u))
        ga = rng.uniform(0, 360, batch)

        ge = np.empty(batch)
        n_main = int(batch * (1 - rare_fraction))
        ge[:n_main] = rng.uniform(-90, -40, n_main)
        ge[n_main:] = rng.uniform(-90, 90, batch - n_main)
        if limb_emphasis and n_main > 0:
            n_limb = max(1, n_main // 5)
            idx = rng.choice(n_main, n_limb, replace=False)
            ge[idx] = rng.uniform(-65, -45, n_limb)

        m = enforce_visibility(sa, se, ga, ge, limb_angle_deg) if enforce_vis else np.ones(batch, dtype=bool)
        if m.any():
            take = min(need, int(m.sum()))
            keep = np.flatnonzero(m)[:take]
            sa_all.append(sa[keep]); se_all.append(se[keep])
            ga_all.append(ga[keep]); ge_all.append(ge[keep])
            need -= take

    return (np.concatenate(sa_all), np.concatenate(se_all),
            np.concatenate(ga_all), np.concatenate(ge_all))

# Per-process generator cache (built in worker init)
_GEN_CACHE = {}

def _worker_init(det_list, mat_type=2, occult=False):
    """
    Runs once in each worker to build a local cache of DRMGenMock for all detectors.
    """
    global _GEN_CACHE
    from gbm_drm_gen.drmgen_mock import DRMGenMock
    _GEN_CACHE = {}
    for det in det_list:
        _GEN_CACHE[det] = DRMGenMock(det_name=det, mat_type=mat_type, occult=occult)
    # Do NOT touch .matrix here; let each worker JIT on first use

def process_sample_cached(args):
    """
    Worker function: uses the per-process cache. Returns (features, drm_flat, det_id, det_name).
    """
    det_name, src_az, src_el, geo_az, geo_el = args
    gen = _GEN_CACHE[det_name]
    # Assuming DRMGenMock has a recompute method that takes angles in degrees.
    drm_matrix = gen.recompute(src_az, src_el, geo_az, geo_el)

    nx, ny, nz = det_orient_vec[det_name]
    features = np.array([src_az, src_el, geo_az, geo_el, nx, ny, nz], dtype=np.float32)
    det_id = np.int16(DET_NAME_TO_ID[det_name])
    return features, drm_matrix.reshape(-1).astype(np.float32), det_id, det_name

def build_dataset(det_list,
                  n_samples_per_det,
                  output_file,
                  seed_global=42,
                  processes=None,
                  chunksize=256,
                  bufsize=1024,
                  mat_type=2,
                  occult=False):
    """
    Build HDF5 dataset for given detectors with a single 'spawn' Pool.

    - det_list: ["NAI_00", ..., "NAI_11"] or ["BGO_00","BGO_01"]
    - n_samples_per_det: number of samples per detector
    - output_file: HDF5 path
    """
    rng = np.random.default_rng(seed_global)
    processes = processes or cpu_count()

    # Build the task list
    tasks = []
    for det_name in det_list:
        print(f"Sampling for {det_name}...")
        sa, se, ga, ge = sample_coords(
            n_samples_per_det,
            limb_emphasis=True,
            rare_fraction=0.05,
            seed=rng.integers(0, 1_000_000_000),
            enforce_vis=ENFORCE_VISIBILITY,
            limb_angle_deg=LIMB_ANGLE_DEG,
        )
        tasks.extend((det_name, a, b, c, d) for a, b, c, d in zip(sa, se, ga, ge))

    # Probe one result in the parent (build a temporary worker cache for a single det)
    # To avoid importing DRMGenMock at top-level, simulate via the worker path:
    # Create a tiny one-off cache in-process for sizing only
    from gbm_drm_gen.drmgen_mock import DRMGenMock
    _tmp = DRMGenMock(det_name=det_list[0], mat_type=mat_type, occult=occult)
    f0_det, a0 = det_list[0], tasks[0]
    _ = _tmp.recompute(a0[1], a0[2], a0[3], a0[4])
    nx, ny, nz = det_orient_vec[f0_det]
    f0 = np.array([a0[1], a0[2], a0[3], a0[4], nx, ny, nz], dtype=np.float32)
    y0 = _tmp.matrix.reshape(-1).astype(np.float32)
    flat_len = y0.size
    feat_len = f0.size

    # Create extendable HDF5 datasets
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        ds_X  = f.create_dataset("inputs",    shape=(0, feat_len), maxshape=(None, feat_len), dtype="f4", chunks=True)
        ds_Y  = f.create_dataset("drms",      shape=(0, flat_len), maxshape=(None, flat_len), dtype="f4", chunks=True)
        ds_id = f.create_dataset("det_id",    shape=(0,),         maxshape=(None,),          dtype="i2", chunks=True)
        ds_nm = f.create_dataset("det_name",  shape=(0,),         maxshape=(None,),          dtype=h5py.string_dtype("utf-8"), chunks=True)
        f.attrs["mat_type"] = mat_type
        f.attrs["occult"]   = bool(occult)

        def append_batch(batch):
            if not batch: return
            n_old = ds_X.shape[0]
            n_new = n_old + len(batch)
            ds_X.resize((n_new, feat_len)); ds_Y.resize((n_new, flat_len))
            ds_id.resize((n_new,)); ds_nm.resize((n_new,))
            for j, (features, drm_flat, det_id_j, det_name_j) in enumerate(batch):
                ds_X[n_old + j]  = features
                ds_Y[n_old + j]  = drm_flat
                ds_id[n_old + j] = det_id_j
                ds_nm[n_old + j] = det_name_j

        print(f"Launching Pool with {processes} processes; writing to {output_file}")
        ctx = get_context("spawn")
        with ctx.Pool(processes=processes,
                      initializer=_worker_init,
                      initargs=(det_list, mat_type, occult)) as pool:
            buffer = []
            for res in tqdm(pool.imap_unordered(process_sample_cached, tasks, chunksize=chunksize),
                            total=len(tasks)):
                buffer.append(res)
                if len(buffer) >= bufsize:
                    append_batch(buffer); buffer.clear()
            if buffer:
                append_batch(buffer)

    print(f"Saved {output_file}")

if __name__ == "__main__":
    # small example
    N_SAMPLES_PER_DET = 10
    OUTPUT_FILE = "nais_training.h5"
    NAI_DETECTORS = [f"NAI_{i:02d}" for i in range(12)]
    build_dataset(NAI_DETECTORS, N_SAMPLES_PER_DET, OUTPUT_FILE, seed_global=42,
                  processes=min(16, cpu_count()), chunksize=256, bufsize=1024, mat_type=2, occult=False)
