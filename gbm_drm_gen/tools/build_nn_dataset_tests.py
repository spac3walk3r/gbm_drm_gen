#!/usr/bin/env python3
"""
Parallel NN DRM training dataset generator.

Generates (input_features, DRMs) for a list of GBM detectors (either all NaIs or BGOs),
sampling realistic spacecraft geometry and embedding detector orientation vectors.

- Uses DRMGenMock(det_name, occult=False) and per-process caching for speed.
- Samples only visible geometries (source above Earth limb) using a fixed limb angle.
- Saves inputs, drms, det_id (int), and det_name (UTF-8 string) to HDF5.

For using locally and doing small dataset generation tests.

"""

import os, logging, warnings

# Silence thread-count warnings and avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Make logging quiet (must be set before threeML/astromodels import)
logging.basicConfig(level=logging.ERROR)
for name in ("threeML", "astromodels"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Silence the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)



import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from gbm_drm_gen.drmgen_mock import DRMGenMock

# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------

N_SAMPLES_PER_DET = 10                  # per detector
OUTPUT_FILE = "nais_training.h5"        # change to bgos_training.h5 for BGOs
SEED_GLOBAL = 42

# Visibility control (keep source above Earth limb by at least this angle)
ENFORCE_VISIBILITY = True
LIMB_ANGLE_DEG = 23.8  # ~Fermi-typical limb angle at ~565 km altitude

# Detector orientation (az, zen) in degrees (detector normal in spacecraft frame)
det_orient_deg = {
    "NAI_00": (45.89, 20.58),
    "NAI_01": (45.11, 45.31),
    "NAI_02": (58.44, 90.21),
    "NAI_03": (314.87, 45.24),
    "NAI_04": (303.15, 90.27),
    "NAI_05": (3.35, 89.97),
    "NAI_06": (224.93, 20.43),
    "NAI_07": (224.62, 46.18),
    "NAI_08": (236.61, 89.97),
    "NAI_09": (135.19, 45.55),
    "NAI_10": (123.73, 90.42),
    "NAI_11": (183.74, 90.32),
    "BGO_00": (0.00, 90.00),
    "BGO_01": (180.00, 90.00),
}

# Integer IDs for detectors (useful for ML; stable across runs)
DET_NAME_TO_ID = {f"NAI_{i:02d}": i for i in range(12)}
DET_NAME_TO_ID.update({"BGO_00": 12, "BGO_01": 13})

def azzen_to_unitvec(az_deg, zen_deg):
    """Convert azimuth/zenith to unit vector in SC frame."""
    az = np.radians(az_deg)
    zen = np.radians(zen_deg)
    nx = np.sin(zen) * np.cos(az)
    ny = np.sin(zen) * np.sin(az)
    nz = np.cos(zen)
    return np.array([nx, ny, nz], dtype=np.float32)

det_orient_vec = {det: azzen_to_unitvec(*angles) for det, angles in det_orient_deg.items()}

# --------------------------------------------------------------------------------
# SAMPLING HELPERS
# --------------------------------------------------------------------------------

def unitvec_from_az_el(az, el):
    """Vector from az/el (deg) in SC frame, batch-aware. Returns array shape (3, N)."""
    azr = np.radians(az)
    elr = np.radians(el)
    x = np.cos(elr) * np.cos(azr)
    y = np.cos(elr) * np.sin(azr)
    z = np.sin(elr)
    return np.stack([x, y, z], axis=0)

def enforce_visibility(src_az, src_el, geo_az, geo_el, limb_angle_deg=23.8):
    """
    Keep samples where angle(source_dir, Earth-center dir) >= limb_angle_deg.
    geo_az/el define the SC position direction (Earth-center direction).
    """
    s = unitvec_from_az_el(src_az, src_el)
    e = unitvec_from_az_el(geo_az, geo_el)
    dot = np.clip(np.sum(s * e, axis=0), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return ang >= limb_angle_deg  # boolean mask

def sample_coords(n_samples,
                  limb_emphasis=True,
                  rare_fraction=0.05,
                  seed=None,
                  enforce_vis=True,
                  limb_angle_deg=23.8):
    """
    Sample source (src_az, src_el) and Earth geometry (geo_az, geo_el) in SC coords.
    - src_az ~ U[0,360), src_el ~ uniform on sphere
    - geo_az ~ U[0,360), geo_el biased to negative (Earth "below") with optional limb emphasis
    - If enforce_vis, filter so source is above limb by limb_angle_deg
    """
    rng = np.random.default_rng(seed)
    sa_all, se_all, ga_all, ge_all = [], [], [], []
    need = n_samples
    batch = max(4 * n_samples, 1000)
    while need > 0:
        sa = rng.uniform(0, 360, batch)
        u = rng.uniform(-1, 1, batch)          # uniform on sphere
        se = np.degrees(np.arcsin(u))
        ga = rng.uniform(0, 360, batch)

        ge = np.empty(batch)
        n_main = int(batch * (1 - rare_fraction))
        ge[:n_main] = rng.uniform(-90, -40, n_main)   # mostly "down"
        ge[n_main:] = rng.uniform(-90, 90, batch - n_main)
        if limb_emphasis and n_main > 0:
            n_limb = max(1, n_main // 5)
            idx = rng.choice(n_main, n_limb, replace=False)
            ge[idx] = rng.uniform(-65, -45, n_limb)   # emphasize limb region

        if enforce_vis:
            m = enforce_visibility(sa, se, ga, ge, limb_angle_deg)
        else:
            m = np.ones(batch, dtype=bool)

        if m.any():
            take = min(need, int(m.sum()))
            keep_idx = np.flatnonzero(m)[:take]
            sa_all.append(sa[keep_idx]); se_all.append(se[keep_idx])
            ga_all.append(ga[keep_idx]); ge_all.append(ge[keep_idx])
            need -= take

    return (np.concatenate(sa_all), np.concatenate(se_all),
            np.concatenate(ga_all), np.concatenate(ge_all))

# --------------------------------------------------------------------------------
# PARALLEL WORKER
# --------------------------------------------------------------------------------

# Per-process cache of DRM generators to avoid reloading DBs
_GEN_CACHE = {}

def process_sample(det_name, src_az, src_el, geo_az, geo_el):
    """Generate one sample: (features, DRM_flat, det_id, det_name)."""
    gen = _GEN_CACHE.get(det_name)
    if gen is None:
        gen = DRMGenMock(det_name=det_name, occult=False)  # occult off: no zero DRMs
        _GEN_CACHE[det_name] = gen

    drm_matrix = gen.recompute(src_az, src_el, geo_az, geo_el)

    # Input features: src_az, src_el, geo_az, geo_el, detector normal (nx, ny, nz)
    nx, ny, nz = det_orient_vec[det_name]
    features = np.array([src_az, src_el, geo_az, geo_el, nx, ny, nz], dtype=np.float32)

    det_id = np.int16(DET_NAME_TO_ID[det_name])

    return features, drm_matrix.flatten().astype(np.float32), det_id, det_name

# --------------------------------------------------------------------------------
# DATASET BUILD
# --------------------------------------------------------------------------------

def build_dataset(det_list, n_samples_per_det, output_file, seed_global=42):
    """Build HDF5 dataset for given detectors."""
    rng = np.random.default_rng(seed_global)
    all_features = []
    all_drms = []
    all_det_ids = []
    all_det_names = []

    for det_name in det_list:
        print(f"Generating for {det_name}...")

        src_az, src_el, geo_az, geo_el = sample_coords(
            n_samples_per_det,
            limb_emphasis=True,
            rare_fraction=0.05,
            seed=rng.integers(0, 1_000_000_000),
            enforce_vis=ENFORCE_VISIBILITY,
            limb_angle_deg=LIMB_ANGLE_DEG,
        )

        with Pool(processes=cpu_count()) as pool:
            worker = partial(process_sample, det_name)
            iterable = zip(src_az, src_el, geo_az, geo_el)
            results = list(tqdm(pool.starmap(worker, iterable), total=n_samples_per_det))

        # Unpack results
        if results:
            features_det, drms_det, ids_det, names_det = zip(*results)
            all_features.extend(features_det)
            all_drms.extend(drms_det)
            all_det_ids.extend(ids_det)
            all_det_names.extend(names_det)

    # Convert to arrays
    all_features = np.stack(all_features).astype(np.float32)
    all_drms = np.stack(all_drms).astype(np.float32)
    all_det_ids = np.asarray(all_det_ids, dtype=np.int16)
    all_det_names = np.asarray(all_det_names, dtype=object)

    # Save to HDF5
    with h5py.File(output_file, "w") as f:
        f.create_dataset("inputs", data=all_features)
        f.create_dataset("drms", data=all_drms)
        f.create_dataset("det_id", data=all_det_ids)
        # Store names as variable-length UTF-8 strings
        str_dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("det_name", data=all_det_names.astype(str_dt))

    print(f"Saved {output_file}")
    print(f"Features shape: {all_features.shape}")
    print(f"DRMs shape: {all_drms.shape}")
    print(f"det_id shape: {all_det_ids.shape}, det_name shape: {all_det_names.shape}")

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example for NaI model:
    NAI_DETECTORS = [f"NAI_{i:02d}" for i in range(12)]
    build_dataset(NAI_DETECTORS, N_SAMPLES_PER_DET, OUTPUT_FILE, SEED_GLOBAL)

    # Example for BGO model (uncomment to use):
    # BGO_DETECTORS = ["BGO_00", "BGO_01"]
    # build_dataset(BGO_DETECTORS, N_SAMPLES_PER_DET, "bgos_training.h5", SEED_GLOBAL)