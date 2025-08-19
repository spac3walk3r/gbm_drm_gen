#!/usr/bin/env python3
"""
Parallel NN DRM training dataset generator.

Generates (input_features, DRMs) for a list of GBM detectors (either all NaIs or BGOs),
sampling realistic spacecraft geometry and embedding detector orientation vectors.
"""

import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from gbm_drm_gen.drmgen_mock import DRMGenMock

# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------

N_SAMPLES_PER_DET = 2000  # per detector
OUTPUT_FILE = "nais_training.h5"   # change to bgos_training.h5 for BGOs
SEED_GLOBAL = 42

# Detector orientation (az, zen) degrees
det_orient_deg = {
    "NAI_00": (45.9, 20.6),
    "NAI_01": (45.1, 45.3),
    "NAI_02": (58.4, 90.2),
    "NAI_03": (314.9, 45.2),
    "NAI_04": (303.2, 90.3),
    "NAI_05": (3.4, 89.8),
    "NAI_06": (224.9, 20.4),
    "NAI_07": (224.6, 46.2),
    "NAI_08": (236.6, 90.0),
    "NAI_09": (135.2, 45.6),
    "NAI_10": (123.7, 90.4),
    "NAI_11": (183.7, 90.3),
    "BGO_00": (90.0, 90.0),     # approx, replace with rsp_moddb when possible
    "BGO_01": (270.0, 90.0),    # approx
}

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
# FUNCTIONS
# --------------------------------------------------------------------------------

def sample_coords(n_samples, limb_emphasis=True, rare_fraction=0.05, seed=None):
    """Sample source and Earth geometry in SC coords."""
    rng = np.random.default_rng(seed)

    src_az = rng.uniform(0, 360, n_samples)
    u = rng.uniform(-1, 1, n_samples)
    src_el = np.degrees(np.arcsin(u))

    geo_az = rng.uniform(0, 360, n_samples)

    geo_el = np.empty(n_samples)
    n_main = int(n_samples * (1 - rare_fraction))
    geo_el[:n_main] = rng.uniform(-90, -40, n_main)
    geo_el[n_main:] = rng.uniform(-90, 90, n_samples - n_main)

    if limb_emphasis:
        n_limb = n_main // 5
        idx = rng.choice(n_main, n_limb, replace=False)
        geo_el[idx] = rng.uniform(-65, -45, n_limb)

    return src_az, src_el, geo_az, geo_el

def process_sample(det_name, src_az, src_el, geo_az, geo_el):
    """Generate one sample: (features, DRM_flat)."""
    drm_obj = DRMGenMock(
        src_az=src_az,
        src_el=src_el,
        geo_az=geo_az,
        geo_el=geo_el,
        det_name=det_name
    )
    drm_matrix = drm_obj.get_drm()

    # Input features: src_az, src_el, geo_az, geo_el, det_orient_vector (nx, ny, nz) (detector’s normal vector in SC coords)
    nx, ny, nz = det_orient_vec[det_name]
    features = np.array([src_az, src_el, geo_az, geo_el, nx, ny, nz], dtype=np.float32)

    return features, drm_matrix.flatten().astype(np.float32)

def build_dataset(det_list, n_samples_per_det, output_file, seed_global=42):
    """Build HDF5 dataset for given detectors."""
    rng = np.random.default_rng(seed_global)
    all_features = []
    all_drms = []

    for det_name in det_list:
        print(f"Generating for {det_name}...")
        src_az, src_el, geo_az, geo_el = sample_coords(
            n_samples_per_det,
            limb_emphasis=True,
            rare_fraction=0.05,
            seed=rng.integers(0, 1e9)
        )

        # Parallel map over samples for this detector
        with Pool(processes=cpu_count()) as pool:
            worker = partial(process_sample, det_name)
            results = list(tqdm(pool.imap(worker, src_az, src_el, geo_az, geo_el),
                                total=n_samples_per_det))

        features_det, drms_det = zip(*results)
        all_features.extend(features_det)
        all_drms.extend(drms_det)

    all_features = np.stack(all_features)
    all_drms = np.stack(all_drms)

    # Save to HDF5
    with h5py.File(output_file, "w") as f:
        f.create_dataset("inputs", data=all_features)
        f.create_dataset("drms", data=all_drms)

    print(f"Saved {output_file}")
    print(f"Features shape: {all_features.shape}")
    print(f"DRMs shape: {all_drms.shape}")

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # EXAMPLE for NaI model:
    NAI_DETECTORS = [f"NAI_{i:02d}" for i in range(12)]
    build_dataset(NAI_DETECTORS, N_SAMPLES_PER_DET, "nais_training.h5")

    # EXAMPLE for BGO model:
    # BGO_DETECTORS = ["BGO_00", "BGO_01"]
    # build_dataset(BGO_DETECTORS, N_SAMPLES_PER_DET, "bgos_training.h5")