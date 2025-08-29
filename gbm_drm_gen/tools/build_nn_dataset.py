#!/usr/bin/env python3
"""
High-throughput NN DRM training dataset generator.

- Single multiprocessing Pool
- Per-process DRM generator cache built in a Pool initializer (works with 'spawn')
- Streams results to HDF5 in batches (append), avoiding large RAM spikes
- Samples only visible geometries (source above Earth limb by a fixed angle)
- Augments features with cos_to_nadir and cos_offaxis
- Supports sampling geometry from a PositionInterpolator (trigdat/poshist)
- Robust time-range clamping for interpolator
- Supports multiple trigdat files and multiple poshist files (streaming per file)
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
LIMB_ANGLE_DEG = 66.8  # angle between source and nadir to clear Earth limb

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
    return np.stack([x, y, z], axis=0)  # shape (3, ...)

def enforce_visibility(src_az, src_el, geo_az, geo_el, limb_angle_deg=LIMB_ANGLE_DEG):
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
                  limb_angle_deg=LIMB_ANGLE_DEG):
    rng = np.random.default_rng(seed)
    sa_all, se_all, ga_all, ge_all = [], [], [], []
    need = n_samples
    batch = max(4 * n_samples, 1000)
    total = 0
    accepted = 0

    while need > 0:
        sa = rng.uniform(0, 360, batch)
        u = rng.uniform(-1, 1, batch)
        se = np.degrees(np.arcsin(u))  # uniform on sphere
        ga = rng.uniform(0, 360, batch)

        ge = np.empty(batch)
        n_main = int(batch * (1 - rare_fraction))
        ge[:n_main] = rng.uniform(-90, -40, n_main)
        ge[n_main:] = rng.uniform(-90, 90, batch - n_main)
        if limb_emphasis and n_main > 0:
            n_limb = max(1, n_main // 5)
            idx = rng.choice(n_main, n_limb, replace=False)
            ge[idx] = rng.uniform(-65, -45, n_limb)

        if enforce_vis:
            m = enforce_visibility(sa, se, ga, ge, limb_angle_deg)
        else:
            m = np.ones(batch, dtype=bool)

        total += batch
        accepted += int(m.sum())

        if m.any():
            take = min(need, int(m.sum()))
            keep = np.flatnonzero(m)[:take]
            sa_all.append(sa[keep]); se_all.append(se[keep])
            ga_all.append(ga[keep]); ge_all.append(ge[keep])
            need -= take

    print(f"[sample_coords] visibility acceptance: {accepted}/{total} = {accepted/total:.3f} at limb={limb_angle_deg} deg")

    return (np.concatenate(sa_all), np.concatenate(se_all),
            np.concatenate(ga_all), np.concatenate(ge_all))

# Helpers to compute geo and src angles from quaternion and sc_pos (mirror DRMGen math)
def _sc_axes_from_quat(q):
    q0, q1, q2, q3 = q  # scalar-first convention
    scx = np.array([q0*q0 - q1*q1 - q2*q2 + q3*q3,
                    2.0*(q0*q1 + q3*q2),
                    2.0*(q0*q2 - q3*q1)], dtype=np.float64)
    scy = np.array([2.0*(q0*q1 - q3*q2),
                    -q0*q0 + q1*q1 - q2*q2 + q3*q3,
                    2.0*(q1*q2 + q3*q0)], dtype=np.float64)
    scz = np.array([2.0*(q0*q2 + q3*q1),
                    2.0*(q1*q2 - q3*q0),
                    -q0*q0 - q1*q1 + q2*q2 + q3*q3], dtype=np.float64)
    return scx, scy, scz

def _geo_az_el_from(q, sc_pos):
    scx, scy, scz = _sc_axes_from_quat(q)
    geodir = np.array([-np.dot(scx, sc_pos), -np.dot(scy, sc_pos), -np.dot(scz, sc_pos)], dtype=np.float64)
    geodir /= np.linalg.norm(geodir)
    geo_az = np.arctan2(geodir[1], geodir[0])
    if geo_az < 0.0: geo_az += 2*np.pi
    while geo_az > 2*np.pi: geo_az -= 2*np.pi
    geo_el = np.arctan2(np.sqrt(geodir[0]**2 + geodir[1]**2), geodir[2])
    geo_el_deg = 90.0 - np.degrees(geo_el)
    geo_az_deg = np.degrees(geo_az)
    return geo_az_deg, geo_el_deg

def _src_az_el_from(q, ra_deg, dec_deg):
    ra = np.radians(ra_deg); dec = np.radians(dec_deg)
    v = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], dtype=np.float64)
    scx, scy, scz = _sc_axes_from_quat(q)
    v_sc = np.array([np.dot(scx, v), np.dot(scy, v), np.dot(scz, v)], dtype=np.float64)
    el = np.arccos(np.clip(v_sc[2], -1.0, 1.0))
    az = np.arctan2(v_sc[1], v_sc[0])
    if az < 0.0: az += 2*np.pi
    el_deg = 90.0 - np.degrees(el)
    az_deg = np.degrees(az)
    return az_deg, el_deg

def _get_time_bounds(interp):
    """
    Find the valid time range by intersecting the domains of underlying interpolators.
    Tries multiple attribute names to be robust to gbmgeometry versions.
    """
    candidates = []
    for attr in ("_quaternion_t", "_sc_position_t", "_scpos_t", "_sc_pos_t"):
        obj = getattr(interp, attr, None)
        if obj is None:
            continue
        try:
            x = np.asarray(obj.x, dtype=float)
            if x.size > 0:
                candidates.append((float(x.min()), float(x.max())))
        except Exception:
            pass
    if candidates:
        tmin = max(c[0] for c in candidates)
        tmax = min(c[1] for c in candidates)
        return tmin, tmax
    return None

def sample_coords_from_interpolator(n_samples,
                                    trigdat_file=None,
                                    poshist_file=None,
                                    T0=None,
                                    t_start=-200.0,
                                    t_stop=200.0,
                                    seed=None,
                                    enforce_vis=True,
                                    limb_angle_deg=LIMB_ANGLE_DEG):
    """
    Sample (src_az, src_el, geo_az, geo_el) using a PositionInterpolator.
    Robustly clamps the requested time window to the interpolator range.
    For poshist: if t_start/t_stop look like small offsets (<= 7 days),
    interpret them as offsets from the file start time.
    """
    assert (trigdat_file is not None) ^ (poshist_file is not None), "Provide exactly one of trigdat_file or poshist_file"
    import gbmgeometry
    if trigdat_file is not None:
        try:
            interp = gbmgeometry.PositionInterpolator.from_trigdat(trigdat_file=trigdat_file)
        except Exception:
            interp = gbmgeometry.PositionInterpolator.from_trigdat_hdf5(trigdat_file=trigdat_file)
    else:
        try:
            interp = gbmgeometry.PositionInterpolator.from_poshist(poshist_file=poshist_file, T0=T0)
        except Exception:
            interp = gbmgeometry.PositionInterpolator.from_poshist_hdf5(poshist_file=poshist_file, T0=T0)

    tb = _get_time_bounds(interp)
    if tb is not None:
        tmin, tmax = tb
        print(f"[sample_coords_from_interpolator] interpolator time range: [{tmin:.3f}, {tmax:.3f}]")
        # Interpret small numbers as offsets for poshist
        original_start, original_stop = float(t_start), float(t_stop)
        if poshist_file is not None:
            # If requested window is "small" (<= 7 days) we treat it as relative offsets from tmin
            if (original_stop - original_start) > 0 and original_stop <= 7*86400.0 + 1.0:
                t_start = tmin + max(0.0, original_start)
                t_stop  = tmin + max(0.0, original_stop)
                print(f"[sample_coords_from_interpolator] interpreting requested [{original_start:.3f},{original_stop:.3f}] as offsets from file start {tmin:.3f} -> [{t_start:.3f},{t_stop:.3f}]")

        # Clamp to available range
        eps = 1e-6
        t_start = max(t_start, tmin + eps)
        t_stop  = min(t_stop,  tmax  - eps)
        if not (t_start < t_stop):
            raise ValueError(f"Requested time window outside interpolator range: requested [{t_start:.3f},{t_stop:.3f}] vs available [{tmin:.3f},{tmax:.3f}]")
        if (t_start > tmin + eps) or (t_stop < tmax - eps):
            print(f"[sample_coords_from_interpolator] clamped time window to [{t_start:.3f}, {t_stop:.3f}]")
    else:
        print("[sample_coords_from_interpolator] WARNING: could not read time bounds; using requested window")

    rng = np.random.default_rng(seed)
    sa_all, se_all, ga_all, ge_all = [], [], [], []
    need = n_samples
    batch = max(4 * n_samples, 1000)
    total = 0
    accepted = 0

    while need > 0:
        t = rng.uniform(t_start, t_stop, batch)
        t = np.clip(t, t_start, t_stop)

        sa = np.empty(batch, dtype=np.float64)
        se = np.empty(batch, dtype=np.float64)
        ga = np.empty(batch, dtype=np.float64)
        ge = np.empty(batch, dtype=np.float64)

        for i in range(batch):
            ti = float(t[i])
            try:
                q = interp.quaternion(ti)
            except ValueError:
                ti = min(max(ti, t_start), t_stop)
                q = interp.quaternion(ti)
            sc_pos = np.array(interp.sc_pos(ti), dtype=np.float64)
            ga[i], ge[i] = _geo_az_el_from(q, sc_pos)

            ra = rng.uniform(0, 360)
            u = rng.uniform(-1, 1)
            dec = np.degrees(np.arcsin(u))
            sa[i], se[i] = _src_az_el_from(q, ra, dec)

        if enforce_vis:
            m = enforce_visibility(sa, se, ga, ge, limb_angle_deg)
        else:
            m = np.ones(batch, dtype=bool)

        total += batch
        accepted += int(m.sum())

        if m.any():
            take = min(need, int(m.sum()))
            keep = np.flatnonzero(m)[:take]
            sa_all.append(sa[keep]); se_all.append(se[keep])
            ga_all.append(ga[keep]); ge_all.append(ge[keep])
            need -= take

    print(f"[sample_coords_from_interpolator] visibility acceptance: {accepted}/{total} = {accepted/total:.3f} at limb={limb_angle_deg} deg; sampled t∈[{t_start:.3f},{t_stop:.3f}]")
    return (np.concatenate(sa_all), np.concatenate(se_all),
            np.concatenate(ga_all), np.concatenate(ge_all))

# Features
def make_features(det_name, src_az, src_el, geo_az, geo_el):
    s = unitvec_from_az_el(src_az, src_el).astype(np.float32)
    g = unitvec_from_az_el(geo_az, geo_el).astype(np.float32)
    nx, ny, nz = det_orient_vec[det_name]
    n = np.array([nx, ny, nz], dtype=np.float32)
    cos_to_nadir = float(np.clip(np.dot(s, g), -1.0, 1.0))
    cos_offaxis = float(np.clip(np.dot(s, n), -1.0, 1.0))
    return np.array([src_az, src_el, geo_az, geo_el, nx, ny, nz,
                     cos_to_nadir, cos_offaxis], dtype=np.float32)

# Per-process generator cache
_GEN_CACHE = {}

def _worker_init(det_list, mat_type=2, occult=False):
    global _GEN_CACHE
    from gbm_drm_gen.drmgen_mock import DRMGenMock
    _GEN_CACHE = {det: DRMGenMock(det_name=det, mat_type=mat_type, occult=occult) for det in det_list}

def process_sample_cached(args):
    det_name, src_az, src_el, geo_az, geo_el = args
    gen = _GEN_CACHE[det_name]
    drm_matrix = gen.recompute(src_az, src_el, geo_az, geo_el)
    features = make_features(det_name, src_az, src_el, geo_az, geo_el)
    det_id = np.int16(DET_NAME_TO_ID[det_name])
    return features, drm_matrix.reshape(-1).astype(np.float32), det_id, det_name

def _prepare_h5(output_file, mat_type, occult, feat_len, flat_len):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    f = h5py.File(output_file, "w")
    ds_X  = f.create_dataset("inputs",    shape=(0, feat_len), maxshape=(None, feat_len), dtype="f4", chunks=True)
    ds_Y  = f.create_dataset("drms",      shape=(0, flat_len), maxshape=(None, flat_len), dtype="f4", chunks=True)
    ds_id = f.create_dataset("det_id",    shape=(0,),         maxshape=(None,),          dtype="i2", chunks=True)
    ds_nm = f.create_dataset("det_name",  shape=(0,),         maxshape=(None,),          dtype=h5py.string_dtype("utf-8"), chunks=True)
    f.attrs["mat_type"] = mat_type
    f.attrs["occult"]   = bool(occult)
    return f, ds_X, ds_Y, ds_id, ds_nm

def _append_batch(h5_handles, batch):
    if not batch:
        return
    ds_X, ds_Y, ds_id, ds_nm = h5_handles
    n_old = ds_X.shape[0]
    n_new = n_old + len(batch)
    ds_X.resize((n_new, ds_X.shape[1])); ds_Y.resize((n_new, ds_Y.shape[1]))
    ds_id.resize((n_new,)); ds_nm.resize((n_new,))
    for j, (features, drm_flat, det_id_j, det_name_j) in enumerate(batch):
        ds_X[n_old + j]  = features
        ds_Y[n_old + j]  = drm_flat
        ds_id[n_old + j] = det_id_j
        ds_nm[n_old + j] = det_name_j

def build_dataset(det_list,
                  n_samples_per_det,
                  output_file,
                  seed_global=42,
                  processes=None,
                  chunksize=256,
                  bufsize=1024,
                  mat_type=2,
                  occult=False,
                  trigdat_file=None,
                  poshist_file=None,
                  T0=None,
                  t_start=-200.0,
                  t_stop=200.0,
                  trigdat_files=None,
                  poshist_files=None):
    """
    Build HDF5 dataset for given detectors.

    - If trigdat_files is provided (list of paths), samples are drawn from each file in turn (streaming).
    - If poshist_files is provided (list of paths), samples are drawn from each file in turn (streaming).
    - Else if trigdat_file or poshist_file is provided, samples are drawn from that interpolator.
    - Else uses synthetic sampler.
    - n_samples_per_det applies per detector per file (if multiple files).
    """
    rng = np.random.default_rng(seed_global)
    processes = processes or cpu_count()

    # Probe to determine shapes
    from gbm_drm_gen.drmgen_mock import DRMGenMock
    tmp_gen = DRMGenMock(det_name=det_list[0], mat_type=mat_type, occult=occult)
    _ = tmp_gen.recompute(0.0, 0.0, 0.0, 90.0)
    y0 = tmp_gen.matrix.reshape(-1).astype(np.float32)
    flat_len = y0.size
    feat_len = make_features(det_list[0], 0.0, 0.0, 0.0, 90.0).size

    # Prepare HDF5
    f, ds_X, ds_Y, ds_id, ds_nm = _prepare_h5(output_file, mat_type, occult, feat_len, flat_len)
    h5_handles = (ds_X, ds_Y, ds_id, ds_nm)

    print(f"Launching Pool with {processes} processes; writing to {output_file}")
    ctx = get_context("spawn")
    with ctx.Pool(processes=processes, initializer=_worker_init, initargs=(det_list, mat_type, occult)) as pool:
        def process_tasks(tasks):
            buffer = []
            for res in tqdm(pool.imap_unordered(process_sample_cached, tasks, chunksize=chunksize),
                            total=len(tasks)):
                buffer.append(res)
                if len(buffer) >= bufsize:
                    _append_batch(h5_handles, buffer)
                    buffer.clear()
            if buffer:
                _append_batch(h5_handles, buffer)

        if trigdat_files:
            for tf in sorted(trigdat_files):
                print(f"Sampling from trigdat: {tf}")
                tasks = []
                for det_name in det_list:
                    sa, se, ga, ge = sample_coords_from_interpolator(
                        n_samples_per_det,
                        trigdat_file=tf,
                        poshist_file=None,
                        T0=T0,
                        t_start=t_start,
                        t_stop=t_stop,
                        seed=rng.integers(0, 1_000_000_000),
                        enforce_vis=ENFORCE_VISIBILITY,
                        limb_angle_deg=LIMB_ANGLE_DEG,
                    )
                    tasks.extend((det_name, a, b, c, d) for a, b, c, d in zip(sa, se, ga, ge))
                process_tasks(tasks)

        elif poshist_files:
            for pf in sorted(poshist_files):
                print(f"Sampling from poshist: {pf}")
                tasks = []
                for det_name in det_list:
                    sa, se, ga, ge = sample_coords_from_interpolator(
                        n_samples_per_det,
                        trigdat_file=None,
                        poshist_file=pf,
                        T0=T0,
                        t_start=t_start,
                        t_stop=t_stop,
                        seed=rng.integers(0, 1_000_000_000),
                        enforce_vis=ENFORCE_VISIBILITY,
                        limb_angle_deg=LIMB_ANGLE_DEG,
                    )
                    tasks.extend((det_name, a, b, c, d) for a, b, c, d in zip(sa, se, ga, ge))
                process_tasks(tasks)

        elif (trigdat_file is not None) or (poshist_file is not None):
            tasks = []
            for det_name in det_list:
                sa, se, ga, ge = sample_coords_from_interpolator(
                    n_samples_per_det,
                    trigdat_file=trigdat_file,
                    poshist_file=poshist_file,
                    T0=T0,
                    t_start=t_start,
                    t_stop=t_stop,
                    seed=rng.integers(0, 1_000_000_000),
                    enforce_vis=ENFORCE_VISIBILITY,
                    limb_angle_deg=LIMB_ANGLE_DEG,
                )
                tasks.extend((det_name, a, b, c, d) for a, b, c, d in zip(sa, se, ga, ge))
            process_tasks(tasks)

        else:
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
            process_tasks(tasks)

    f.close()
    print(f"Saved {output_file}")

if __name__ == "__main__":
    # small example
    N_SAMPLES_PER_DET = 10
    OUTPUT_FILE = "nais_training.h5"
    NAI_DETECTORS = [f"NAI_{i:02d}" for i in range(12)]
    build_dataset(NAI_DETECTORS, N_SAMPLES_PER_DET, OUTPUT_FILE, seed_global=42,
                  processes=min(16, cpu_count()), chunksize=256, bufsize=1024, mat_type=2, occult=False)
