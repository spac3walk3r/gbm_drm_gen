import numpy as np
from .drmgen import DRMGen, lu
from gbm_drm_gen.input_edges import tte_edges
from gbm_drm_gen.output_edges import nai_cspec_edges, bgo_00_cspec_edges, bgo_01_cspec_edges
from gbm_drm_gen.basersp_numba import get_trigdat_precalc_database

DET_NAME_LOOKUP = {
    "NAI_00": 0, "NAI_01": 1, "NAI_02": 2, "NAI_03": 3,
    "NAI_04": 4, "NAI_05": 5, "NAI_06": 6, "NAI_07": 7,
    "NAI_08": 8, "NAI_09": 9, "NAI_10": 10, "NAI_11": 11,
    "BGO_00": 12, "BGO_01": 13,
}

DET_NAME_LOOKUP2 = {
    "n0": "NAI_00", "n1": "NAI_01", "n2": "NAI_02", "n3": "NAI_03",
    "n4": "NAI_04", "n5": "NAI_05", "n6": "NAI_06", "n7": "NAI_07",
    "n8": "NAI_08", "n9": "NAI_09", "na": "NAI_10", "nb": "NAI_11",
    "b0": "BGO_00", "b1": "BGO_01",
}

def _unit_vec_from_az_el(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(el)*np.cos(az),
                     np.cos(el)*np.sin(az),
                     np.sin(el)], dtype=np.float64)

class _StaticPositionInterpolator:
    # Minimal stand-in: identity quaternion, SC position sets geo_az/el via direction only
    def __init__(self, geo_az, geo_el):
        self._sc_pos = -_unit_vec_from_az_el(geo_az, geo_el)
    def quaternion(self, t): return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    def sc_pos(self, t):     return self._sc_pos
    def met(self, t):        return float(t)

class DRMGenMock(DRMGen):
    def __init__(self,
                 det_name,
                 mat_type=2,
                 occult=False,
                 time=0.0,
                 ebin_edge_in=None,
                 ebin_edge_out=None):
        # Resolve detector name
        if det_name not in DET_NAME_LOOKUP:
            if det_name not in DET_NAME_LOOKUP2:
                raise RuntimeError(f"{det_name} is not a valid detector name")
            det_name = DET_NAME_LOOKUP2[det_name]
        det_number = DET_NAME_LOOKUP[det_name]

        # Input/output edges
        if ebin_edge_in is None:
            in_edge = tte_edges["bgo"] if det_number > 11 else tte_edges["nai"]
        else:
            in_edge = np.asarray(ebin_edge_in, dtype=np.float32)

        if ebin_edge_out is None:
            if det_name == "BGO_00":
                out_edge = bgo_00_cspec_edges
            elif det_name == "BGO_01":
                out_edge = bgo_01_cspec_edges
            else:
                out_edge = nai_cspec_edges
        else:
            out_edge = np.asarray(ebin_edge_out, dtype=np.float32)

        # Neutral geometry; we’ll update per-sample in recompute()
        pi = _StaticPositionInterpolator(geo_az=0.0, geo_el=90.0)

        super().__init__(
            position_interpolator=pi,
            det_number=det_number,
            ebin_edge_in=in_edge,
            mat_type=mat_type,
            ebin_edge_out=out_edge,
            occult=occult,
            time=time,
        )

        # Disable trigdat precomputed responses (edge mismatch in general)
        self._trigdat = False
        self._trigdat_mask = None
        self._database_precalc_trigdat = get_trigdat_precalc_database(lu[det_number], [0])

    def recompute(self, src_az, src_el, geo_az, geo_el):
        # Update SC position direction to encode Earth geometry
        self.position_interpolator._sc_pos = -_unit_vec_from_az_el(geo_az, geo_el)
        self._sc_quaternions_updater()
        self._compute_spacecraft_coordinates()
        # Build DRM for the requested source direction (SC frame)
        self.set_location_direct_sat_coord(float(src_az), float(src_el))
        return self.matrix

    # convenience
    def get_drm(self):
        return self.matrix
