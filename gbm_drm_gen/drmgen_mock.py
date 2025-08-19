import numpy as np
from gbm_drm_gen import DRMGen
from gbm_drm_gen.basersp_numba import get_database
from gbm_drm_gen.input_edges import tte_edges
from gbm_drm_gen.output_edges import nai_cspec_edges, bgo_00_cspec_edges, bgo_01_cspec_edges

# Detector lookups
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
    "b0": "BGO_00", "b1": "BGO_01"
}


class DRMGenMock(DRMGen):
    def __init__(self, src_az, src_el, geo_az, geo_el,
                 det_name, mat_type=0, occult=False, time=0.0):
        """
        Mock DRM generator: Uses standard TTE input edges but fixed CSPEC output edges.
        This ensures physics kernels remain valid while output DRMs have consistent binning.
        """
        self.src_az = np.array(src_az, dtype=np.float32)
        self.src_el = np.array(src_el, dtype=np.float32)
        self.geo_az = np.array(geo_az, dtype=np.float32)
        self.geo_el = np.array(geo_el, dtype=np.float32)
        self.mat_type = mat_type
        self.occult = occult
        self.time = time

        # Resolve detector name
        if det_name not in DET_NAME_LOOKUP:
            if det_name not in DET_NAME_LOOKUP2:
                raise RuntimeError(f"{det_name} is not a valid detector name")
            det_name = DET_NAME_LOOKUP2[det_name]
        self.det_name = det_name
        self.det_number = DET_NAME_LOOKUP[det_name]

        # --- Select input edges from TTE table (unchanged) ---
        if self.det_number > 11:  # BGO
            in_edge = tte_edges["bgo"]
        else:
            in_edge = tte_edges["nai"]

        # --- Select fixed output edges from our table ---
        if det_name == "BGO_00":
            out_edge = bgo_00_cspec_edges
        elif det_name == "BGO_01":
            out_edge = bgo_01_cspec_edges
        else:
            out_edge = nai_cspec_edges

        self.ebin_edge_in = in_edge
        self.ebin_edge_out = out_edge

        # Initialise base DRM generator
        super().__init__(
            position_interpolator=None,
            det_number=self.det_number,
            ebin_edge_in=self.ebin_edge_in,
            mat_type=self.mat_type,
            ebin_edge_out=self.ebin_edge_out,
            occult=self.occult,
            time=self.time
        )

        # Skip trigdat/Cspec auto edge matching
        self._trigdat = False

        # Attach calibration database for this detector
        self._database_nb = get_database(det_name)

    def set_location_sat_coordinates(self):
        """
        Override spacecraft/detector pointing with custom values.
        """
        self._src_az = self.src_az
        self._src_el = self.src_el
        self._geo_az = self.geo_az
        self._geo_el = self.geo_el