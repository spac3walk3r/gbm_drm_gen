import numpy as np
import astropy.units as u
import gbmgeometry

from gbm_drm_gen.input_edges import tte_edges
from gbm_drm_gen.output_edges import nai_cspec_edges, bgo_00_cspec_edges, bgo_01_cspec_edges
from gbm_drm_gen.nn_utils.runtime import ModelRegistry
from gbm_drm_gen.utils.geometry import ang2cart, is_occulted
from .drmgen import det_name_lookup, det_name_lookup2  # reuse mappings
from responsum import InstrumentResponse

class DRMGenNN:
    """
    Fast DRM generator using the NN surrogate.

    - Uses CSPEC 128-bin output edges and TTE 140-bin input edges (must match training).
    - Accepts source location as RA/Dec or direct spacecraft coords (az/el).
    - Uses ModelRegistry to load models (NaI model with embedding, plus BGO_00 and BGO_01).
    - If occult=True, returns a zero matrix when direction is occulted by Earth.
    """

    def __init__(self,
                 position_interpolator: gbmgeometry.PositionInterpolator,
                 det_name: str,
                 time: float = 0.0,
                 models_dir: str = None,
                 device: str = "cpu",
                 occult: bool = False):

        # Normalize det name (long NAI_.. / BGO_.. or short n0/na/b0/b1)
        if det_name in det_name_lookup2:
            det_name = det_name_lookup2[det_name]
        if det_name not in det_name_lookup:
            raise RuntimeError(f"Unknown detector name: {det_name}")

        self._det_name = det_name
        self._det_number = det_name_lookup[det_name]

        # Energy grids (must match training): TTE 140 input, CSPEC 128 output
        self._in_edge = tte_edges["bgo"] if self._det_number > 11 else tte_edges["nai"]
        if det_name == "BGO_00":
            self._out_edge = bgo_00_cspec_edges
        elif det_name == "BGO_01":
            self._out_edge = bgo_01_cspec_edges
        else:
            self._out_edge = nai_cspec_edges

        self._nobins_in = len(self._in_edge) - 1
        self._nobins_out = len(self._out_edge) - 1

        # Occultation handling
        self._occult = bool(occult)
        self._occulted_DRM = np.zeros((self._nobins_in, self._nobins_out), dtype=np.float32)

        # Geometry source
        self._position_interpolator = position_interpolator
        self._time = float(time)
        self._sc_quaternions_updater()
        self._compute_spacecraft_coordinates()

        # NN registry (default to env if models_dir not given)
        if models_dir is None:
            models_dir = os.environ.get("GBM_DRM_GEN_NN_DIR", ".")
        self._nn = ModelRegistry(models_dir=models_dir, device=device)

        # Internal storage (n_in, n_out); matrix property returns .T like DRMGen
        self._drm = np.zeros((self._nobins_in, self._nobins_out), dtype=np.float32)

    # Public API parity with DRMGen
    @property
    def ebounds(self):
        return self._out_edge

    @property
    def monte_carlo_energies(self):
        return self._in_edge

    @property
    def matrix(self):
        # DRMGen.matrix returns [n_out, n_in], so we return the transpose of our [n_in, n_out]
        return self._drm.T

    def set_time(self, time):
        self._time = float(time)
        self._sc_quaternions_updater()
        self._compute_spacecraft_coordinates()

    def to_3ML_response(self, ra, dec, coverage_interval=None):
        self.set_location(ra, dec)
        return InstrumentResponse(self.matrix, self.ebounds, self.monte_carlo_energies,
                                  coverage_interval=coverage_interval)

    def to_3ML_response_direct_sat_coord(self, az, el):
        self.set_location_direct_sat_coord(az, el)
        return InstrumentResponse(self.matrix, self.ebounds, self.monte_carlo_energies)

    def set_location(self, ra, dec):
        az, el = self._get_coords(ra, dec)
        self.set_location_direct_sat_coord(az, el)

    def set_location_direct_sat_coord(self, az, el):
        # Optional Earth occultation check in spacecraft coords
        if self._occult and is_occulted(az, el, self._sc_pos):
            self._drm = self._occulted_DRM
            return

        # Predict DRM [n_out, n_in] on linear scale using the NN registry
        drm_out_by_in = self._nn.predict(self._det_name, float(az), float(el),
                                         float(self._geo_az), float(self._geo_el))
        # Store internally as [n_in, n_out], like DRMGen uses internally
        self._drm = drm_out_by_in.T.astype(np.float32, copy=False)

    # Internal geometry (mirror of DRMGen logic)
    def _sc_quaternions_updater(self):
        self._quaternions = self._position_interpolator.quaternion(self._time)
        self._sc_pos = self._position_interpolator.sc_pos(self._time)

    def _compute_spacecraft_coordinates(self):
        q = self._quaternions
        self._scx = np.zeros(3); self._scy = np.zeros(3); self._scz = np.zeros(3)

        self._scx[0] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        self._scx[1] = 2.0*(q[0]*q[1] + q[3]*q[2])
        self._scx[2] = 2.0*(q[0]*q[2] - q[3]*q[1])

        self._scy[0] = 2.0*(q[0]*q[1] - q[3]*q[2])
        self._scy[1] = -q[0]**2 + q[1]**2 - q[2]**2 + q[3]**2
        self._scy[2] = 2.0*(q[1]*q[2] + q[3]*q[0])

        self._scz[0] = 2.0*(q[0]*q[2] + q[3]*q[1])
        self._scz[1] = 2.0*(q[1]*q[2] - q[3]*q[0])
        self._scz[2] = -q[0]**2 - q[1]**2 + q[2]**2 + q[3]**2

        geodir = np.array([-self._scx.dot(self._sc_pos),
                           -self._scy.dot(self._sc_pos),
                           -self._scz.dot(self._sc_pos)], dtype=float)
        geodir /= max(1e-12, np.linalg.norm(geodir))
        geo_az = np.arctan2(geodir[1], geodir[0])
        if geo_az < 0.0:
            geo_az += 2*np.pi
        geo_el = np.arctan2(np.hypot(geodir[0], geodir[1]), geodir[2])

        self._geo_el = 90.0 - np.rad2deg(geo_el)
        self._geo_az = np.rad2deg(geo_az)

    def _get_coords(self, ra, dec):
        source_pos = ang2cart(ra, dec)
        source_pos_sc = np.array([self._scx.dot(source_pos),
                                  self._scy.dot(source_pos),
                                  self._scz.dot(source_pos)], dtype=float)
        el = 90.0 - np.rad2deg(np.arccos(np.clip(source_pos_sc[2], -1.0, 1.0)))
        az = np.rad2deg(np.arctan2(source_pos_sc[1], source_pos_sc[0]))
        if az < 0.0:
            az += 360.0
        return az, el

    @classmethod
    def from_128_bin_data_nn(cls, det_name, time=0.0, cspecfile=None, trigdat=None, poshist=None, T0=None,
                             occult=False, models_dir=None, device="cpu"):
        """
        Build a DRMGenNN from trigdat or poshist. cspecfile is ignored (edges are fixed to CSPEC).
        """
        if det_name in det_name_lookup2:
            det_name = det_name_lookup2[det_name]
        if det_name not in det_name_lookup:
            raise RuntimeError(f"{det_name} is not valid")

        if trigdat:
            try:
                pi = gbmgeometry.PositionInterpolator.from_trigdat(trigdat_file=trigdat)
            except Exception:
                pi = gbmgeometry.PositionInterpolator.from_trigdat_hdf5(trigdat_file=trigdat)
        elif poshist:
            try:
                pi = gbmgeometry.PositionInterpolator.from_poshist(poshist_file=poshist, T0=T0)
            except Exception:
                pi = gbmgeometry.PositionInterpolator.from_poshist_hdf5(poshist_file=poshist, T0=T0)
        else:
            raise RuntimeError("No trigdat or poshist file used!")

        return cls(position_interpolator=pi,
                   det_name=det_name,
                   time=time,
                   models_dir=models_dir,
                   device=device,
                   occult=occult)