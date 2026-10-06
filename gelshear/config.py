import json
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class Config:
    # --- frame source ---
    source: str = "0"            # camera index ("0", "1", ...) or path to a video file
    width: int = 640             # resize frames to this width (aspect kept); 0 = keep native
    crop_frac: float = 0.0       # crop this fraction off every border first (dark rims etc.)
    ref_frames: int = 10         # frames averaged (median) for the no-contact reference

    # --- marker detection / tracking ---
    blackhat_ksize: int = 0      # morphological kernel; 0 = auto (~width/35). Must exceed marker diameter
    min_area_frac: float = 0.3   # keep blobs with area in [min, max] * median reference marker area
    max_area_frac: float = 3.0
    match_gate_frac: float = 0.45  # max frame-to-frame marker jump, as a fraction of marker pitch

    # --- contact detection (from the photometric / shading channel) ---
    contact_blur_frac: float = 2.5   # median-blur kernel as multiple of marker diameter (erases dots)
    contact_thresh: float = 12.0     # threshold on |blurred frame - blurred reference|, intensity units
    min_contact_area_px: int = 80
    influence_margin_frac: float = 2.0  # how far a contact's deformation reaches beyond its edge, in marker pitches
    unreliable_unpaired_frac: float = 0.6  # above this share of unpaired dots, a contact's estimate is 'unreliable'

    # --- drift handling (lighting / exposure / gel creep over time) ---
    auto_rebaseline: bool = True   # slowly update the no-contact baseline while the gel is idle
    idle_disp_px: float = 0.5      # idle if 95th-percentile marker displacement is below this
    idle_frames: int = 15          # consecutive idle frames before the baseline starts adapting
    rebaseline_rate: float = 0.05  # blend rate per idle frame (shading baseline and shear zero)

    # --- field interpolation / integration ---
    rbf_smoothing: float = 1.0
    grid_step_px: int = 8

    # --- optional physical scale (all three needed for newtons) ---
    marker_pitch_mm: Optional[float] = None
    gel_thickness_mm: Optional[float] = None
    youngs_modulus_mpa: Optional[float] = None

    # --- perspective leak model, fitted from pure-normal presses (key 'c' then 'f') ---
    perspective_alpha: float = 0.0

    # --- visualisation ---
    arrow_scale: float = 4.0       # per-marker displacement arrows
    net_arrow_scale: float = 3.0   # net-shear arrow length per px of summed marker displacement
    contact_arrow_scale: float = 3.0
    history_len: int = 200         # frames shown in the bottom time-series strip

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
