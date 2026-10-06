"""Dot-pairing shear estimator.

1-D intuition: two dots placed symmetrically around a press move in opposite
directions (net zero); under shear they move together (net = shear).

2-D generalisation used here:
  * Pairs are point reflections through the contact centre: p  <->  2c - p.
    Radial (normal) and rotational (torsion) motion is odd under this
    reflection and cancels; shear is even and adds.
  * Normalisation for arbitrary dot layouts / off-centre contacts: the marker
    field is interpolated (thin-plate RBF) onto a uniform grid, so every point
    has an exact mirror partner and an equal area weight, no matter where c is
    relative to the physical dots.
  * Multiple contacts: summing the whole window pairs every contact with itself
    simultaneously (linear superposition), so no per-contact segmentation is
    needed for the net value.
  * Window edges: points whose mirror partner falls outside the window are
    unpaired. They are dropped from the sum ("only count dots that have a
    partner"), which removes the uncancelled normal-load excess.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from .contact import Contact, contact_diff, find_contacts
from .markers import MarkerTracker, auto_blackhat_ksize, detect_markers, marker_pitch


@dataclass
class ContactShear:
    center: np.ndarray
    radius: float
    R_sym: float            # radius of the symmetric disk that fits in the window
    edge: bool              # influence disk cut by the window edge
    unpaired_frac: float    # share of the influence disk dropped for lacking a partner
    shear_px: np.ndarray    # this contact's shear, summed-marker-px units


@dataclass
class Result:
    S_all: np.ndarray        # raw integral over the whole window (diagnostic)
    S_paired: np.ndarray     # integral over paired points only (main estimate, before corrections)
    S_corrected: np.ndarray  # minus perspective leak and tare
    net_px: np.ndarray       # S_corrected / pitch^2: equivalent summed marker displacement (px)
    force_N: Optional[np.ndarray]
    confidence: str          # "ok" | "edge-corrected" | "unreliable" | "no-markers"
    contacts: List[ContactShear]
    persp_feature: np.ndarray
    n_valid: int
    unpaired_mask: Optional[np.ndarray] = field(default=None, repr=False)
    diff: Optional[np.ndarray] = field(default=None, repr=False)   # shading-change map (debug view)
    idle: bool = False
    reason: str = ""                                                # why confidence is not "ok"


class Pipeline:
    def __init__(self, cfg, ref_frame):
        self.cfg = cfg
        self.set_reference(ref_frame)

    # ------------------------------------------------------------------ setup
    def set_reference(self, ref_frame):
        cfg = self.cfg
        self.ref_frame = ref_frame
        self.ksize = auto_blackhat_ksize(cfg, ref_frame.shape)
        pts, areas = detect_markers(ref_frame, self.ksize)
        if len(pts) < 10:
            raise RuntimeError(f"Only {len(pts)} markers found in reference; adjust blackhat_ksize / lighting")
        med = np.median(areas)
        keep = (areas >= cfg.min_area_frac * med) & (areas <= cfg.max_area_frac * med)
        pts = pts[keep]
        self.area_range = (cfg.min_area_frac * med, cfg.max_area_frac * med)
        self.marker_diam = 2 * np.sqrt(med / np.pi)
        self.pitch = marker_pitch(pts)
        self.tracker = MarkerTracker(pts, gate=cfg.match_gate_frac * self.pitch)

        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        self.bbox = (x0, y0, x1, y1)
        s = cfg.grid_step_px
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-9, s), np.arange(y0, y1 + 1e-9, s))
        self.grid = np.column_stack([gx.ravel(), gy.ravel()])
        self.cell_area = float(s * s)

        k = max(5, int(cfg.contact_blur_frac * self.marker_diam)) | 1
        self.contact_ksize = k
        self.ref_blur = cv2.medianBlur(ref_frame, k).astype(np.float32)
        self.idle_count = 0
        self.tare = np.zeros(2)
        h, w = ref_frame.shape[:2]
        self.optical_center = np.array([w / 2.0, h / 2.0])
        self.mm_per_px = (cfg.marker_pitch_mm / self.pitch) if cfg.marker_pitch_mm else None

    def reset_tracking(self):
        self.tracker.reset()

    def _inside(self, p):
        x0, y0, x1, y1 = self.bbox
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)

    def _to_force(self, S):
        cfg = self.cfg
        if self.mm_per_px and cfg.gel_thickness_mm and cfg.youngs_modulus_mpa:
            G = cfg.youngs_modulus_mpa / 3.0                 # near-incompressible silicone, MPa = N/mm^2
            return (G / cfg.gel_thickness_mm) * S * self.mm_per_px ** 3
        return None

    # ------------------------------------------------------------- per frame
    def process(self, frame):
        cfg = self.cfg
        det, areas = detect_markers(frame, self.ksize)
        lo, hi = self.area_range
        self.tracker.update(det[(areas >= lo) & (areas <= hi)])
        ref, disp, valid = self.tracker.ref, self.tracker.disp, self.tracker.valid

        diff, blurred = contact_diff(frame, self.ref_blur, self.contact_ksize)
        contacts, _ = find_contacts(diff, cfg.contact_thresh, cfg.min_contact_area_px, self.optical_center)
        feat = sum((c.persp for c in contacts), np.zeros(2))

        zero = np.zeros(2)
        if valid.sum() < 10:
            return Result(zero, zero, zero, zero, None, "no-markers", [], feat, int(valid.sum()),
                          diff=diff, reason="fewer than 10 markers tracked: press r to re-reference")

        # Normalisation step: dots -> uniform grid, so every point has a mirror partner and equal weight.
        rbf = RBFInterpolator(ref[valid], disp[valid], kernel="thin_plate_spline",
                              smoothing=cfg.rbf_smoothing, degree=1)
        U = rbf(self.grid)
        S_all = U.sum(axis=0) * self.cell_area

        # Pairing: drop points whose mirror through an edge contact's centre lies outside the window.
        margin = cfg.influence_margin_frac * self.pitch
        x0, y0, x1, y1 = self.bbox
        unpaired = np.zeros(len(self.grid), bool)
        per_contact, any_edge, worst = [], False, 0.0
        s = cfg.grid_step_px
        for c in contacts:
            need = c.radius + margin
            dist_edge = min(c.center[0] - x0, x1 - c.center[0], c.center[1] - y0, y1 - c.center[1])
            near = np.hypot(*(self.grid - c.center).T) < need
            lonely = near & ~self._inside(2 * c.center - self.grid)
            unpaired |= lonely
            frac = lonely.sum() / max(near.sum(), 1)
            is_edge = need > dist_edge
            any_edge |= is_edge
            worst = max(worst, frac)

            # Per-contact shear: sum over the largest disk symmetric about c (pairs only by construction).
            R = max(min(need, dist_edge), 0.0)
            shear = np.zeros(2)
            if R >= s:
                r = np.arange(-R, R + 1e-9, s)
                ox, oy = np.meshgrid(r, r)
                off = np.column_stack([ox.ravel(), oy.ravel()])
                off = off[np.hypot(off[:, 0], off[:, 1]) <= R]
                shear = rbf(c.center + off).sum(axis=0) * self.cell_area - cfg.perspective_alpha * c.persp
            per_contact.append(ContactShear(c.center, c.radius, R, is_edge, float(frac), shear / self.pitch ** 2))

        S_paired = U[~unpaired].sum(axis=0) * self.cell_area
        S_corr = S_paired - cfg.perspective_alpha * feat - self.tare
        conf = "ok" if not any_edge else ("edge-corrected" if worst < cfg.unreliable_unpaired_frac else "unreliable")
        reason = ""
        if conf != "ok":
            reason = "contact near window edge"
        if valid.mean() < 0.7:
            conf, reason = "unreliable", f"only {valid.mean():.0%} markers tracked: press r if gel is untouched"

        # Drift handling: when the markers say the gel is untouched, slowly pull the shading baseline
        # and the shear zero toward the current frame. Contacts that exist only because the lighting
        # drifted then fade out instead of accumulating.
        idle = bool(np.percentile(np.hypot(*disp[valid].T), 95) < cfg.idle_disp_px)
        self.idle_count = self.idle_count + 1 if idle else 0
        if cfg.auto_rebaseline and self.idle_count >= cfg.idle_frames:
            a = cfg.rebaseline_rate
            self.ref_blur = (1 - a) * self.ref_blur + a * blurred
            self.tare = (1 - a) * self.tare + a * (S_paired - cfg.perspective_alpha * feat)

        return Result(S_all, S_paired, S_corr, S_corr / self.pitch ** 2, self._to_force(S_corr), conf,
                      per_contact, feat, int(valid.sum()), unpaired, diff, idle, reason)

    def set_tare(self, result):
        self.tare = result.S_paired - self.cfg.perspective_alpha * result.persp_feature


def fit_perspective_alpha(features, S_values):
    """Least-squares scalar alpha with S_leak = alpha * feature, from pure-normal presses."""
    F = np.asarray(features).ravel()
    S = np.asarray(S_values).ravel()
    denom = float(F @ F)
    return float(F @ S / denom) if denom > 0 else 0.0
