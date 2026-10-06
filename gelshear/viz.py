import cv2
import numpy as np

CONF_COLOR = {"ok": (60, 200, 60), "edge-corrected": (0, 165, 255),
              "unreliable": (0, 0, 255), "no-markers": (0, 0, 255)}


def _text(img, t, org, scale=0.45, color=(255, 255, 255)):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _history_strip(history, width, height, unit):
    strip = np.full((height, width, 3), 30, np.uint8)
    if len(history) < 2:
        return strip
    h = np.array(history)
    lim = max(np.abs(h).max() * 1.1, 1e-6 if unit == "N" else 2.0)
    mid = height // 2
    cv2.line(strip, (0, mid), (width, mid), (90, 90, 90), 1)
    xs = np.linspace(0, width - 1, len(h)).astype(np.int32)
    for k, col in ((0, (80, 80, 255)), (1, (255, 180, 60))):
        ys = (mid - h[:, k] / lim * (mid - 4)).astype(np.int32)
        cv2.polylines(strip, [np.column_stack([xs, ys])], False, col, 1, cv2.LINE_AA)
    _text(strip, f"x", (6, 14), 0.4, (80, 80, 255))
    _text(strip, f"y", (20, 14), 0.4, (255, 180, 60))
    _text(strip, f"+/-{lim:.2f} {unit}", (width - 110, 14), 0.4)
    return strip


def draw(frame, pipe, res, history=(), status="", show_diff=False):
    cfg = pipe.cfg
    img = frame.copy()
    if show_diff and res.diff is not None:
        # debug view: where the shading differs from the baseline; bright = detected as contact
        d = np.clip(res.diff / max(cfg.contact_thresh * 3, 1e-6) * 255, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        heat[res.diff > cfg.contact_thresh] = (255, 255, 255)
        img = cv2.addWeighted(img, 0.35, heat, 0.65, 0)
    tr = pipe.tracker

    # unpaired grid points (dropped from the sum) shown as faint red
    if res.unpaired_mask is not None and res.unpaired_mask.any():
        for p in pipe.grid[res.unpaired_mask]:
            cv2.circle(img, tuple(np.int32(p)), 1, (0, 0, 180), -1)

    for p0, p, ok in zip(tr.ref, tr.pos, tr.valid):
        if ok:
            tip = p0 + (p - p0) * cfg.arrow_scale
            cv2.arrowedLine(img, tuple(np.int32(p0)), tuple(np.int32(tip)), (0, 200, 0), 1, tipLength=0.3)
        else:
            cv2.circle(img, tuple(np.int32(p)), 2, (0, 0, 255), -1)

    x0, y0, x1, y1 = pipe.bbox
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (120, 120, 120), 1)

    # per-contact: symmetric pairing disk + its own shear arrow
    for c in res.contacts:
        col = (0, 0, 255) if c.unpaired_frac >= cfg.unreliable_unpaired_frac else \
              (0, 165, 255) if c.edge else (255, 160, 0)
        ctr = tuple(np.int32(c.center))
        if c.R_sym > 1:
            cv2.circle(img, ctr, int(c.R_sym), col, 1)
        cv2.drawMarker(img, ctr, col, cv2.MARKER_CROSS, 10, 2)
        tip = c.center + c.shear_px * cfg.contact_arrow_scale
        cv2.arrowedLine(img, ctr, tuple(np.int32(tip)), col, 2, tipLength=0.25)

    # net shear: arrow from image centre + numeric readout
    h, w = img.shape[:2]
    o = np.array([w / 2, h / 2])
    ccol = CONF_COLOR.get(res.confidence, (255, 255, 255))
    cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(o + res.net_px * cfg.net_arrow_scale)),
                    (255, 0, 255), 4, tipLength=0.25)
    cv2.circle(img, tuple(np.int32(o)), 4, (255, 0, 255), -1)

    if res.force_N is not None:
        val, unit = res.force_N, "N"
        head = f"Shear  Fx {val[0]:+.2f}  Fy {val[1]:+.2f}  |F| {np.hypot(*val):.2f} N"
    else:
        val, unit = res.net_px, "px"
        head = f"Shear  x {val[0]:+.1f}  y {val[1]:+.1f}  |s| {np.hypot(*val):.1f}  (summed marker px)"
    _text(img, head, (8, 20), 0.55)
    _text(img, f"confidence: {res.confidence}" + (f"  ({res.reason})" if res.reason else "")
          + ("  [idle]" if res.idle else ""), (8, 40), 0.45, ccol)
    _text(img, f"markers {res.n_valid}/{len(tr.ref)}   contacts {len(res.contacts)}   {status}", (8, 58), 0.42)

    strip = _history_strip(list(history), img.shape[1], 90, unit)
    return np.vstack([img, strip])