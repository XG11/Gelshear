"""Hardware-free check of the whole pipeline on rendered GelSight-like frames.

Renders a dot grid on a shaded background, applies known displacement fields
(radial normal load, even shear, rotational torsion, optional perspective leak),
runs detection -> tracking -> integration, and reports how well normal load
cancels and shear is recovered.
"""
import numpy as np

from gelshear import Config, Pipeline
from gelshear.pipeline import fit_perspective_alpha

H, W = 480, 640
PITCH, MARGIN = 24, 20
SPREAD = 28.0            # spatial decay of the in-plane fields (px), ~ a few gel thicknesses
BETA = 0.0006            # perspective leak strength
rng = np.random.default_rng(0)

gy, gx = np.mgrid[MARGIN:H - MARGIN + 1:PITCH, MARGIN:W - MARGIN + 1:PITCH]
REF = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
YY, XX = np.mgrid[0:H, 0:W].astype(float)
CENTER = np.array([W / 2, H / 2])


def depth_at(p, contacts):
    d = np.zeros(len(p))
    for c in contacts:
        d += c["N"] * np.exp(-np.sum((p - c["c"]) ** 2, 1) / (2 * c["a"] ** 2))
    return d


def displacement(p, contacts, persp=True):
    u = np.zeros_like(p)
    for c in contacts:
        r = p - c["c"]
        g = np.exp(-np.sum(r ** 2, 1) / (2 * SPREAD ** 2))[:, None]
        u += 0.6 * c["N"] * r / SPREAD * g                          # normal: radial, odd
        u += np.asarray(c.get("T", (0, 0)), float) * g              # shear: even
        u += c.get("Q", 0) * np.column_stack([-r[:, 1], r[:, 0]]) / SPREAD * g  # torsion: odd
    if persp:
        u += BETA * depth_at(p, contacts)[:, None] * (p - CENTER)
    return u


def render(contacts, persp=True):
    base = np.stack([150 + 20 * XX / W, 140 + 15 * YY / H, 130 * np.ones_like(XX)], -1)
    pix = np.column_stack([XX.ravel(), YY.ravel()])
    dep = depth_at(pix, contacts).reshape(H, W)[..., None]
    img = base + dep * np.array([8.0, 5.0, -4.0])                  # contact shading
    pos = REF + displacement(REF, contacts, persp)
    for x, y in pos:
        x0, x1, y0, y1 = int(x) - 6, int(x) + 7, int(y) - 6, int(y) + 7
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        sub = np.exp(-((XX[y0:y1, x0:x1] - x) ** 2 + (YY[y0:y1, x0:x1] - y) ** 2) / (2 * 2.2 ** 2))
        img[y0:y1, x0:x1] *= (1 - 0.75 * sub)[..., None]
    img += rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def true_shear_integral(contacts):
    return sum(np.asarray(c.get("T", (0, 0)), float) for c in contacts) * 2 * np.pi * SPREAD ** 2


def run(pipe, contacts, steps=4, persp=True):
    pipe.reset_tracking()
    for k in range(1, steps + 1):              # ramp the load so tracking is exercised
        scaled = [{**c, "N": c["N"] * k / steps, "Q": c.get("Q", 0) * k / steps,
                   "T": tuple(np.asarray(c.get("T", (0, 0))) * k / steps)} for c in contacts]
        res = pipe.process(render(scaled, persp))
    return res


cfg = Config(width=0)
pipe = Pipeline(cfg, render([], persp=False))
print(f"markers: {len(pipe.tracker.ref)}  pitch: {pipe.pitch:.1f}px  diam: {pipe.marker_diam:.1f}px")
UNIT = 2 * np.pi * SPREAD ** 2 * 2.0   # integral of a 2 px shear, for scaling leak numbers

# ---- self-calibrate perspective from pure presses (no force sensor needed)
feats, Ss = [], []
for c in [(200, 160), (450, 300), (320, 240), (180, 330), (480, 150), (300, 120)]:
    r = run(pipe, [dict(c=np.array(c, float), a=22, N=8)])
    feats.append(r.persp_feature); Ss.append(r.S_paired)
cfg.perspective_alpha = fit_perspective_alpha(feats, Ss)
print(f"fitted perspective alpha: {cfg.perspective_alpha:.3e}\n")

cases = {
    "normal, off-center":         [dict(c=np.array([230., 190.]), a=22, N=9)],
    "normal, two contacts":       [dict(c=np.array([200., 250.]), a=20, N=8),
                                   dict(c=np.array([430., 220.]), a=26, N=10)],
    "normal, near edge":          [dict(c=np.array([45., 240.]), a=22, N=9)],
    "normal, very near corner":   [dict(c=np.array([40., 40.]), a=22, N=9)],
    "shear +y near edge":         [dict(c=np.array([590., 240.]), a=22, N=8, T=(0.0, 2.0))],
    "torsion + normal":           [dict(c=np.array([330., 250.]), a=22, N=8, Q=2.0)],
    "shear +x, + normal":         [dict(c=np.array([300., 230.]), a=22, N=8, T=(2.0, 0.0))],
    "shear diag, strong normal":  [dict(c=np.array([360., 260.]), a=22, N=14, T=(1.4, -1.4))],
    "two contacts, opposite shear": [dict(c=np.array([200., 240.]), a=22, N=8, T=(2.0, 0.0)),
                                     dict(c=np.array([440., 240.]), a=22, N=8, T=(-1.0, 1.0))],
}
print(f"{'case':30s} {'true':>16s} {'raw (all dots)':>16s} {'paired+corr':>16s}  confidence")
for name, cs in cases.items():
    r = run(pipe, cs)
    f = lambda v: f"({v[0]/UNIT:+.2f},{v[1]/UNIT:+.2f})"
    print(f"{name:30s} {f(true_shear_integral(cs)):>16s} {f(r.S_all - cfg.perspective_alpha*r.persp_feature):>16s} "
          f"{f(r.S_corrected):>16s}  {r.confidence}")
