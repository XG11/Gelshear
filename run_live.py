"""Live (or video-file) net-shear extraction from a marker GelSight.

Keys:
  r  re-capture no-contact reference (keep the gel untouched)
  z  tare: zero the current net shear
  c  start/stop perspective calibration: do PURE NORMAL presses at many positions
  f  fit perspective correction from the calibration presses and save config
  s  start/stop recording per-frame data to an .npz
  d  toggle debug view of the shading-change map (white = counted as contact)
  q  quit (Esc also works)
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from gelshear import Config, Pipeline
from gelshear.pipeline import fit_perspective_alpha
from gelshear.source import FrameSource
from gelshear.viz import draw


def capture_reference(src, n):
    frames = []
    while len(frames) < n:
        f = src.read()
        if f is None:
            break
        frames.append(f)
    if not frames:
        raise RuntimeError("No frames for reference")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="camera index or video path (overrides config)")
    ap.add_argument("--config", default="config.json", help="loaded if it exists; calibration is saved here")
    ap.add_argument("--headless", action="store_true", help="no window; print estimates (for video files)")
    args = ap.parse_args()

    try:
        cfg = Config.load(args.config)
    except FileNotFoundError:
        cfg = Config()
    if args.source is not None:
        cfg.source = args.source

    src = FrameSource(cfg)
    print("Capturing reference: do not touch the gel...")
    pipe = Pipeline(cfg, capture_reference(src, cfg.ref_frames))
    print(f"{len(pipe.tracker.ref)} markers, pitch {pipe.pitch:.1f}px, diameter {pipe.marker_diam:.1f}px")

    calib, calibrating = ([], []), False
    rec, recording = [], False
    status, t_prev, frame_i = "", time.time(), 0
    history = deque(maxlen=cfg.history_len)
    show_diff = False

    while True:
        frame = src.read()
        if frame is None:
            break
        res = pipe.process(frame)
        frame_i += 1

        if calibrating and res.contacts:
            calib[0].append(res.persp_feature); calib[1].append(res.S_paired)
        if recording:
            rec.append(dict(t=time.time(), pos=pipe.tracker.pos.copy(), valid=pipe.tracker.valid.copy(),
                            S_all=res.S_all, S_paired=res.S_paired, S_corrected=res.S_corrected,
                            net_px=res.net_px, persp_feature=res.persp_feature, n_contacts=len(res.contacts),
                            confidence=res.confidence,
                            force=res.force_N if res.force_N is not None else np.full(2, np.nan)))

        history.append(res.force_N if res.force_N is not None else res.net_px)
        if args.headless:
            m = res.net_px
            print(f"{frame_i:5d}  net=({m[0]:+.2f},{m[1]:+.2f})px  contacts={len(res.contacts)}"
                  f"  valid={res.n_valid}  {res.confidence}  {res.reason}")
            continue

        now = time.time()
        fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now
        tag = (" [CALIB]" if calibrating else "") + (" [REC]" if recording else "")
        cv2.imshow("gelshear", draw(frame, pipe, res, history, f"{fps:4.1f} fps{tag}  {status}", show_diff))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            pipe.set_reference(capture_reference(src, cfg.ref_frames)); status = "reference updated"
        elif key == ord("d"):
            show_diff = not show_diff
        elif key == ord("z"):
            pipe.set_tare(res); status = "tared"
        elif key == ord("c"):
            calibrating = not calibrating
            if calibrating:
                calib = ([], [])
            status = "calibrating: pure presses only" if calibrating else f"{len(calib[0])} calib frames"
        elif key == ord("f"):
            if len(calib[0]) >= 20:
                cfg.perspective_alpha = fit_perspective_alpha(*calib)
                cfg.save(args.config); status = f"alpha={cfg.perspective_alpha:.3e} saved"
            else:
                status = "need >=20 calibration frames with contact"
        elif key == ord("s"):
            recording = not recording
            if not recording and rec:
                fn = time.strftime("rec_%Y%m%d_%H%M%S.npz")
                np.savez(fn, ref=pipe.tracker.ref, pitch=pipe.pitch,
                         **{k: np.array([r[k] for r in rec]) for k in rec[0]})
                status, rec = f"saved {fn}", []

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
