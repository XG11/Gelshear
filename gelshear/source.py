import cv2


def preprocess(frame, cfg):
    if cfg.crop_frac > 0:
        h, w = frame.shape[:2]
        cy, cx = int(h * cfg.crop_frac), int(w * cfg.crop_frac)
        frame = frame[cy:h - cy, cx:w - cx]
    if cfg.width and frame.shape[1] != cfg.width:
        s = cfg.width / frame.shape[1]
        frame = cv2.resize(frame, (cfg.width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return frame


class FrameSource:
    """Camera index or video file, returning preprocessed BGR frames."""

    def __init__(self, cfg):
        src = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source {cfg.source!r}")
        self.cfg = cfg

    def read(self):
        ok, frame = self.cap.read()
        return preprocess(frame, self.cfg) if ok else None

    def release(self):
        self.cap.release()
