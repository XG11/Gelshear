import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def auto_blackhat_ksize(cfg, shape):
    k = cfg.blackhat_ksize or max(7, int(shape[1] / 35))
    return k | 1  # odd


def detect_markers(frame_bgr, ksize):
    """Dark-dot detection that ignores smooth contact shading.

    Black-hat (closing - image) keeps dark features smaller than the kernel and
    suppresses the low-frequency photometric shading from contact. Centroids are
    weighted by black-hat response, so they are not pulled by shading gradients
    the way plain thresholded centroids are.
    Returns (N,2) xy positions and (N,) blob areas.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    _, mask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((0, 2)), np.zeros(0)
    idx = np.arange(1, n)
    com = np.array(ndimage.center_of_mass(bh.astype(np.float32), labels, idx))  # (row, col)
    return com[:, ::-1].copy(), stats[1:, cv2.CC_STAT_AREA].astype(float)


def marker_pitch(pts):
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


class MarkerTracker:
    """Frame-to-frame nearest-neighbour tracking against a fixed reference set.

    Each reference marker keeps an identity. Lost markers are marked invalid and
    their predicted position follows the mean motion of nearby valid markers, so
    they can be re-acquired once they reappear.
    """

    def __init__(self, ref_pts, gate):
        self.ref = ref_pts.copy()
        self.gate = gate
        self._ref_tree = cKDTree(self.ref)
        self.reset()

    def reset(self):
        self.pos = self.ref.copy()
        self.valid = np.ones(len(self.ref), bool)

    @property
    def disp(self):
        return self.pos - self.ref

    def update(self, det):
        new_valid = np.zeros(len(self.ref), bool)
        if len(det):
            d, j = cKDTree(det).query(self.pos, k=1, distance_upper_bound=self.gate)
            used = np.zeros(len(det), bool)
            for i in np.argsort(d):          # closest pairs claim detections first
                if not np.isfinite(d[i]):
                    break
                if not used[j[i]]:
                    used[j[i]] = True
                    self.pos[i] = det[j[i]]
                    new_valid[i] = True
        self.valid = new_valid
        inv = ~new_valid
        if inv.any() and new_valid.sum() >= 4:
            vidx = np.flatnonzero(new_valid)
            _, nn = cKDTree(self.ref[vidx]).query(self.ref[inv], k=4)
            self.pos[inv] = self.ref[inv] + self.disp[vidx][nn].mean(axis=1)
