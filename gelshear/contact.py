from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Contact:
    center: np.ndarray   # (x, y), shading-weighted centroid
    radius: float        # equivalent radius sqrt(area/pi)
    area: int
    strength: float      # summed shading change, a rough indentation proxy
    persp: np.ndarray = None  # this contact's perspective feature, sum w*(p - o)


def contact_diff(frame, ref_blur, ksize):
    """Shading change with markers removed (median blur wider than a dot).
    Returns (diff map, blurred frame) so the caller can reuse the blur for re-baselining."""
    a = cv2.medianBlur(frame, ksize).astype(np.float32)
    d = np.linalg.norm(a - ref_blur, axis=2)
    return cv2.GaussianBlur(d, (5, 5), 0), a


def find_contacts(diff, thresh, min_area, optical_center=(0.0, 0.0)):
    mask = (diff > thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w_all = np.clip(diff - thresh, 0, None)
    contacts = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        w = w_all[ys, xs] + 1e-6
        c = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
        persp = np.array([np.sum(w * (xs - optical_center[0])), np.sum(w * (ys - optical_center[1]))])
        contacts.append(Contact(c, float(np.sqrt(area / np.pi)), area, float(w.sum()), persp))
    return contacts, mask.astype(bool)


def perspective_feature(diff, mask, thresh, optical_center):
    """sum_w * (p - o) over the contact region: predicts the radial-about-the-lens
    marker shift caused by pressing the gel toward the camera."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(2)
    w = np.clip(diff[ys, xs] - thresh, 0, None)
    return np.array([np.sum(w * (xs - optical_center[0])), np.sum(w * (ys - optical_center[1]))])
