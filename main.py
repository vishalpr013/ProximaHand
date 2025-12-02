"""
ProximaHand — Real-time hand proximity POC (smoothed outlines)

- Skin + optional background-subtraction based hand segmentation.
- Contour smoothing using uniform resampling + Chaikin corner-cutting.
- Fingertip candidate detection (convexity defects with robust fallback).
- Virtual object with SAFE / WARNING / DANGER states.
- HSV sampler ('s'), mask toggle ('t'), and simple UI trackbars for tuning.
"""

import cv2
import numpy as np
import time
from collections import deque

# ---------------- Globals / UI ----------------
cancel_clicked = False
button_rect = (10, 10, 140, 50)  # coordinates for the CANCEL button
last_mouse = (0, 0)


def nothing(x):
    """No-op callback for OpenCV trackbars."""
    pass


def create_hsv_trackbars(win_name='HSV'):
    """Create trackbars used to tune detection and smoothing parameters."""
    cv2.namedWindow(win_name)
    # HSV skin range controls
    cv2.createTrackbar('H_min', win_name, 0, 179, nothing)
    cv2.createTrackbar('H_max', win_name, 25, 179, nothing)
    cv2.createTrackbar('S_min', win_name, 30, 255, nothing)
    cv2.createTrackbar('S_max', win_name, 231, 255, nothing)
    cv2.createTrackbar('V_min', win_name, 60, 255, nothing)
    cv2.createTrackbar('V_max', win_name, 255, 255, nothing)
    # Face removal padding and preprocessing toggles
    cv2.createTrackbar('FacePad', win_name, 0, 200, nothing)
    cv2.createTrackbar('CLAHE', win_name, 1, 1, nothing)
    cv2.createTrackbar('BG_SUB', win_name, 0, 1, nothing)  # background subtraction on/off
    cv2.createTrackbar('Exposure', win_name, 50, 100, nothing)
    cv2.createTrackbar('AreaThresh', win_name, 2000, 20000, nothing)
    # Smoothing controls
    cv2.createTrackbar('SmoothRes', win_name, 150, 600, nothing)   # resampling resolution
    cv2.createTrackbar('ChaikinIters', win_name, 3, 6, nothing)    # Chaikin iterations


def read_hsv_trackbars(win_name='HSV'):
    """Read and return the current trackbar values."""
    hmin = cv2.getTrackbarPos('H_min', win_name)
    hmax = cv2.getTrackbarPos('H_max', win_name)
    smin = cv2.getTrackbarPos('S_min', win_name)
    smax = cv2.getTrackbarPos('S_max', win_name)
    vmin = cv2.getTrackbarPos('V_min', win_name)
    vmax = cv2.getTrackbarPos('V_max', win_name)
    facepad = cv2.getTrackbarPos('FacePad', win_name)
    clahe_on = cv2.getTrackbarPos('CLAHE', win_name)
    bg_sub_on = cv2.getTrackbarPos('BG_SUB', win_name)
    exp_slider = cv2.getTrackbarPos('Exposure', win_name)
    area_thresh = cv2.getTrackbarPos('AreaThresh', win_name)
    smooth_res = cv2.getTrackbarPos('SmoothRes', win_name)
    chaikin_iters = cv2.getTrackbarPos('ChaikinIters', win_name)
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    return lower, upper, facepad, bool(clahe_on), bool(bg_sub_on), exp_slider, int(area_thresh), int(smooth_res), int(chaikin_iters)


def mouse_callback(event, x, y, flags, param):
    """Track mouse position and handle CANCEL clicks."""
    global cancel_clicked, button_rect, last_mouse
    last_mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = button_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            cancel_clicked = True


# ---------------- Utility: sampling & preprocessing ----------------
def sample_hsv_from_frame(frame, x, y, size=10):
    """
    Sample a small square ROI around (x,y) and return an HSV range (lower, upper)
    built around median H,S,V values from that ROI. Useful for fast calibration.
    """
    h, w = frame.shape[:2]
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w - 1, x + size)
    y2 = min(h - 1, y + size)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_med = int(np.median(hsv[:, :, 0]))
    s_med = int(np.median(hsv[:, :, 1]))
    v_med = int(np.median(hsv[:, :, 2]))
    # small paddings around median values
    h_pad = 10
    s_pad = 40
    v_pad = 60
    lower = np.array([max(0, h_med - h_pad), max(0, s_med - s_pad), max(0, v_med - v_pad)], dtype=np.uint8)
    upper = np.array([min(179, h_med + h_pad), min(255, s_med + s_pad), min(255, v_med + v_pad)], dtype=np.uint8)
    return lower, upper


# ---------------- Fingertip detection (robust) ----------------
def robust_get_fingertips(contour, max_points=5):
    """
    Return fingertip candidate points from a contour.
    Uses convexityDefects when available (guarded with try/except) and falls back to hull points.
    """
    if contour is None or len(contour) < 5:
        return []
    eps = 0.01 * cv2.arcLength(contour, True)
    try:
        simple = cv2.approxPolyDP(contour, eps, True)
    except Exception:
        simple = contour
    if simple is None or len(simple) < 5:
        return []
    try:
        hull_idx = cv2.convexHull(simple, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            raise ValueError('hull too small')
        defects = cv2.convexityDefects(simple, hull_idx)
        pts = []
        if defects is not None:
            # depth filter avoids tiny defects/noise
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]
                start = tuple(simple[s][0])
                end = tuple(simple[e][0])
                if depth > 1000:
                    pts.append(start)
                    pts.append(end)
        # unique and top-first sort
        uniq = []
        for p in pts:
            if p not in uniq:
                uniq.append(p)
        uniq_sorted = sorted(uniq, key=lambda x: x[1])
        return uniq_sorted[:max_points]
    except Exception:
        # fallback: use convex hull points (top-most ones)
        try:
            hull_pts = cv2.convexHull(simple, returnPoints=True)
            hull_list = [tuple(pt[0]) for pt in hull_pts.reshape(-1, 1, 2)]
            hull_sorted = sorted(hull_list, key=lambda x: x[1])
            final = []
            for p in hull_sorted:
                if not any((abs(p[0] - q[0]) < 12 and abs(p[1] - q[1]) < 12) for q in final):
                    final.append(p)
                if len(final) >= max_points:
                    break
            return final
        except Exception:
            return []


# ---------------- Preprocessing helper ----------------
def apply_clahe_to_v(bgr):
    """
    Apply CLAHE to the V (value) channel of the image HSV — enhances local contrast and perceived brightness.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------- Contour smoothing helpers ----------------
def contour_to_xy_list(contour):
    """Return an ordered list of (x,y) from a contour array."""
    pts = contour.reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in pts]


def resample_contour(points, n_samples=150):
    """
    Uniformly resample an ordered closed contour into n_samples points.
    Works by computing arc-length positions and interpolating along segments.
    """
    if len(points) < 2:
        return points[:]
    pts = np.array(points, dtype=np.float32)
    # diffs with wrap-around (close loop)
    diffs = np.diff(pts, axis=0, append=[pts[0]])
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_len = seg_lengths.sum()
    if total_len == 0:
        return points[:]
    # cumulative position for each vertex
    cum = np.cumsum(seg_lengths)
    cum = np.insert(cum, 0, 0.0)[:-1]
    sample_ds = np.linspace(0, total_len, n_samples, endpoint=False)
    resampled = []
    for d in sample_ds:
        idx = np.searchsorted(cum, d) - 1
        if idx < 0:
            idx = 0
        i0 = idx % len(pts)
        i1 = (i0 + 1) % len(pts)
        seg_start = cum[i0]
        seg_len = seg_lengths[i0]
        t = 0 if seg_len == 0 else (d - seg_start) / seg_len
        p = (1 - t) * pts[i0] + t * pts[i1]
        resampled.append((float(p[0]), float(p[1])))
    return resampled


def chaikin_curve(points, iterations=2):
    """
    Smooth a closed contour using Chaikin's corner-cutting algorithm.
    Very fast, stable, and suitable for real-time smoothing.
    """
    if len(points) < 3:
        return points[:]
    pts = [(float(x), float(y)) for x, y in points]
    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.append(q)
            new_pts.append(r)
        pts = new_pts
    return [(int(round(x)), int(round(y))) for x, y in pts]


# ---------------- Main loop ----------------
def main():
    global cancel_clicked, button_rect, last_mouse

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    # Reasonable working resolution for real-time CPU performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    create_hsv_trackbars()
    win_name = 'ProximaHand POC - Debug Friendly (Smoothed)'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    prev_time = time.time()
    fps = 0

    while True:
        if cancel_clicked:
            print('Cancel clicked — exiting')
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w = frame.shape[:2]

        # read UI tuneable parameters
        lower, upper, facepad, clahe_on, bg_sub_on, exp_slider, area_thresh, smooth_res, chaikin_iters = read_hsv_trackbars()

        # attempt to set camera exposure (may be ignored by some webcams/drivers)
        try:
            exp_val = float(int((exp_slider / 100.0) * 4) - 6)  # safe mapped range
            cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        except Exception:
            pass

        # compute basic stats early (used for overlay)
        avg_v = int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]))
        try:
            cam_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            cam_exp = -1

        # optional local contrast enhancement (CLAHE) to improve segmentation in low light
        proc_frame = apply_clahe_to_v(frame) if clahe_on else frame

        # build HSV skin mask
        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # optional background subtraction to require motion
        if bg_sub_on:
            fg = bg_sub.apply(proc_frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            fg = cv2.medianBlur(fg, 5)
            combined = cv2.bitwise_and(mask, fg)
        else:
            combined = mask

        # morphological cleanup on combined mask
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        combined = cv2.GaussianBlur(combined, (7, 7), 0)

        # detect faces and remove them from the mask (avoid detecting face as hand)
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, fw, fh) in faces:
            pad = int(facepad)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)
            combined[y1:y2, x1:x2] = 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # find contours on the cleaned mask and select the largest plausible hand blob
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = None
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                if cv2.contourArea(cnt) >= area_thresh:
                    hand_contour = cnt
                    break

        # draw virtual object and determine SAFE/WARNING/DANGER
        obj_center = (int(w * 0.7), int(h * 0.5))
        obj_radius = int(min(h, w) * 0.12)
        state = 'SAFE'

        if hand_contour is not None:
            # convert contour to a simple XY list
            pts = contour_to_xy_list(hand_contour)

            # only smooth sufficiently large contours to avoid working on noise
            if len(pts) >= 6:
                # resample to a stable number of points then apply Chaikin smoothing
                resampled = resample_contour(pts, n_samples=max(50, smooth_res))
                smoothed = chaikin_curve(resampled, iterations=max(1, chaikin_iters))
                # draw the smoothed closed polyline (antialiased)
                if len(smoothed) > 1:
                    cv2.polylines(frame, [np.array(smoothed, dtype=np.int32)], isClosed=True,
                                  color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

                # show fingertip candidates (optional)
                tips = robust_get_fingertips(hand_contour, max_points=5)
                for p in tips:
                    cv2.circle(frame, p, 8, (255, 255, 0), -1)
            else:
                # fallback: draw raw contour
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

            # distance-based state logic (closest contour point to object boundary)
            contour_pts = hand_contour.reshape(-1, 2)
            dists = np.linalg.norm(contour_pts - np.array(obj_center), axis=1)
            boundary_dists = dists - obj_radius
            min_boundary_dist = float(boundary_dists.min())
            idx = int(np.argmin(boundary_dists))
            closest_pt = tuple(contour_pts[idx].astype(int))
            cv2.circle(frame, closest_pt, 6, (0, 0, 255), -1)

            # thresholds (pixels)
            danger_thresh = 20
            warning_thresh = 100
            if min_boundary_dist <= danger_thresh:
                state = 'DANGER'
            elif min_boundary_dist <= warning_thresh:
                state = 'WARNING'
            else:
                state = 'SAFE'

            cv2.putText(frame, f'Dist: {int(min_boundary_dist)} px', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # clearly indicate absence of the hand to avoid hallucination
            cv2.putText(frame, 'Hand not detected', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

        # draw object and overlay state text
        if state == 'SAFE':
            obj_color = (0, 255, 0)
        elif state == 'WARNING':
            obj_color = (0, 165, 255)
        else:
            obj_color = (0, 0, 255)
        cv2.circle(frame, obj_center, obj_radius, obj_color, 3)

        overlay_text = 'DANGER DANGER' if state == 'DANGER' else state
        text_size, _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)
        tx, ty = 10, 90
        cv2.rectangle(frame, (tx - 6, ty - text_size[1] - 6),
                      (tx + text_size[0] + 6, ty + 6), (0, 0, 0), -1)
        txt_col = (0, 255, 0) if state == 'SAFE' else ((0, 165, 255) if state == 'WARNING' else (0, 0, 255))
        cv2.putText(frame, overlay_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, txt_col, 3)

        # CANCEL button
        bx, by, bw, bh = button_rect
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.putText(frame, 'CANCEL', (bx + 18, by + 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        # small raw thumbnail (top-right) for quick visual debugging
        thumb = cv2.resize(frame, (200, 150))
        th_h, th_w = thumb.shape[:2]
        frame[0:th_h, w - th_w:w] = thumb

        # stats overlay (average V / exposure / area threshold)
        cv2.putText(frame, f'AvgV:{avg_v}', (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f'Exp:{cam_exp:.1f}', (w - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f'AreaT:{area_thresh}', (w - 200, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # frame timing display
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur_time - prev_time)) if cur_time != prev_time else fps
        prev_time = cur_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # show final image
        cv2.imshow(win_name, frame)
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('t'):
            # toggle mask window for debugging
            if cv2.getWindowProperty('mask', cv2.WND_PROP_VISIBLE) < 1:
                cv2.imshow('mask', combined)
            else:
                cv2.destroyWindow('mask')
        elif key == ord('s'):
            # sample HSV around last mouse position and set trackbars
            sample = sample_hsv_from_frame(frame, last_mouse[0], last_mouse[1], size=10)
            if sample is not None:
                lower_s, upper_s = sample
                cv2.setTrackbarPos('H_min', 'HSV', int(lower_s[0]))
                cv2.setTrackbarPos('H_max', 'HSV', int(upper_s[0]))
                cv2.setTrackbarPos('S_min', 'HSV', int(lower_s[1]))
                cv2.setTrackbarPos('S_max', 'HSV', int(upper_s[1]))
                cv2.setTrackbarPos('V_min', 'HSV', int(lower_s[2]))
                cv2.setTrackbarPos('V_max', 'HSV', int(upper_s[2]))
                print(f'Sampled HSV median -> H:{(lower_s[0] + upper_s[0]) // 2} '
                      f'S:{(lower_s[1] + upper_s[1]) // 2} V:{(lower_s[2] + upper_s[2]) // 2}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
