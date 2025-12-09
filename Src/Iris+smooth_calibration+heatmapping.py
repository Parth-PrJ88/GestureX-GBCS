"""
src/gaze_prototype_with_alignment.py

Prototype: webcam-based gaze control with:
- alignment overlay (reference head+shoulders image)
- face & lighting checks (red/green outlines + notifications)
- fullscreen bright white calibration with smooth red-dot transitions (3x3)
- pause/resume calibration when face/lighting lost; show camera preview while paused
- fluid on-screen gaze cursor circle + OS cursor movement
- heat-trace overlay showing gaze concentrations

Requires:
    pip install mediapipe opencv-python numpy pyautogui scipy PyGetWindow

Save alignment reference image as: assets/align_reference.png (optional)
"""
import sys, time, math, ctypes
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import csv
from scipy.stats import pearsonr

# Optional helper for window activation on non-Windows
try:
    import pygetwindow as gw
except Exception:
    gw = None

# ----------------------
# Configuration & paths
# ----------------------
OUT_CSV = Path("data/eye_landmarks.csv")
OUT_MAP = Path("data/gaze_map.json")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Alignment image (provide your head+shoulders PNG with alpha or regular PNG)
ALIGN_IMAGE_PATH = Path("assets/align_reference.png")  # change if necessary

# Calibration params
SAMPLES_PER_POINT = 30          # frames to collect per target (approx 1s @30fps)
PRE_FIXATION_SEC = 0.8          # wait before sampling at each target
TRANSITION_SEC = 0.8            # time to smoothly move dot between points
GRID_POINTS = 9                 # 3x3 calibration
TARGET_COLOR = (0, 0, 255)      # red (BGR)
BACKGROUND_BRIGHTNESS = 255     # full white background during calibration

# Camera preview inset while aligning / paused during calibration
CAM_PREVIEW_SIZE = (360, 270)
PREVIEW_MARGIN = 20

# Face & lighting checks
MIN_FACE_AREA_PX = 3000         # minimum pixel area of face bbox to consider "present" (tune as needed)
MIN_FACE_BRIGHTNESS = 60        # mean brightness inside face bbox (0..255) required
# if brightness lower => consider "too dark"

# Cursor control smoothing / behavior
EMA_ALPHA = 0.18                 # exponential moving average alpha for smoothing cursor(previouslly 0.18)
MOVE_THRESHOLD_PX = 5           # min pixel delta to move OS cursor(previously 3)
VIRTUAL_CURSOR_RADIUS = 60      # radius of on-screen circle representing gaze(previously 24)
HEATMAP_DECAY = 0.95            # decay factor per frame for heatmap
HEATMAP_SCALE = 4               # downscale factor for heatmap array (smaller -> faster)(previously 4)
# size of calibration dot (pixels)
TARGET_RADIUS = 30

# Screen size (use to place calibration dots at real screen coords)
SCREEN_W, SCREEN_H = pyautogui.size()

# ----------------------
# MediaPipe initialization
# ----------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# iris indices (MediaPipe refined)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ----------------------
# Helper functions
# ----------------------
def avg_landmark(landmarks, idxs):
    pts = [(landmarks[i].x, landmarks[i].y) for i in idxs]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return float(np.mean(xs)), float(np.mean(ys))

def get_iris_centers(results):
    """Return normalized iris centers (lx,ly, rx,ry) or None if no face"""
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    lx, ly = avg_landmark(lm, LEFT_IRIS)
    rx, ry = avg_landmark(lm, RIGHT_IRIS)
    return lx, ly, rx, ry

def get_face_bbox_pixels(results, frame_w, frame_h):
    """Compute face bounding box in pixel coordinates from landmarks (or None)."""
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    minx = int(min(xs) * frame_w); maxx = int(max(xs) * frame_w)
    miny = int(min(ys) * frame_h); maxy = int(max(ys) * frame_h)
    # clamp
    minx = max(0, minx); miny = max(0, miny)
    maxx = min(frame_w-1, maxx); maxy = min(frame_h-1, maxy)
    return (minx, miny, maxx, maxy)

def face_present_and_bright(results, frame, min_area_px=MIN_FACE_AREA_PX, min_brightness=MIN_FACE_BRIGHTNESS):
    """
    Returns (present_bool, brightness_mean, bbox). present_bool True if face detected and area/brightness sufficient.
    """
    if not results.multi_face_landmarks:
        return False, 0.0, None
    h, w = frame.shape[:2]
    bbox = get_face_bbox_pixels(results, w, h)
    if bbox is None:
        return False, 0.0, None
    minx, miny, maxx, maxy = bbox
    area = (maxx - minx) * (maxy - miny)
    face_crop = frame[miny:maxy, minx:maxx]
    if face_crop.size == 0:
        return False, 0.0, bbox
    # brightness: convert to grayscale and compute mean
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    present = (area >= min_area_px) and (mean_brightness >= min_brightness)
    return present, mean_brightness, bbox

# ----------------------
# Window control helpers (bring to front)
# ----------------------
def bring_window_to_front_windows(title, retries=6, delay=0.12):
    """Windows: find window by title and briefly set it topmost to bring forward."""
    try:
        user32 = ctypes.windll.user32
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        for _ in range(retries):
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                user32.SetWindowPos(hwnd, HWND_TOPMOST, 0,0,0,0, SWP_NOMOVE | SWP_NOSIZE)
                time.sleep(0.03)
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0,0,0,0, SWP_NOMOVE | SWP_NOSIZE)
                return True
            time.sleep(delay)
    except Exception:
        return False
    return False

def bring_window_pygetwindow(title, retries=6, delay=0.12):
    """Fallback: try pygetwindow to activate a window by title."""
    if gw is None:
        return False
    try:
        for _ in range(retries):
            wins = gw.getWindowsWithTitle(title)
            if wins:
                try:
                    wins[0].activate()
                    return True
                except Exception:
                    pass
            time.sleep(delay)
    except Exception:
        return False
    return False

def bring_window_to_front(title):
    """Best-effort bring a named cv2 window to the front."""
    if sys.platform.startswith("win"):
        if bring_window_to_front_windows(title):
            return True
    return bring_window_pygetwindow(title)

# ----------------------
# Mapping helpers
# ----------------------
def build_design_matrix(eye_xy):
    x = eye_xy[:,0]; y = eye_xy[:,1]
    return np.column_stack([x, y, x*y, x**2, y**2, np.ones_like(x)])

def fit_poly_mapping(data):
    """
    data: list of (ex, ey, sx, sy)
    returns wx, wy coefficient arrays
    """
    arr = np.array(data)
    eye = arr[:,0:2]; sx = arr[:,2]; sy = arr[:,3]
    Phi = build_design_matrix(eye)
    wx, *_ = np.linalg.lstsq(Phi, sx, rcond=None)
    wy, *_ = np.linalg.lstsq(Phi, sy, rcond=None)
    return wx, wy

def predict_poly(wx, wy, ex, ey):
    f = np.array([ex, ey, ex*ey, ex*ex, ey*ey, 1.0])
    px = float(np.dot(f, wx))
    py = float(np.dot(f, wy))
    return px, py

def save_mapping(wx, wy, invert_x=False, invert_y=False):
    mapping = {"wx": wx.tolist(), "wy": wy.tolist(),
               "screen": [SCREEN_W, SCREEN_H],
               "invert_x": bool(invert_x), "invert_y": bool(invert_y)}
    with open(OUT_MAP, "w") as f:
        json.dump(mapping, f)

def load_mapping():
    if not OUT_MAP.exists():
        return None
    return json.load(open(OUT_MAP))

# ----------------------
# Heatmap helpers
# ----------------------
heatmap_h = max(1, SCREEN_H // HEATMAP_SCALE)
heatmap_w = max(1, SCREEN_W // HEATMAP_SCALE)
heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

def heatmap_add(px, py, strength=1.0, sigma=10.0):
    """
    Add a Gaussian blob to the downscaled `heatmap` array.
    px, py : pixel coordinates on the full screen (float or int)
    strength: amplitude multiplier
    sigma: gaussian sigma in heatmap-pixel units (not screen pixels)
    """
    global heatmap, heatmap_w, heatmap_h

    # convert full-screen coords to heatmap coords
    cx = int(round(px / HEATMAP_SCALE))
    cy = int(round(py / HEATMAP_SCALE))

    # quick bounds check
    if cx < 0 or cx >= heatmap_w or cy < 0 or cy >= heatmap_h:
        return

    # limit sigma to a reasonable range to keep patch small & fast
    sigma = max(1.0, float(sigma))
    half_size = int(min(max(3 * sigma, 3), max(heatmap_w, heatmap_h)))

    # compute patch bounds in heatmap coordinates
    x0 = max(0, cx - half_size)
    x1 = min(heatmap_w - 1, cx + half_size)
    y0 = max(0, cy - half_size)
    y1 = min(heatmap_h - 1, cy + half_size)

    # create coordinate grid for the patch with correct orientation: rows (y) x cols (x)
    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    X, Y = np.meshgrid(xs - cx, ys - cy)   # X has shape (len(ys), len(xs)) after meshgrid with (xs, ys) order
    # Note: np.meshgrid(x_vals, y_vals) by default returns arrays shaped (len(y_vals), len(x_vals))
    # so X corresponds to x-offsets, Y to y-offsets, both shaped (rows, cols)

    # Gaussian (rows x cols)
    gauss = np.exp(- (X.astype(np.float32)**2 + Y.astype(np.float32)**2) / (2.0 * (sigma**2)))

    # Add gaussian patch to heatmap slice (shapes must match)
    heatmap[y0:y1+1, x0:x1+1] += (strength * gauss)

    # clip to avoid runaway values
    np.clip(heatmap, 0.0, None, out=heatmap)


def heatmap_decay():
    global heatmap
    heatmap *= HEATMAP_DECAY
    # clip
    np.clip(heatmap, 0.0, None, out=heatmap)

def heatmap_render():
    """Return an upscaled BGR image showing heatmap overlay (uint8)."""
    hm = heatmap.copy()
    # normalize to 0..255
    if hm.max() > 0:
        hm = hm / (hm.max()) * 255.0
    hm = np.clip(hm, 0, 255).astype(np.uint8)
    # apply color map
    hm_color = cv2.applyColorMap(cv2.resize(hm, (SCREEN_W, SCREEN_H)), cv2.COLORMAP_JET)
    # make alpha from intensity
    alpha = cv2.resize(hm, (SCREEN_W, SCREEN_H)).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 1)
    overlay = (hm_color.astype(np.float32) * alpha[..., None] + 0.0).astype(np.uint8)
    return overlay

# ----------------------
# UI helpers (canvas, countdown)
# ----------------------
def make_fullscreen_canvas(color=BACKGROUND_BRIGHTNESS):
    """Create a canvas sized to the screen. color may be 0..255 (grayscale)."""
    return np.full((SCREEN_H, SCREEN_W, 3), color, dtype=np.uint8)

def draw_camera_inset(canvas, cam_frame):
    small = cv2.resize(cam_frame, CAM_PREVIEW_SIZE)
    h, w = CAM_PREVIEW_SIZE[1], CAM_PREVIEW_SIZE[0]
    x0, y0 = PREVIEW_MARGIN, PREVIEW_MARGIN
    canvas[y0:y0+h, x0:x0+w] = small
    cv2.rectangle(canvas, (x0-2, y0-2), (x0+w+2, y0+h+2), (200,200,200), 2)
    cv2.putText(canvas, "Camera preview", (x0, y0+h+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

def countdown_on_canvas(win_name, seconds=3, cam=None):
    """Large centered countdown on fullscreen canvas, optionally showing camera inset."""
    for n in range(seconds, 0, -1):
        canvas = make_fullscreen_canvas(255)
        txt = str(n)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(SCREEN_W, SCREEN_H) / 600.0
        thickness = max(6, int(6 * scale))
        size = cv2.getTextSize(txt, font, fontScale=6*scale, thickness=thickness)[0]
        x = (SCREEN_W - size[0]) // 2
        y = (SCREEN_H + size[1]) // 2
        cv2.putText(canvas, txt, (x, y), font, 6*scale, (0, 0, 255), thickness, cv2.LINE_AA)
        if cam is not None:
            ret, frame = cam.read()
            if ret:
                draw_camera_inset(canvas, cv2.flip(frame, 1))
        cv2.imshow(win_name, canvas)
        cv2.waitKey(1)
        time.sleep(1)

# ----------------------
# Alignment / face-check mode
# ----------------------
def mode_alignment():
    """
    Show camera preview with translucent reference image overlay (centered).
    When face is aligned and bright, user presses 's' to continue to menu or calibration.
    If face not visible/too dark, overlay camera preview outline red and show message.
    """
    cap = cv2.VideoCapture(0)
    win = "Alignment - position your head inside the reference"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 700)

    # load alignment image if exists
    align_img = None
    if ALIGN_IMAGE_PATH.exists():
        align_img = cv2.imread(str(ALIGN_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
        if align_img is None:
            align_img = None

    print("Alignment mode: Press 's' when aligned and face is green/ok (or 'q' to quit).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # face checks
        present, brightness, bbox = face_present_and_bright(results, frame)
        # draw camera preview with overlay reference
        # create slightly translucent overlay where the alignment image will be placed
        preview = cv2.flip(frame, 1)  # mirror for user comfort
        overlay = preview.copy()
        if align_img is not None:
            # resize align image to 60% of preview width
            scale = 0.6
            aw = int(preview.shape[1] * scale)
            ah = int(align_img.shape[0] * (aw / align_img.shape[1]))
            align_resized = cv2.resize(align_img, (aw, ah), interpolation=cv2.INTER_AREA)
            # if image has alpha channel
            if align_resized.shape[2] == 4:
                alpha = align_resized[:, :, 3] / 255.0
                for c in range(3):
                    overlay[(preview.shape[0]//2 - ah//2):(preview.shape[0]//2 + ah//2),
                            (preview.shape[1]//2 - aw//2):(preview.shape[1]//2 + aw//2), c] = \
                        (alpha * align_resized[:, :, c] + (1-alpha) * overlay[(preview.shape[0]//2 - ah//2):(preview.shape[0]//2 + ah//2),
                                                                              (preview.shape[1]//2 - aw//2):(preview.shape[1]//2 + aw//2), c])
            else:
                # simple blend
                y0 = preview.shape[0]//2 - ah//2
                x0 = preview.shape[1]//2 - aw//2
                overlay[y0:y0+ah, x0:x0+aw] = cv2.addWeighted(align_resized, 0.85, overlay[y0:y0+ah, x0:x0+aw], 0.15, 0)

        # draw colored outline depending on face condition
        outline_color = (0, 255, 0) if present else (0, 0, 255)  # green if OK else red
        cv2.rectangle(overlay, (5, 5), (preview.shape[1]-5, preview.shape[0]-5), outline_color, 4)

        # messages
        if present:
            cv2.putText(overlay, "Face detected. Press 's' to start calibration.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(overlay, "Face not found or too dark. Adjust camera/lighting.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            if bbox is not None:
                # draw small red bbox in preview (mirrored coords)
                minx, miny, maxx, maxy = bbox
                # convert bbox to mirrored preview coords
                pw = preview.shape[1]
                # landmarks were from non-flipped frame; map x accordingly for preview
                mx0 = pw - int(maxx / w * preview.shape[1])
                mx1 = pw - int(minx / w * preview.shape[1])
                my0 = int(miny / h * preview.shape[0])
                my1 = int(maxy / h * preview.shape[0])
                cv2.rectangle(overlay, (mx0, my0), (mx1, my1), (0,0,255), 2)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and present:
            # user confirmed alignment and face OK
            break
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(win)

# ----------------------
# Calibration flow
# ----------------------
def mode_calibration(num_points=9):
    """
    Smooth fullscreen calibration:
    - Uses a bright white fullscreen canvas so the display itself helps face detection.
    - Dots are red. They move smoothly between targets (linear interpolation).
    - At each target, we sample SAMPLES_PER_POINT frames while dot is visible.
    - If face is lost or too dark, pause and show camera preview with red outline & message.
      When resumed, show green outline briefly then hide preview and continue.
    - After collection: fit polynomial mapping, detect inversion, save mapping.
    """
    print("Starting calibration. Make sure you're aligned and press Enter to begin.")
    input("Press Enter to start (or Ctrl+C to cancel)...")

    if num_points not in (9, 16):
        num_points = 9
    n = 3 if num_points == 9 else 4
    xs = np.linspace(0.15, 0.85, n)
    ys = np.linspace(0.15, 0.85, n)
    targets = [(int(SCREEN_W * x), int(SCREEN_H * y)) for y in ys for x in xs]

    cap = cv2.VideoCapture(0)
    # warm up camera
    for _ in range(8):
        cap.read()

    win = "Calibration - look at the red dot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    bring_window_to_front(win)

    # initial countdown with camera inset
    countdown_on_canvas(win, seconds=3, cam=cap)

    collected = []  # (ex,ey,sx,sy) normalized ex/ey in camera frame

    for i, (tx, ty) in enumerate(targets):
        # compute next target; we will animate from current position to the target.
        # if first target, start from center
        if i == 0:
            cur_x, cur_y = SCREEN_W // 2, SCREEN_H // 2
        else:
            # current position: last target
            cur_x, cur_y = targets[i-1]

        # move smoothly from cur -> target over TRANSITION_SEC showing dot moving
        steps = max(1, int(TRANSITION_SEC * 60))  # aim ~60fps animation
        for s in range(steps):
            alpha = (s+1) / steps
            px = int(cur_x + alpha * (tx - cur_x))
            py = int(cur_y + alpha * (ty - cur_y))
            canvas = make_fullscreen_canvas(255)  # full bright white background
            # draw a moving red dot (bigger during transition)
            rr = int(TARGET_RADIUS * (0.8 + 0.4 * (1 - abs(0.5 - alpha))))
            cv2.circle(canvas, (px, py), rr, TARGET_COLOR, -1)
            # no camera preview during normal calibration (hidden)
            # display canvas
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return

        # now dwell at exact target and collect SAMPLES_PER_POINT frames
        samples = 0
        # Before sampling ensure face present and bright
        # If face is not ok, show camera preview and pause until ok
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if present:
                # brief green confirmation overlay before sampling
                canvas = make_fullscreen_canvas(255)
                cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
                cv2.putText(canvas, "Starting sample...", (30, SCREEN_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
                cv2.imshow(win, canvas)
                cv2.waitKey(300)  # short confirmation
                break
            else:
                # show camera preview with red outline and message
                canvas = make_fullscreen_canvas(255)
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                              (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                              (0,0,255), 4)
                cv2.putText(canvas, "Face not detected or too dark. Adjust camera/lighting.", (30, SCREEN_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow(win, canvas)
                # wait until fixed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
                time.sleep(0.05)
        # sample loop
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                # pause: show preview with red outline & message
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    canvas = make_fullscreen_canvas(255)
                    draw_camera_inset(canvas, cv2.flip(frame2, 1))
                    cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                  (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                  (0,0,255), 4)
                    cv2.putText(canvas, "Paused: face lost or too dark. Fix and wait...", (30, SCREEN_H - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return
                    # re-check
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
                        # show green briefly then remove preview and resume
                        ok_canvas = make_fullscreen_canvas(255)
                        draw_camera_inset(ok_canvas, cv2.flip(frame2, 1))
                        cv2.rectangle(ok_canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                      (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                      (0,255,0), 4)
                        cv2.putText(ok_canvas, "Fixed. Resuming in 0.7s...", (30, SCREEN_H - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                        cv2.imshow(win, ok_canvas); cv2.waitKey(1)
                        time.sleep(0.7)
                        break
                # after fixing, continue sampling loop (re-evaluate current frame)
                continue

            # compute iris centers and average as normalized center
            centers = get_iris_centers(results)
            if centers:
                lx_n, ly_n, rx_n, ry_n = centers
                ex = (lx_n + rx_n) / 2.0
                ey = (ly_n + ry_n) / 2.0
                collected.append((ex, ey, tx, ty))
                samples += 1

            # show small dot while sampling (so user knows it's still there)
            canvas = make_fullscreen_canvas(255)
            cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
            cv2.putText(canvas, f"Collecting {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return

    # End of all targets
    cap.release()
    cv2.destroyAllWindows()

    # Fit mapping
    if len(collected) < 6:
        print("Not enough samples collected.")
        return
    wx, wy = fit_poly_mapping(collected)

    # detect inversion via Pearson correlation
    pred_x = np.array([predict_poly(wx, wy, ex, ey)[0] for ex, ey, _, _ in collected])
    pred_y = np.array([predict_poly(wx, wy, ex, ey)[1] for ex, ey, _, _ in collected])
    true_x = np.array([r[2] for r in collected])
    true_y = np.array([r[3] for r in collected])

    invert_x = invert_y = False
    try:
        cx, _ = pearsonr(pred_x, true_x)
        cy, _ = pearsonr(pred_y, true_y)
        if cx < 0: invert_x = True
        if cy < 0: invert_y = True
    except Exception:
        pass

    save_mapping(wx, wy, invert_x=invert_x, invert_y=invert_y)
    errors = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
    print("Calibration finished. Mean error: {:.1f}px, Median: {:.1f}px".format(errors.mean(), np.median(errors)))
    print("Saved mapping to", OUT_MAP)

# ----------------------
# Cursor control overlay + movement
# ----------------------
def mode_cursor_control():
    """
    Load saved mapping and run live cursor mode:
    - Move OS cursor via pyautogui.moveTo(ema_x, ema_y)
    - Also show a fullscreen topmost overlay that renders heatmap + a larger, semi-transparent circle
      so audience sees a fluid, bigger 'cursor' and heat traces.
    """
    mapping = load_mapping()
    if mapping is None:
        print("No mapping found. Run calibration first.")
        return
    wx = np.array(mapping["wx"]); wy = np.array(mapping["wy"])
    invert_x = bool(mapping.get("invert_x", False))
    invert_y = bool(mapping.get("invert_y", False))

    cap = cv2.VideoCapture(0)
    ema_x = None; ema_y = None
    win = "Gaze Overlay (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    bring_window_to_front(win)

    last_move_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            centers = get_iris_centers(results)
            if centers:
                lx_n, ly_n, rx_n, ry_n = centers
                ex = (lx_n + rx_n) / 2.0
                ey = (ly_n + ry_n) / 2.0
                px, py = predict_poly(wx, wy, ex, ey)
                if invert_x: px = SCREEN_W - px
                if invert_y: py = SCREEN_H - py
                px = float(np.clip(px, 0, SCREEN_W-1))
                py = float(np.clip(py, 0, SCREEN_H-1))

                # smoothing
                if ema_x is None:
                    ema_x, ema_y = px, py
                else:
                    ema_x = EMA_ALPHA * px + (1 - EMA_ALPHA) * ema_x
                    ema_y = EMA_ALPHA * py + (1 - EMA_ALPHA) * ema_y

                # move OS cursor with threshold
                curx, cury = pyautogui.position()
                dx = abs(ema_x - curx); dy = abs(ema_y - cury)
                if dx > MOVE_THRESHOLD_PX or dy > MOVE_THRESHOLD_PX:
                    pyautogui.moveTo(int(ema_x), int(ema_y))
                    last_move_time = time.time()

                # add to heatmap
                heatmap_add(ema_x, ema_y, strength=1.0, sigma=8.0)
            # heatmap decay each frame
            heatmap_decay()

            # render overlay: heatmap plus semi-transparent circle
            canvas = make_fullscreen_canvas(0)  # black background overlay
            hm_img = heatmap_render()
            # blend heatmap with canvas
            canvas = cv2.addWeighted(canvas, 1.0, hm_img, 0.7, 0)
            # draw circle where ema predicts (bigger, semi-transparent)
            if ema_x is not None:
                cv2.circle(canvas, (int(ema_x), int(ema_y)), VIRTUAL_CURSOR_RADIUS, (0, 255, 255), -1)
                cv2.circle(canvas, (int(ema_x), int(ema_y)), VIRTUAL_CURSOR_RADIUS, (255, 255, 255), 2)
            cv2.putText(canvas, "Gaze overlay. Press 'q' to quit.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # allow closing with window X
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ----------------------
# Main menu
# ----------------------
if __name__ == "__main__":
    print("""
    GAZE PROTOTYPE MENU
    1 -> Alignment (camera + alignment reference)
    2 -> Calibration (fullscreen smooth red-dot 3x3)
    3 -> Cursor control (overlay + OS cursor moves)
    q -> Quit
    """)
    while True:
        cmd = input("Enter mode: ").strip().lower()
        if cmd == "1":
            mode_alignment()
        elif cmd == "2":
            mode_calibration(num_points=3*3 if GRID_POINTS == 9 else 16)
        elif cmd == "3":
            mode_cursor_control()
        elif cmd == "q":
            break
        else:
            print("Invalid. Choose 1,2,3 or q.")