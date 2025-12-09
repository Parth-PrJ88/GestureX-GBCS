# src/gaze_prototype_with_alignment_4x4_with_drags_fixed_endpoint.py
# Updated prototype:
#  - 4x4 grid (phase 1)
#  - Phase 2: slower center->edge drags, arrow indicator at center, record drag samples to JSON,
#             then place a calibration point at each edge and sample SAMPLES_PER_POINT there
#  - Fix: ensure drag animation ends exactly at the edge and immediately start sampling there (no pop)
#
# Requirements:
# pip install mediapipe opencv-python numpy pyautogui scipy PyGetWindow
#
# Save alignment reference image (optional) at: assets/align_reference.png

import sys, time, math, ctypes
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import csv
from scipy.stats import pearsonr

# ==========================================
# CURSOR CONTROL (HIDE + SHOW)
# ==========================================
import ctypes
import threading
import time

_cursor_hidden = False  # keeps track of visibility state

def hide_cursor():
    """
    Hides mouse cursor using Windows API.
    Safe for 64bit systems.
    Does nothing if already hidden.
    """
    global _cursor_hidden
    if _cursor_hidden:  
        return  # already hidden — skip
    
    # ctypes.windll.user32.ShowCursor(0) → hide request
    # multiple calls reduce counter and ensures hiding
    for _ in range(10):
        ctypes.windll.user32.ShowCursor(False)

    _cursor_hidden = True


def show_cursor():
    """
    Makes cursor visible again.
    Called after calibration or cursor control mode ends.
    """
    global _cursor_hidden
    if not _cursor_hidden:
        return  # cursor already visible — skip

    for _ in range(10):
        ctypes.windll.user32.ShowCursor(True)

    _cursor_hidden = False
# ==========================================

try:
    import pygetwindow as gw
except Exception:
    gw = None

# ----------------------
# Configuration & paths
# ----------------------
OUT_CSV = Path("data/eye_landmarks.csv")
OUT_MAP = Path("data/gaze_map.json")
OUT_DRAG_JSON = Path("data/drag_samples.json")   # new: store drag samples for analysis
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_DRAG_JSON.parent.mkdir(parents=True, exist_ok=True)

ALIGN_IMAGE_PATH = Path("assets/align_reference.png")  # optional alignment image

# Calibration params
SAMPLES_PER_POINT = 30        # per-grid/corner/edge-point samples
SAMPLES_PER_DRAG = 30         # samples recorded while dragging center->edge
PRE_FIXATION_SEC = 0.8
TRANSITION_SEC = 0.8          # transition time for normal grid/corners
DRAG_TRANSITION_SEC = 2.2     # << SLOWER drag transitions (user requested) - tune as needed
GRID_POINTS = 16              # default 4x4
TARGET_COLOR = (0, 0, 255)
BACKGROUND_BRIGHTNESS = 255
TARGET_RADIUS = 30

# Camera preview
CAM_PREVIEW_SIZE = (360, 270)
PREVIEW_MARGIN = 20

# Face & lighting checks
MIN_FACE_AREA_PX = 3000
MIN_FACE_BRIGHTNESS = 60

# Cursor smoothing & heatmap
EMA_ALPHA = 0.18
MOVE_THRESHOLD_PX = 5
VIRTUAL_CURSOR_RADIUS = 60
HEATMAP_DECAY = 0.95
HEATMAP_SCALE = 4

# Jump behavior: if predicted cursor is farther than this, snap (fast move)
JUMP_RESET_PX = 200   # tune: 150-300

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# ----------------------
# MediaPipe initialization
# ----------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

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
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    lx, ly = avg_landmark(lm, LEFT_IRIS)
    rx, ry = avg_landmark(lm, RIGHT_IRIS)
    return lx, ly, rx, ry

def get_face_bbox_pixels(results, frame_w, frame_h):
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    minx = int(min(xs) * frame_w); maxx = int(max(xs) * frame_w)
    miny = int(min(ys) * frame_h); maxy = int(max(ys) * frame_h)
    minx = max(0, minx); miny = max(0, miny)
    maxx = min(frame_w-1, maxx); maxy = min(frame_h-1, maxy)
    return (minx, miny, maxx, maxy)

def face_present_and_bright(results, frame, min_area_px=MIN_FACE_AREA_PX, min_brightness=MIN_FACE_BRIGHTNESS):
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
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    present = (area >= min_area_px) and (mean_brightness >= min_brightness)
    return present, mean_brightness, bbox

# ----------------------
# OS window helpers
# ----------------------
def bring_window_to_front_windows(title, retries=8, delay=0.08):
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

def bring_window_pygetwindow(title, retries=8, delay=0.08):
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
    if sys.platform.startswith("win"):
        if bring_window_to_front_windows(title):
            return True
    return bring_window_pygetwindow(title)

# ----------------------
# Hide/show system cursor (Windows). No-op on others.
# ----------------------
def hide_system_cursor():
    if sys.platform.startswith("win"):
        # decrement ShowCursor until it returns <=0 to hide; store nothing (simple)
        ctypes.windll.user32.ShowCursor(False)

def show_system_cursor():
    if sys.platform.startswith("win"):
        ctypes.windll.user32.ShowCursor(True)

# ----------------------
# Mapping helpers (unchanged)
# ----------------------
def build_design_matrix(eye_xy):
    x = eye_xy[:,0]; y = eye_xy[:,1]
    return np.column_stack([x, y, x*y, x**2, y**2, np.ones_like(x)])

def fit_poly_mapping(data):
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
# Heatmap helpers (fixed broadcasting orientation)
# ----------------------
heatmap_h = max(1, SCREEN_H // HEATMAP_SCALE)
heatmap_w = max(1, SCREEN_W // HEATMAP_SCALE)
heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

def heatmap_add(px, py, strength=1.0, sigma=10.0):
    global heatmap, heatmap_w, heatmap_h

    cx = int(round(px / HEATMAP_SCALE))
    cy = int(round(py / HEATMAP_SCALE))
    if cx < 0 or cx >= heatmap_w or cy < 0 or cy >= heatmap_h:
        return

    sigma = max(1.0, float(sigma))
    half_size = int(min(max(3 * sigma, 3), max(heatmap_w, heatmap_h)))

    x0 = max(0, cx - half_size)
    x1 = min(heatmap_w - 1, cx + half_size)
    y0 = max(0, cy - half_size)
    y1 = min(heatmap_h - 1, cy + half_size)

    dx = np.arange(x0, x1 + 1) - cx
    dy = np.arange(y0, y1 + 1) - cy
    X, Y = np.meshgrid(dx, dy)
    gauss = np.exp(- (X.astype(np.float32)**2 + Y.astype(np.float32)**2) / (2.0 * (sigma**2)))
    heatmap[y0:y1+1, x0:x1+1] += (strength * gauss)
    np.clip(heatmap, 0.0, None, out=heatmap)

def heatmap_decay():
    global heatmap
    heatmap *= HEATMAP_DECAY
    np.clip(heatmap, 0.0, None, out=heatmap)

def heatmap_render():
    hm = heatmap.copy()
    if hm.max() > 0:
        hm = hm / (hm.max()) * 255.0
    hm = np.clip(hm, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(cv2.resize(hm, (SCREEN_W, SCREEN_H)), cv2.COLORMAP_JET)
    alpha = cv2.resize(hm, (SCREEN_W, SCREEN_H)).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 1)
    overlay = (hm_color.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    return overlay

# ----------------------
# UI helpers
# ----------------------
def make_fullscreen_canvas(color=BACKGROUND_BRIGHTNESS):
    return np.full((SCREEN_H, SCREEN_W, 3), color, dtype=np.uint8)

def draw_camera_inset(canvas, cam_frame):
    small = cv2.resize(cam_frame, CAM_PREVIEW_SIZE)
    h, w = CAM_PREVIEW_SIZE[1], CAM_PREVIEW_SIZE[0]
    x0, y0 = PREVIEW_MARGIN, PREVIEW_MARGIN
    canvas[y0:y0+h, x0:x0+w] = small
    cv2.rectangle(canvas, (x0-2, y0-2), (x0+w+2, y0+h+2), (200,200,200), 2)
    cv2.putText(canvas, "Camera preview", (x0, y0+h+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

def countdown_on_canvas(win_name, seconds=3, cam=None):
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
# Alignment mode (brings window front)
# ----------------------
def mode_alignment():
    cap = cv2.VideoCapture(0)
    win = "Alignment - position your head inside the reference"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 700)

    align_img = None
    if ALIGN_IMAGE_PATH.exists():
        align_img = cv2.imread(str(ALIGN_IMAGE_PATH), cv2.IMREAD_UNCHANGED)

    # show first frame quickly then bring to front (fixes "first camera opens in background")
    ret, frame = cap.read()
    if ret:
        preview = cv2.flip(frame, 1)
        cv2.imshow(win, preview)
        cv2.waitKey(1)
        bring_window_to_front(win)

    print("Alignment mode: Press 's' when aligned and face is green/ok (or 'q' to quit).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        present, brightness, bbox = face_present_and_bright(results, frame)
        preview = cv2.flip(frame, 1)
        overlay = preview.copy()
        if align_img is not None:
            scale = 0.6
            aw = int(preview.shape[1] * scale)
            ah = int(align_img.shape[0] * (aw / align_img.shape[1]))
            align_resized = cv2.resize(align_img, (aw, ah), interpolation=cv2.INTER_AREA)
            if align_resized.shape[2] == 4:
                alpha = align_resized[:, :, 3] / 255.0
                for c in range(3):
                    overlay[(preview.shape[0]//2 - ah//2):(preview.shape[0]//2 + ah//2),
                            (preview.shape[1]//2 - aw//2):(preview.shape[1]//2 + aw//2), c] = \
                        (alpha * align_resized[:, :, c] + (1-alpha) * overlay[(preview.shape[0]//2 - ah//2):(preview.shape[0]//2 + ah//2),
                                                                              (preview.shape[1]//2 - aw//2):(preview.shape[1]//2 + aw//2), c])
            else:
                y0 = preview.shape[0]//2 - ah//2
                x0 = preview.shape[1]//2 - aw//2
                overlay[y0:y0+ah, x0:x0+aw] = cv2.addWeighted(align_resized, 0.85, overlay[y0:y0+ah, x0:x0+aw], 0.15, 0)

        outline_color = (0, 255, 0) if present else (0, 0, 255)
        cv2.rectangle(overlay, (5, 5), (preview.shape[1]-5, preview.shape[0]-5), outline_color, 4)
        if present:
            cv2.putText(overlay, "Face detected. Press 's' to start calibration.", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(overlay, "Face not found or too dark. Adjust camera/lighting.", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            if bbox is not None:
                minx, miny, maxx, maxy = bbox
                pw = preview.shape[1]
                mx0 = pw - int(maxx / w * preview.shape[1])
                mx1 = pw - int(minx / w * preview.shape[1])
                my0 = int(miny / h * preview.shape[0])
                my1 = int(maxy / h * preview.shape[0])
                cv2.rectangle(overlay, (mx0, my0), (mx1, my1), (0,0,255), 2)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and present:
            break
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(win)

# ----------------------
# Calibration flow (updated)
# ----------------------
def mode_calibration(num_points=16):
    """
    Phase 1: 4x4 grid smooth sampling
    Phase 2: center->edge slow drags; save drag samples to JSON; after each drag sample at the edge point
    Phase 3: corners TL->TR->BR->BL
    """
    print("Starting calibration...")
    time.sleep(1)  # just delay 1 sec

    if num_points == 9:
        n = 3
    else:
        n = 4
        num_points = 16

    # grid targets (phase 1)
    xs = np.linspace(0.12, 0.88, n)
    ys = np.linspace(0.12, 0.88, n)
    grid_targets = [(int(SCREEN_W * x), int(SCREEN_H * y)) for y in ys for x in xs]

    # phase 2 edges from center
    center = (SCREEN_W // 2, SCREEN_H // 2)
    edge_targets = [
        ("left",  (int(SCREEN_W * 0.05), center[1])),
        ("right", (int(SCREEN_W * 0.95), center[1])),
        ("up",    (center[0], int(SCREEN_H * 0.05))),
        ("down",  (center[0], int(SCREEN_H * 0.95))),
    ]

    # phase 3 corners
    corners = [
        (int(SCREEN_W * 0.05), int(SCREEN_H * 0.05)),   # TL
        (int(SCREEN_W * 0.95), int(SCREEN_H * 0.05)),   # TR
        (int(SCREEN_W * 0.95), int(SCREEN_H * 0.95)),   # BR
        (int(SCREEN_W * 0.05), int(SCREEN_H * 0.95)),   # BL
    ]

    cap = cv2.VideoCapture(0)
    for _ in range(8):
        cap.read()

    win = "Calibration - look at the red dot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas_show = make_fullscreen_canvas(255)
    cv2.imshow(win, canvas_show); cv2.waitKey(1)
    bring_window_to_front(win)
    hide_system_cursor()

    countdown_on_canvas(win, seconds=3, cam=cap)

    collected = []  # all collected samples across phases
    drag_record_list = []  # will store dictionaries for JSON (phase 2)

    # Helper to draw arrow at center pointing to (tx,ty)
    def draw_center_arrow(canvas, tx, ty):
        cx, cy = center
        vecx = tx - cx; vecy = ty - cy
        norm = math.hypot(vecx, vecy)
        if norm < 1: norm = 1.0
        # arrow length limited to 300 px for visibility
        length = min(300, int(norm * 0.6) if norm > 0 else 100)
        ux = int(round(cx + (vecx / norm) * length))
        uy = int(round(cy + (vecy / norm) * length))
        cv2.arrowedLine(canvas, (cx, cy), (ux, uy), (0,0,255), 8, tipLength=0.25)

    # time-based animator; can optionally record while moving
    def animate_transition_and_maybe_sample(sx0, sy0, sx1, sy1,
                                           record_while=False, max_samples=0, instruction=None,
                                           drag_mode=False, edge_name=None):
        rec = []
        t_start = time.time()
        samples_taken = 0
        duration = DRAG_TRANSITION_SEC if drag_mode else TRANSITION_SEC
        while True:
            elapsed = time.time() - t_start
            alpha = min(1.0, elapsed / duration)
            px = int(round(sx0 + alpha * (sx1 - sx0)))
            py = int(round(sy0 + alpha * (sy1 - sy0)))
            canvas = make_fullscreen_canvas(255)
            rr = int(TARGET_RADIUS * (0.8 + 0.4 * (1 - abs(0.5 - alpha))))
            cv2.circle(canvas, (px, py), rr, TARGET_COLOR, -1)

            # if drag_mode and we are starting from center, draw arrow pointing to target
            if drag_mode and (sx0, sy0) == center:
                draw_center_arrow(canvas, sx1, sy1)
                if instruction:
                    cv2.putText(canvas, instruction, (30, SCREEN_H - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow(win, canvas)

            # capture frame and optionally record sample
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if record_while and samples_taken < max_samples:
                centers = get_iris_centers(results)
                if centers:
                    lx_n, ly_n, rx_n, ry_n = centers
                    ex = (lx_n + rx_n) / 2.0
                    ey = (ly_n + ry_n) / 2.0
                    ts = time.time()
                    rec.append({"timestamp": ts, "ex": float(ex), "ey": float(ey),
                                "px": int(px), "py": int(py), "phase": "drag", "edge": edge_name})
                    samples_taken += 1

            if alpha >= 1.0:
                # Ensure final exact endpoint frame is rendered (no off-by-one rounding mismatch)
                final_canvas = make_fullscreen_canvas(255)
                rr_final = int(TARGET_RADIUS * 0.6)
                cv2.circle(final_canvas, (sx1, sy1), rr_final, TARGET_COLOR, -1)
                # keep arrow on final frame if it was a center->edge drag
                if drag_mode and (sx0, sy0) == center:
                    draw_center_arrow(final_canvas, sx1, sy1)
                    if instruction:
                        cv2.putText(final_canvas, instruction, (30, SCREEN_H - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow(win, final_canvas)
                cv2.waitKey(1)
                # tiny hold so user sees dot landed exactly where sampling will start
                time.sleep(0.12)
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return rec, True
        return rec, False

    # -------------------------
    # Phase 1: grid sampling (smooth)
    # -------------------------
    for i, (tx, ty) in enumerate(grid_targets):
        if i == 0:
            cur_x, cur_y = center
        else:
            cur_x, cur_y = grid_targets[i-1]

        _, early_quit = animate_transition_and_maybe_sample(cur_x, cur_y, tx, ty, record_while=False)
        if early_quit:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

        # pre-sample check
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if present:
                canvas = make_fullscreen_canvas(255)
                cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
                cv2.putText(canvas, "Starting sample...", (30, SCREEN_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
                cv2.imshow(win, canvas)
                cv2.waitKey(300)
                break
            else:
                canvas = make_fullscreen_canvas(255)
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                              (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                              (0,0,255), 4)
                cv2.putText(canvas, "Face not detected or too dark. Adjust camera/lighting.", (30, SCREEN_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return
                time.sleep(0.05)

        # sample at grid point
        samples = 0
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                # pause and resume behavior
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
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
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
                continue

            centers = get_iris_centers(results)
            if centers:
                lx_n, ly_n, rx_n, ry_n = centers
                ex = (lx_n + rx_n) / 2.0
                ey = (ly_n + ry_n) / 2.0
                collected.append((ex, ey, tx, ty))
                samples += 1

            canvas = make_fullscreen_canvas(255)
            cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
            cv2.putText(canvas, f"Collecting {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

    # -------------------------
    # Phase 2: slower center -> edges drag recording + post-drag calibration point
    # -------------------------
    for edge_name, (etx, ety) in edge_targets:
        instruction = f"Now drag your head/gaze from CENTER to the {edge_name.upper()} edge. Follow the red dot."
        # animate center->edge while recording SAMPLES_PER_DRAG
        rec, early = animate_transition_and_maybe_sample(center[0], center[1], etx, ety,
                                                         record_while=True, max_samples=SAMPLES_PER_DRAG,
                                                         instruction=instruction,
                                                         drag_mode=True, edge_name=edge_name)
        if early:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

        # append recorded drag samples to main collected and also save to drag_record_list for JSON
        for d in rec:
            collected.append((d["ex"], d["ey"], d["px"], d["py"]))
        drag_record_list.extend(rec)

        # Immediately begin sampling at the exact endpoint (etx,ety) — no extra pop
        samples = 0
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                # pause-resume as before
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
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
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
                continue

            centers = get_iris_centers(results)
            if centers:
                lx_n, ly_n, rx_n, ry_n = centers
                ex = (lx_n + rx_n) / 2.0
                ey = (ly_n + ry_n) / 2.0
                collected.append((ex, ey, etx, ety))
                samples += 1

            canvas = make_fullscreen_canvas(255)
            # draw the exact same endpoint we used in the animator so sampling looks continuous
            cv2.circle(canvas, (etx, ety), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
            cv2.putText(canvas, f"Edge-sampling {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

    # Save drag samples JSON for analysis
    try:
        existing = []
        if OUT_DRAG_JSON.exists():
            with open(OUT_DRAG_JSON, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        existing.extend(drag_record_list)
        with open(OUT_DRAG_JSON, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Saved {len(drag_record_list)} drag samples to {OUT_DRAG_JSON}")
    except Exception as e:
        print("Warning: failed to save drag JSON:", e)

    # -------------------------
    # Phase 3: corners sampling
    # -------------------------
    for i, (tx, ty) in enumerate(corners):
        if i == 0:
            start_x, start_y = center
        else:
            start_x, start_y = corners[i-1]
        _, early_quit = animate_transition_and_maybe_sample(start_x, start_y, tx, ty, record_while=False)
        if early_quit:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

        # pre-sample ensure face ok
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if present:
                canvas = make_fullscreen_canvas(255)
                cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
                cv2.putText(canvas, "Starting corner sample...", (30, SCREEN_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
                cv2.imshow(win, canvas)
                cv2.waitKey(300)
                break
            else:
                canvas = make_fullscreen_canvas(255)
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                              (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                              (0,0,255), 4)
                cv2.putText(canvas, "Face not detected or too dark. Adjust camera/lighting.", (30, SCREEN_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return
                time.sleep(0.05)

        # sample corner
        samples = 0
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                # pause-resume as before
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
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
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
                continue

            centers = get_iris_centers(results)
            if centers:
                lx_n, ly_n, rx_n, ry_n = centers
                ex = (lx_n + rx_n) / 2.0
                ey = (ly_n + ry_n) / 2.0
                collected.append((ex, ey, tx, ty))
                samples += 1

            canvas = make_fullscreen_canvas(255)
            cv2.circle(canvas, (tx, ty), int(TARGET_RADIUS * 0.6), TARGET_COLOR, -1)
            cv2.putText(canvas, f"Collecting corner {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

    # End calibration: fit mapping
    cap.release()
    show_system_cursor()
    cv2.destroyAllWindows()

    if len(collected) < 6:
        print("Not enough samples collected.")
        return
    wx, wy = fit_poly_mapping(collected)

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
# Cursor control (snappy + heatmap)
# ----------------------
def mode_cursor_control():
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
    cv2.imshow(win, make_fullscreen_canvas(0)); cv2.waitKey(1); bring_window_to_front(win)

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

                if ema_x is None:
                    ema_x, ema_y = px, py
                else:
                    dist = math.hypot(px - ema_x, py - ema_y)
                    if dist > JUMP_RESET_PX:
                        ema_x, ema_y = px, py
                    else:
                        dynamic_alpha = EMA_ALPHA + min(0.8, (dist / max(SCREEN_W, SCREEN_H)) * 3.0)
                        dynamic_alpha = min(1.0, dynamic_alpha)
                        ema_x = dynamic_alpha * px + (1 - dynamic_alpha) * ema_x
                        ema_y = dynamic_alpha * py + (1 - dynamic_alpha) * ema_y

                curx, cury = pyautogui.position()
                dx = abs(ema_x - curx); dy = abs(ema_y - cury)
                if dx > MOVE_THRESHOLD_PX or dy > MOVE_THRESHOLD_PX:
                    pyautogui.moveTo(int(ema_x), int(ema_y))

                heatmap_add(ema_x, ema_y, strength=1.0, sigma=8.0)

            heatmap_decay()

            canvas = make_fullscreen_canvas(0)
            hm_img = heatmap_render()
            canvas = cv2.addWeighted(canvas, 1.0, hm_img, 0.7, 0)
            if ema_x is not None:
                overlay = canvas.copy()
                cv2.circle(overlay, (int(ema_x), int(ema_y)), VIRTUAL_CURSOR_RADIUS, (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)
                cv2.circle(canvas, (int(ema_x), int(ema_y)), VIRTUAL_CURSOR_RADIUS, (255, 255, 255), 2)
            cv2.putText(canvas, "Gaze overlay. Press 'q' to quit.", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
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
    2 -> Calibration (smooth 4x4 + slow drags + edge sampling + corners)
    3 -> Cursor control (overlay + OS cursor moves)
    q -> Quit
    """)
    while True:
        cmd = input("Enter mode: ").strip().lower()
        if cmd == "1":
            hide_cursor()
            try:
                mode_alignment()
            finally:
                show_cursor()
        elif cmd == "2":
            hide_cursor()
            try:
                mode_calibration(num_points=3*3 if GRID_POINTS == 9 else 16)
            finally:
                show_cursor()
        elif cmd == "3":
            hide_cursor()
            try:
                mode_cursor_control()
            finally:
                show_cursor()
        elif cmd == "q":
            show_cursor()  # extra safety
            break
        else:
            print("Invalid. Choose 1,2,3 or q.")