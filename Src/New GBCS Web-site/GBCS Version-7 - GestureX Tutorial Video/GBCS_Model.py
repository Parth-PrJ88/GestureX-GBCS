# src/gaze_prototype_with_alignment_4x4_with_drags_fixed_endpoint.py
# Requirements:
# pip install mediapipe opencv-python numpy pyautogui scipy PyGetWindow keyboard
#
# Save alignment reference image (optional) at: assets/align_reference.png

import sys, time, math, ctypes
import threading
import concurrent.futures
from pathlib import Path
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import csv
from scipy.stats import pearsonr
import keyboard

# True when launched from Flask
# -------------------------------------------
WEB_MODE = len(sys.argv) > 1

try:
    import pygetwindow as gw
except Exception:
    gw = None


# --------------------------------------------
# Write JSON file for State Management
# --------------------------------------------

DATA_DIR = Path("data")
PROGRESS_FILE = DATA_DIR / "progress.json"
STOP_FILE = DATA_DIR / "stop.json"

def update_progress(
    phase,
    progress,
    message,
    running=True,
    completed=False
):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "phase": phase,
                "progress": progress,
                "message": message,
                "running": running,
                "completed": completed
            },
            f,
            indent=4
        )

# Reset Workflow
# --------------------------------------------
def reset_workflow(message="Workflow cancelled."):
    update_progress(
        "Idle",
        0,
        message,
        running=False,
        completed=False
    )

def cursor_stopped():
    update_progress(
        "Calibration",
        66,
        "Cursor Control stopped by user. Ready to restart.",
        running=False,
    )

def failsafe_triggered():
    update_progress(
        "Calibration",
        66,
        "Failsafe Triggered. Please Recalibrate Before Starting Cursor Control again to Avoid this, because the Current Calibration may Not Accurate.",
        running=False,
        completed=False
    )

# Workflow Stop Control
# --------------------------------------------
STOP_FILE = Path("data/stop.json")

def request_stop():
    with open(STOP_FILE, "w") as f:
        json.dump({"stop": True}, f)

def clear_stop():
    with open(STOP_FILE, "w") as f:
        json.dump({"stop": False}, f)

def should_stop():
    try:
        with open(STOP_FILE, "r") as f:
            return json.load(f).get("stop", False)
    except:
        return False


# ----------------------
# Configuration & paths
# ----------------------
OUT_CSV = Path("data/eye_landmarks.csv")
OUT_MAP = Path("data/gaze_map.json")
OUT_DRAG_JSON = Path("data/drag_samples.json")
PROFILES_DIR = Path("data/profiles")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_DRAG_JSON.parent.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

ALIGN_IMAGE_PATH = Path("assets/align_reference.png")

# Calibration params
SAMPLES_PER_POINT = 30
SAMPLES_PER_DRAG = 30
PRE_FIXATION_SEC = 0.8
TRANSITION_SEC = 0.8
DRAG_TRANSITION_SEC = 2.2
GRID_POINTS = 16
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

# Jump behaviour
JUMP_RESET_PX = 200

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# Wink / gesture parameters
EYE_ASPECT_RATIO_THRESHOLD = 0.22
WINK_MIN_DURATION_MS = 1500  # 1.5 seconds required for a wink to count as click
WINK_COOLDOWN_MS = 800      # cooldown between repeated wink clicks
DOUBLE_CLICK_HOLD_MS = 800  # time holding left wink required to trigger a double click

# Wink robustness: smoothing + hysteresis to prevent corner/edge failures
EAR_EMA_ALPHA = 0.35
WINK_RESET_FRAMES = 5

# Pinch hold robustness: smoothing + hysteresis
PINCH_DIST_EMA_ALPHA = 0.4   
PINCH_RELEASE_FRAMES = 8     

# Global state for active profile tracking
ACTIVE_PROFILE_NAME = "default"

# ----------------------
# MediaPipe init
# ----------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# Initialize Hand tracking for Drag functionality
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ----------------------
# Helpers
# ----------------------
def ease_in_out_cubic(t):
    """Provides a smooth acceleration and deceleration curve for animations."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - math.pow(-2 * t + 2, 3) / 2

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

def calculate_ear(landmarks, indices):
    p_left = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p_top = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p_bottom = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p_right = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])

    v_dist = np.linalg.norm(p_top - p_bottom)
    h_dist = np.linalg.norm(p_left - p_right)

    if h_dist == 0: return 0.0
    return v_dist / h_dist

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
# Window helpers
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

def hide_system_cursor():
    if sys.platform.startswith("win"):
        ctypes.windll.user32.ShowCursor(False)

def show_system_cursor():
    if sys.platform.startswith("win"):
        ctypes.windll.user32.ShowCursor(True)

def disable_windows_power_throttling():
    if not sys.platform.startswith("win"):
        return
    try:
        kernel32 = ctypes.windll.kernel32

        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.SetProcessInformation.restype = ctypes.c_int
        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong
        ]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.GetLastError.restype = ctypes.c_ulong

        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4  

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        state = PROCESS_POWER_THROTTLING_STATE(
            1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0  
        )

        handle = kernel32.GetCurrentProcess()

        ok1 = kernel32.SetProcessInformation(
            handle, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)
        )
        err1 = 0 if ok1 else kernel32.GetLastError()

        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        ok2 = kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)
        err2 = 0 if ok2 else kernel32.GetLastError()

        print(f"[Power Throttling] SetProcessInformation ok={bool(ok1)} (err={err1}) | "
              f"SetPriorityClass ok={bool(ok2)} (err={err2})")
    except Exception as e:
        print(f"[WARN] Could not disable Windows power throttling: {e}")

# ------------------------------------------------------------------
# Mapping & Profile-Management helpers
# ------------------------------------------------------------------

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

def save_mapping(wx, wy, invert_x=False, invert_y=False, profile_name="default"):
    mapping = {
        "profile_name": profile_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wx": wx.tolist(), 
        "wy": wy.tolist(),
        "screen": [SCREEN_W, SCREEN_H],
        "invert_x": bool(invert_x), 
        "invert_y": bool(invert_y)
    }
    
    with open(OUT_MAP, "w") as f:
        json.dump(mapping, f, indent=4)

    print("\n[SUCCESS] Calibration saved to active runtime.")

def load_mapping(profile_filename=None):
    if profile_filename:
        target_path = PROFILES_DIR / f"{profile_filename}.json"
        if target_path.exists():
            return json.load(open(target_path))
    
    if OUT_MAP.exists():
        return json.load(open(OUT_MAP))
    return None


# Profile-Management For CMD
# ================================================================================================================

def interactive_profile_menu():
    global ACTIVE_PROFILE_NAME
    while True:
        print("\n=== CALIBRATION NODE PROFILE MANAGEMENT ===")
        profiles = list(PROFILES_DIR.glob("*.json"))
        
        current_loaded = "None"
        if OUT_MAP.exists():
            try:
                with open(OUT_MAP, "r") as f:
                    data = json.load(f)
                    current_loaded = f"{data.get('profile_name', 'default')} ({data.get('timestamp', 'Unknown')})"
            except Exception:
                current_loaded = "gaze_map.json (Active cache)"

        print(f"Current System Active Profile: {current_loaded}")
        print("-" * 43)
        
        if not profiles:
            print("No saved profile calibration nodes found inside 'data/profiles/'.")
        else:
            print("Available nodes:")
            for i, p_path in enumerate(profiles, start=1):
                try:
                    with open(p_path, "r") as f:
                        meta = json.load(f)
                    print(f"  {i} -> {meta.get('profile_name', p_path.stem)} [{meta.get('timestamp', '')}]")
                except Exception:
                    print(f"  {i} -> {p_path.stem} (Raw JSON Mapping Error)")

        print("\nOptions:")
        print("  [number] -> Load selected profile node configuration")
        print("  m        -> Return to Master Menu")
        
        choice = input("\nSelect an action: ").strip()
        if choice.lower() == 'm':
            break
            
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                selected_profile = profiles[idx]
                try:
                    with open(selected_profile, "r") as sf:
                        profile_data = json.load(sf)
                    with open(OUT_MAP, "w") as df:
                        json.dump(profile_data, df, indent=4)
                    
                    ACTIVE_PROFILE_NAME = profile_data.get("profile_name", selected_profile.stem)
                    print(f"\n[LOADED] System active profile switched to: '{ACTIVE_PROFILE_NAME}'")
                    break
                except Exception as e:
                    print(f"Error swapping node profile content maps: {e}")
            else:
                print("Index out of selection range.")
        else:
            print("Invalid Option selection.")


# Profile-Management For Web
# ================================================================================================================

def get_profiles():
    profiles = []
    if not PROFILES_DIR.exists():
        return profiles

    for profile_file in PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_file, "r") as f:
                data = json.load(f)
            profiles.append({
                "name": data.get("profile_name", profile_file.stem),
                "timestamp": data.get("timestamp", ""),
                "filename": profile_file.stem
            })
        except Exception:
            print(f"Skipping invalid profile: {profile_file.name}")

    profiles.sort(key=lambda x: x["timestamp"], reverse=True)
    return profiles

def load_profile(filename):
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            workflow = json.load(f)
        if workflow.get("progress", 0) < 33:
            return False, "Please Complete Alignment before loading a Profile."
    
    global ACTIVE_PROFILE_NAME
    profile_path = PROFILES_DIR / f"{filename}.json"
    if not profile_path.exists():
        return False, "Profile not found."

    try:
        with open(profile_path, "r") as f:
            profile_data = json.load(f)

        with open(OUT_MAP, "w") as f:
            json.dump(profile_data, f, indent=4)

        ACTIVE_PROFILE_NAME = profile_data.get("profile_name", filename)

        update_progress(
            phase = "Calibration",
            progress = 66,
            message = "Profile loaded Successfully. Ready for Cursor Control.",
            running = False,
            completed = False
        )
        return True, ACTIVE_PROFILE_NAME
    except Exception as e:
        return False, str(e)

def save_current_profile(profile_name):
    if not profile_name:
        profile_name = "Default"
    if not OUT_MAP.exists():
        return False, "No Calibration Found."
   
    try:
        with open(OUT_MAP, "r") as f:
            profile = json.load(f)

        if "wx" not in profile or "wy" not in profile:
            return False, "No Valid Calibration Found."

        profile["profile_name"] = profile_name
        profile["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_filename = profile_name.lower().replace(" ", "_")
        profile_path = PROFILES_DIR / f"{safe_filename}.json"

        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=4)

        with open(OUT_MAP, "w") as f:
            json.dump(profile, f, indent=4)

        return True, "Profile: " + profile_name + " Saved Successfully."
    except Exception as e:
        return False, str(e)

def delete_profile(filename):
    profile_path = PROFILES_DIR / f"{filename}.json"
    if not profile_path.exists():
        return False, "Profile Not Found."

    try:
        with open(profile_path, "r") as f:
            profile_data = json.load(f)

        active_name = ""
        if OUT_MAP.exists():
            with open(OUT_MAP, "r") as f:
                active = json.load(f)
                active_name = active.get("profile_name", "")

        profile_path.unlink()

        if active_name.lower() == profile_data.get("profile_name", "").lower():
            active["profile_name"] = "default"
            with open(OUT_MAP, "w") as f:
                json.dump(active, f, indent=4)

        return True, "Profile Deleted."
    except Exception as e:
        return False, str(e)


# ----------------------
# Heatmap helpers
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
    cv2.rectangle(canvas, (x0-2, y0-2), (x0+w+2, y0+h+2), (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Camera preview", (x0, y0+h+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

def draw_text_with_shadow(canvas, text, position, font, scale, color, thickness):
    x, y = position
    cv2.putText(canvas, text, (x + 2, y + 2), font, scale, (150, 150, 150), thickness, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

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
        
        if cam is not None:
            ret, frame = cam.read()
            if ret:
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                
        draw_text_with_shadow(canvas, txt, (x, y), font, 6*scale, TARGET_COLOR, thickness)
        
        cv2.imshow(win_name, canvas)
        cv2.waitKey(1)
        time.sleep(1)

def draw_plus_icon(canvas, cx, cy, plus_size, color=(255,255,255), thickness=2):
    x0 = int(round(cx - plus_size))
    x1 = int(round(cx + plus_size))
    y0 = int(round(cy - plus_size))
    y1 = int(round(cy + plus_size))
    cv2.line(canvas, (x0, cy), (x1, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, y0), (cx, y1), color, thickness, cv2.LINE_AA)

def draw_target_with_plus(canvas, x, y, radius, show_plus=False):
    cv2.circle(canvas, (int(x), int(y)), max(1, int(round(radius * 1.15))), (200, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(x), int(y)), max(1, int(round(radius))), TARGET_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(x), int(y)), max(1, int(round(radius * 0.15))), (255, 255, 255), -1, cv2.LINE_AA)

    if show_plus:
        plus_half_len = max(1, int(round(radius * 0.45)))
        thickness = max(2, int(round(radius * 0.12)))
        draw_plus_icon(canvas, int(x), int(y), plus_half_len, color=(255,255,255), thickness=thickness)

# ----------------------
# Alignment mode
# ----------------------
def mode_alignment():
    clear_stop()
    update_progress(
        "Alignment",
        10,
        "Initializing camera and alignment...",
    )

    cap = cv2.VideoCapture(0)
    win = "Alignment - position your head inside the reference"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 700)

    align_img = None
    if ALIGN_IMAGE_PATH.exists():
        align_img = cv2.imread(str(ALIGN_IMAGE_PATH), cv2.IMREAD_UNCHANGED)

    ret, frame = cap.read()
    if ret:
        preview = cv2.flip(frame, 1)
        cv2.imshow(win, preview)
        cv2.waitKey(1)
        bring_window_to_front(win)

    print("Alignment mode: Press 's' when aligned and face is green/ok (or 'q' to quit).")

    while True:
        if should_stop():
            cap.release()
            cv2.destroyAllWindows()
            reset_workflow("Alignment Cancelled.")
            return

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
        cv2.rectangle(overlay, (5, 5), (preview.shape[1]-5, preview.shape[0]-5), outline_color, 4, cv2.LINE_AA)
        if present:
            cv2.putText(overlay, "Face Detected. Background is Clear.", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(overlay, "Press 's' On Window to Start Calibration.", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "Face Not Found or Too Dark.", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "Adjust Camera/Lighting.", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
            
            if bbox is not None:
                minx, miny, maxx, maxy = bbox
                pw = preview.shape[1]
                mx0 = pw - int(maxx / w * preview.shape[1])
                mx1 = pw - int(minx / w * preview.shape[1])
                my0 = int(miny / h * preview.shape[0])
                my1 = int(maxy / h * preview.shape[0])
                cv2.rectangle(overlay, (mx0, my0), (mx1, my1), (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')) and present:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            reset_workflow("Alignment Cancelled.")
            return
        if key == ord('q') :
            cap.release()
            cv2.destroyWindow(win)
            reset_workflow("Alignment Cancelled.")
            return

    update_progress(
        "Alignment",
        33,
        "Alignment completed. Ready for Calibration.",
        running=False,
        completed=False
    )

    cap.release()
    cv2.destroyWindow(win)

# ----------------------
# Calibration flow
# ----------------------
def mode_calibration(num_points=16):
    clear_stop()
    update_progress(
        "Calibration",
        40,
        "Calibration started. New window will open then Please follow the targets..."
    )

    print("Starting calibration...")

    if not WEB_MODE:
        print("Make sure you're aligned.")
        input("Press Enter to start (or Ctrl+C to cancel)...")
    
    if num_points == 9:
        n = 3
    else:
        n = 4
        num_points = 16

    xs = np.linspace(0.12, 0.88, n)
    ys = np.linspace(0.12, 0.88, n)
    grid_targets = [(int(SCREEN_W * x), int(SCREEN_H * y)) for y in ys for x in xs]

    center = (SCREEN_W // 2, SCREEN_H // 2)
    edge_targets = [
        ("left",  (int(SCREEN_W * 0.05), center[1])),
        ("right", (int(SCREEN_W * 0.95), center[1])),
        ("up",    (center[0], int(SCREEN_H * 0.05))),
        ("down",  (center[0], int(SCREEN_H * 0.95))),
    ]

    corners = [
        (int(SCREEN_W * 0.05), int(SCREEN_H * 0.05)),
        (int(SCREEN_W * 0.95), int(SCREEN_H * 0.05)),
        (int(SCREEN_W * 0.95), int(SCREEN_H * 0.95)),
        (int(SCREEN_W * 0.05), int(SCREEN_H * 0.95)),
    ]

    cap = cv2.VideoCapture(0)
    for _ in range(8):
        cap.read()

    win = "Calibration - look at the red dot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, make_fullscreen_canvas(255)); cv2.waitKey(1)
    bring_window_to_front(win)
    hide_system_cursor()

    countdown_on_canvas(win, seconds=3, cam=cap)

    def show_center_message(text, secs=3):
        t0 = time.time()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(SCREEN_W, SCREEN_H) / 1600.0 * 2.0
        thickness = max(2, int(3 * scale))
        while time.time() - t0 < secs:
            ret, frame = cap.read()
            canvas = make_fullscreen_canvas(255)
            if ret:
                draw_camera_inset(canvas, cv2.flip(frame, 1))
            lines = []
            max_width = int(SCREEN_W * 0.8)
            words = str(text).split()
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                (tw, th), _ = cv2.getTextSize(test, font, fontScale=2.0*scale, thickness=thickness)
                if tw > max_width and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = test
            if cur:
                lines.append(cur)
            total_h = len(lines) * int(40 * scale)
            y0 = (SCREEN_H // 2) - (total_h // 2)
            for i, ln in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(ln, font, fontScale=2.0*scale, thickness=thickness)
                x = (SCREEN_W - tw) // 2
                y = int(y0 + i * 40 * scale + th)
                cv2.putText(canvas, ln, (x, y), font, 2.0*scale, (0,0,255), thickness, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                reset_workflow("Calibration cancelled.")
                return True
        return False

    if show_center_message("Follow the dots", secs=3):
        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

    collected = []
    drag_record_list = []

    def animate_transition_and_maybe_sample(sx0, sy0, sx1, sy1,
                                           record_while=False, max_samples=0, instruction=None,
                                           drag_mode=False, edge_name=None):
        rec = []
        t_start = time.time()
        samples_taken = 0
        duration = DRAG_TRANSITION_SEC if drag_mode else TRANSITION_SEC
        peak_rr = 0.0
        
        while True:
            if should_stop():
                return rec, True, float(TARGET_RADIUS), float(peak_rr)
            
            elapsed = time.time() - t_start
            raw_alpha = min(1.0, elapsed / duration)
            alpha = ease_in_out_cubic(raw_alpha)
            
            px = int(round(sx0 + alpha * (sx1 - sx0)))
            py = int(round(sy0 + alpha * (sy1 - sy0)))
            canvas = make_fullscreen_canvas(255)
            
            rr = TARGET_RADIUS * (0.8 + 0.4 * (1 - abs(0.5 - raw_alpha)))
            if rr > peak_rr:
                peak_rr = rr
            rr_i = int(round(rr))
            
            draw_target_with_plus(canvas, px, py, rr_i, show_plus=True)
            cv2.imshow(win, canvas)

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

            if raw_alpha >= 1.0:
                rr_final = TARGET_RADIUS
                final_canvas = make_fullscreen_canvas(255)
                draw_target_with_plus(final_canvas, sx1, sy1, int(round(rr_final)), show_plus=True)
                cv2.imshow(win, final_canvas)
                cv2.waitKey(1)
                time.sleep(0.12)
                return rec, False, float(rr_final), float(peak_rr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                reset_workflow("Calibration cancelled.")
                return rec, True, float(TARGET_RADIUS * 0.6), float(peak_rr)

    # Phase 1: grid sampling
    for i, (tx, ty) in enumerate(grid_targets):
        if should_stop():
            cap.release()
            show_system_cursor()
            cv2.destroyAllWindows()
            reset_workflow("Calibration stopped by User.")
            return

        if i == 0:
            cur_x, cur_y = center
        else:
            cur_x, cur_y = grid_targets[i-1]

        rec, early_quit, end_radius, peak_radius = animate_transition_and_maybe_sample(cur_x, cur_y, tx, ty, record_while=False)
        if early_quit:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
            reset_workflow("Calibration cancelled by User.")      
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if present:
                displayed_radius = float(peak_radius if peak_radius and peak_radius > 0 else end_radius)
                canvas = make_fullscreen_canvas(255)
                draw_target_with_plus(canvas, tx, ty, displayed_radius, show_plus=True)
                cv2.putText(canvas, "Starting sample...", (30, SCREEN_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                cv2.waitKey(300)
                break
            else:
                canvas = make_fullscreen_canvas(255)
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                              (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                              (0,0,255), 4, cv2.LINE_AA)
                cv2.putText(canvas, "Face Not Detected or Too Dark. Adjust Camera/Lighting.", (30, SCREEN_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                    reset_workflow("Calibration cancelled by User.") 
                    return
                time.sleep(0.05)

        samples = 0
        try:
            displayed_radius
        except NameError:
            displayed_radius = float(TARGET_RADIUS)
        sample_target_radius = TARGET_RADIUS * 0.6
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    canvas = make_fullscreen_canvas(255)
                    draw_camera_inset(canvas, cv2.flip(frame2, 1))
                    cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                  (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                  (0,0,255), 4, cv2.LINE_AA)
                    cv2.putText(canvas, "Paused: Face Lost or Too Dark. Fix and Wait...", (30, SCREEN_H - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                        reset_workflow("Calibration cancelled by User.") 
                        return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
                        ok_canvas = make_fullscreen_canvas(255)
                        draw_camera_inset(ok_canvas, cv2.flip(frame2, 1))
                        cv2.rectangle(ok_canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                      (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                      (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(ok_canvas, "Fixed. Resuming in 0.7s...", (30, SCREEN_H - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
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

            displayed_radius += (sample_target_radius - displayed_radius) * 0.12

            canvas = make_fullscreen_canvas(255)
            draw_target_with_plus(canvas, tx, ty, displayed_radius, show_plus=True)
            cv2.putText(canvas, f"Collecting {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                reset_workflow("Calibration cancelled by User.") 
                return

    # Phase 2: drags + edge sampling
    phase2_instruction = "Drag Head with the Pointer."
    if show_center_message(phase2_instruction, secs=3):
        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

    for edge_name, (etx, ety) in edge_targets:
        rec, early, end_radius, peak_radius = animate_transition_and_maybe_sample(center[0], center[1], etx, ety,
                                                                                   record_while=True, max_samples=SAMPLES_PER_DRAG,
                                                                                   instruction=None, drag_mode=True, edge_name=edge_name)
        
        if should_stop():
            cap.release()
            show_system_cursor()
            cv2.destroyAllWindows()
            reset_workflow("Calibration stopped by User.")
            return
        
        if early:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); return

        for d in rec:
            collected.append((d["ex"], d["ey"], d["px"], d["py"]))
        drag_record_list.extend(rec)

        samples = 0
        displayed_radius = float(peak_radius if peak_radius and peak_radius > 0 else end_radius)
        sample_target_radius = TARGET_RADIUS * 0.6
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    canvas = make_fullscreen_canvas(255)
                    draw_camera_inset(canvas, cv2.flip(frame2, 1))
                    cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                  (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                  (0,0,255), 4, cv2.LINE_AA)
                    cv2.putText(canvas, "Paused: Face Lost or Too Dark. Fix and Wait...", (30, SCREEN_H - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                        reset_workflow("Calibration cancelled by User.") 
                        return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
                        ok_canvas = make_fullscreen_canvas(255)
                        draw_camera_inset(ok_canvas, cv2.flip(frame2, 1))
                        cv2.rectangle(ok_canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                      (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                      (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(ok_canvas, "Fixed. Resuming in 0.7s...", (30, SCREEN_H - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
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

            displayed_radius += (sample_target_radius - displayed_radius) * 0.12

            canvas = make_fullscreen_canvas(255)
            draw_target_with_plus(canvas, etx, ety, displayed_radius, show_plus=True)
            cv2.putText(canvas, f"Edge-Sampling {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                reset_workflow("Calibration cancelled by User.")
                return

        return_duration = 0.6
        t_start = time.time()
        while True:
            elapsed = time.time() - t_start
            raw_alpha = min(1.0, elapsed / return_duration)
            alpha = ease_in_out_cubic(raw_alpha)
            
            px = int(round(etx + alpha * (center[0] - etx)))
            py = int(round(ety + alpha * (center[1] - ety)))
            canvas = make_fullscreen_canvas(255)
            rr = TARGET_RADIUS * (0.8 + 0.4 * (1 - abs(0.5 - raw_alpha)))
            rr_i = int(round(rr))
            draw_target_with_plus(canvas, px, py, rr_i, show_plus=True)
            cv2.imshow(win, canvas)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if raw_alpha >= 1.0:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                reset_workflow("Calibration cancelled by User.") 
                return
            time.sleep(0.01)

    try:
        existing = []
        if OUT_DRAG_JSON.exists():
            with open(OUT_DRAG_JSON, "r") as f:
                try: existing = json.load(f)
                except Exception: existing = []
        existing.extend(drag_record_list)
        with open(OUT_DRAG_JSON, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print("Warning: Failed to Save drag JSON:", e)

    # Phase 3: Corners sampling
    for i, (tx, ty) in enumerate(corners):
        if should_stop():
            cap.release()
            show_system_cursor()
            cv2.destroyAllWindows()
            reset_workflow("Calibration stopped by User.")
            return

        if i == 0:
            start_x, start_y = center
        else:
            start_x, start_y = corners[i-1]
        rec, early_quit, end_radius, peak_radius = animate_transition_and_maybe_sample(start_x, start_y, tx, ty, record_while=False)
        if early_quit:
            cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
            reset_workflow("Calibration cancelled by User.")  
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if present:
                displayed_radius = float(peak_radius if peak_radius and peak_radius > 0 else end_radius)
                canvas = make_fullscreen_canvas(255)
                draw_target_with_plus(canvas, tx, ty, displayed_radius, show_plus=True)
                cv2.putText(canvas, "Starting corner sample...", (30, SCREEN_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                cv2.waitKey(300)
                break
            else:
                canvas = make_fullscreen_canvas(255)
                draw_camera_inset(canvas, cv2.flip(frame, 1))
                cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                              (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                              (0,0,255), 4, cv2.LINE_AA)
                cv2.putText(canvas, "Face not detected or too Dark. Adjust camera/lighting.", (30, SCREEN_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                    reset_workflow("Calibration cancelled by User.")  
                    return
                time.sleep(0.05)

        samples = 0
        try:
            displayed_radius
        except NameError:
            displayed_radius = float(TARGET_RADIUS)
        sample_target_radius = TARGET_RADIUS * 0.6
        while samples < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            present, brightness, bbox = face_present_and_bright(results, frame)
            if not present:
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    canvas = make_fullscreen_canvas(255)
                    draw_camera_inset(canvas, cv2.flip(frame2, 1))
                    cv2.rectangle(canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                  (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                  (0,0,255), 4, cv2.LINE_AA)
                    cv2.putText(canvas, "Paused: Face Lost or Too Dark. Fix and Wait...", (30, SCREEN_H - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                        reset_workflow("Calibration cancelled by User.")  
                        return
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = face_mesh.process(rgb2)
                    present2, brightness2, bbox2 = face_present_and_bright(res2, frame2)
                    if present2:
                        ok_canvas = make_fullscreen_canvas(255)
                        draw_camera_inset(ok_canvas, cv2.flip(frame2, 1))
                        cv2.rectangle(ok_canvas, (PREVIEW_MARGIN-4, PREVIEW_MARGIN-4),
                                      (PREVIEW_MARGIN+CAM_PREVIEW_SIZE[0]+4, PREVIEW_MARGIN+CAM_PREVIEW_SIZE[1]+4),
                                      (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(ok_canvas, "Fixed. Resuming in 0.7s...", (30, SCREEN_H - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
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

            displayed_radius += (sample_target_radius - displayed_radius) * 0.12

            canvas = make_fullscreen_canvas(255)
            draw_target_with_plus(canvas, tx, ty, displayed_radius, show_plus=True)
            cv2.putText(canvas, f"Collecting corner {samples}/{SAMPLES_PER_POINT}", (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); show_system_cursor(); cv2.destroyAllWindows(); 
                reset_workflow("Calibration cancelled by User.")  
                return

    update_progress(
        "Calibration",
        66,
        "Calibration completed. Ready for Cursor Control.",
        running=False,
        completed=False
    )

    cap.release()
    show_system_cursor()
    cv2.destroyAllWindows()

    if len(collected) < 6:
        print("Not Enough Samples Collected.")
        return
    wx, wy = fit_poly_mapping(collected)

    print(f"lx_n={lx_n:.4f} rx_n={rx_n:.4f} ly_n={ly_n:.4f} ry_n={ry_n:.4f}")
    print(f"avgX={(lx_n+rx_n)/2:.4f} avgY={(ly_n+ry_n)/2:.4f}")

    pred_x = np.array([predict_poly(wx, wy, ex, ey)[0] for ex, ey, _, _ in collected])
    pred_y = np.array([predict_poly(wx, wy, ex, ey)[1] for ex, ey, _, _ in collected])
    true_x = np.array([r[2] for r in collected])
    true_y = np.array([r[3] for r in collected])

    invert_x = invert_y = False
    try:
        cx, _ = pearsonr(pred_x, true_x)
        cy, _ = pearsonr(pred_y, true_y)

        print(f"Correlation X: {cx:.4f}")
        print(f"Correlation Y: {cy:.4f}")
        
        if cx < 0: invert_x = True
        if cy < 0: invert_y = True
    except Exception:
        pass

    print(f"Invert X: {invert_x} | Invert Y: {invert_y}")

    print("\n" + "="*40)
    if WEB_MODE:
        user_node_name = "Default"
    else:
        user_node_name = input("Enter a name for this calibration profile [Or press Enter for 'Default']: ").strip()
        if not user_node_name:
            user_node_name = "Default"
        
    save_mapping(wx, wy, invert_x=invert_x, invert_y=invert_y, profile_name=user_node_name)
    
    errors = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
    print("--------------------------------")
    print("Mean Error :", errors.mean())
    print("Median Error :", np.median(errors))
    print("Max Error :", errors.max())
    print("--------------------------------")

# ----------------------
# Fixed GUI Preview Window
# ----------------------
class PreviewWindow:
    def __init__(self, win_name, width=480, height=360):
        self.win_name = win_name
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._frame = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def update(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def _run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.width, self.height)
        
        try:
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print(f"[Preview] Could not set topmost property: {e}")
            
        throttle_until = 0.0

        while not self._stop_event.is_set():
            current_time = time.time()
            
            if current_time < throttle_until:
                time.sleep(0.02)
                continue

            try:
                visible = cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE)
                if visible <= 0:
                    time.sleep(0.1)
                    continue
            except Exception:
                time.sleep(0.05)
                continue

            with self._lock:
                frame = self._frame
                self._frame = None

            if frame is not None:
                t0 = time.perf_counter()
                
                cv2.imshow(self.win_name, frame)
                cv2.waitKey(1)  
                
                dt = time.perf_counter() - t0
                
                if dt > 0.050:
                    throttle_until = time.time() + 0.05 
            else:
                t0 = time.perf_counter()
                cv2.waitKey(1)
                dt = time.perf_counter() - t0
                
                if dt > 0.050:
                    throttle_until = time.time() + 0.05
                else:
                    time.sleep(0.02)

        cv2.destroyWindow(self.win_name)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)


def mode_cursor_control():
    stopped_by_user = False
    failsafe_active = False
    clear_stop()
    disable_windows_power_throttling()
    
    update_progress(
        "Cursor Control",
        100,
        "Cursor Control Activated, Giving Cursor Control.",
        True,
        False
    )

    # Corrected landmark mapping to match physical eyes in the mirrored view
    PHYSICAL_RIGHT_EYE = [362, 386, 374, 263]
    PHYSICAL_LEFT_EYE = [33, 159, 145, 133]
    
    mapping = load_mapping()
    if mapping is None: 
        print("\n[ERROR] No Active tracking Profile found! Please run Calibration (2) or Load a Profile node first.")
        return
    
    wx = np.array(mapping["wx"])
    wy = np.array(mapping["wy"]) 
    invert_x = True 
    invert_y = mapping.get("invert_y", True)

    print(f"Invert X: {invert_x} | Invert Y: {invert_y}")
    
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
        
    ema_x, ema_y = None, None
    last_click_time = 0
    
    is_left_winked = False
    is_right_winked = False
    
    l_wink_start_ms = None
    double_click_triggered = False
    is_dragging = False

    l_ear_smooth = None
    r_ear_smooth = None
    l_open_count = 0
    r_open_count = 0

    norm_dist_smooth = None
    pinch_release_count = 0

    profile_label = mapping.get("profile_name", "Default")
    print(f"Cursor Control Active using Node Profile: '{profile_label}'. Press 'CTRL+Q' anywhere to exit.")

    preview = PreviewWindow("Wink Debugging (Press CTRL+Q to stop)")
    preview.start()

    mp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            if should_stop():
                stopped_by_user = True
                cursor_stopped()
                return
            if keyboard.is_pressed('ctrl+q'):
                stopped_by_user = True
                print("\nUniversal Exit Triggered. Closing...")
                cursor_stopped()
                break

            ret, frame = cap.read()
            if not ret: 
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_future = mp_executor.submit(face_mesh.process, rgb)
            hands_future = mp_executor.submit(hands.process, rgb)
            results_face = face_future.result()
            results_hands = hands_future.result()
            
            status_text = "Status: Monitoring"
            status_color = (0, 255, 0)

            pinch_seen = False
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    norm_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

                    if norm_dist_smooth is None:
                        norm_dist_smooth = norm_dist
                    else:
                        norm_dist_smooth = (PINCH_DIST_EMA_ALPHA * norm_dist
                                           + (1 - PINCH_DIST_EMA_ALPHA) * norm_dist_smooth)

                    if not is_dragging and norm_dist_smooth < 0.05:
                        pinch_seen = True
                    elif is_dragging and norm_dist_smooth < 0.12:
                        pinch_seen = True

                    if pinch_seen:
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        cv2.circle(frame, ((tx + ix) // 2, (ty + iy) // 2), 12, (0, 255, 0), -1, cv2.LINE_AA)

            current_ms = time.time() * 1000

            if pinch_seen:
                pinch_release_count = 0
                if not is_dragging:
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
                    last_click_time = current_ms
            else:
                if is_dragging:
                    pinch_release_count += 1
                    if pinch_release_count >= PINCH_RELEASE_FRAMES:
                        pyautogui.mouseUp(button='left')
                        is_dragging = False
                        last_click_time = current_ms
                        pinch_release_count = 0
                        norm_dist_smooth = None

            if is_dragging:
                status_text = "DRAGGING (Pinch Active)"
                status_color = (0, 255, 0)

            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                for idx in (PHYSICAL_LEFT_EYE + PHYSICAL_RIGHT_EYE):
                    pt = landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 0, 255), -1, cv2.LINE_AA)

                centers = get_iris_centers(results_face)
                if centers:
                    lx_n, ly_n, rx_n, ry_n = centers
                    px, py = predict_poly(wx, wy, (lx_n + rx_n) / 2.0, (ly_n + ry_n) / 2.0)

                    if invert_x:
                        px = SCREEN_W - px
                    if invert_y:
                        py = SCREEN_H - py

                    SAFE_MARGIN = 10
                    px = float(np.clip(px, SAFE_MARGIN, SCREEN_W - SAFE_MARGIN))
                    py = float(np.clip(py, SAFE_MARGIN, SCREEN_H - SAFE_MARGIN))

                    if ema_x is None: ema_x, ema_y = px, py
                    else:
                        dist = math.hypot(px - ema_x, py - ema_y)
                        alpha = EMA_ALPHA + min(0.8, (dist / max(SCREEN_W, SCREEN_H)) * 3.0)
                        ema_x = alpha * px + (1 - alpha) * ema_x
                        ema_y = alpha * py + (1 - alpha) * ema_y

                    try:
                        pyautogui.moveTo(int(ema_x), int(ema_y), _pause=False)
                    except pyautogui.FailSafeException:
                        failsafe_active = True
                        print("PyAutoGUI FailSafe Triggered.")
                        failsafe_triggered()
                        break

                l_ear_raw = calculate_ear(landmarks, PHYSICAL_LEFT_EYE)
                r_ear_raw = calculate_ear(landmarks, PHYSICAL_RIGHT_EYE)

                if l_ear_smooth is None:
                    l_ear_smooth = l_ear_raw
                    r_ear_smooth = r_ear_raw
                else:
                    l_ear_smooth = EAR_EMA_ALPHA * l_ear_raw + (1 - EAR_EMA_ALPHA) * l_ear_smooth
                    r_ear_smooth = EAR_EMA_ALPHA * r_ear_raw + (1 - EAR_EMA_ALPHA) * r_ear_smooth

                l_ear = l_ear_smooth
                r_ear = r_ear_smooth

                l_diff = r_ear - l_ear
                r_diff = l_ear - r_ear
                
                if l_ear < 0.20 and l_diff > 0.05:
                    l_open_count = 0
                    if l_wink_start_ms is None:
                        l_wink_start_ms = current_ms
                        double_click_triggered = False
                    
                    hold_duration = current_ms - l_wink_start_ms
                    
                    if hold_duration >= DOUBLE_CLICK_HOLD_MS:
                        if not is_dragging:
                            status_text = "DOUBLE CLICK (Left Eye)"
                            status_color = (0, 255, 255)
                            if not double_click_triggered:
                                pyautogui.doubleClick(button='left')
                                double_click_triggered = True
                                last_click_time = current_ms
                    else:
                        if not is_dragging:
                            status_text = "LEFT CLICK (Left Eye) - Holding..."
                            status_color = (0, 0, 255)
                        if not is_left_winked and (current_ms - last_click_time) > WINK_COOLDOWN_MS:
                            if not is_dragging:
                                pyautogui.click(button='left')
                            is_left_winked = True
                            last_click_time = current_ms
                elif l_ear > 0.22:
                    l_open_count += 1
                    if l_open_count >= WINK_RESET_FRAMES:
                        l_wink_start_ms = None
                        is_left_winked = False
                        double_click_triggered = False

                if r_ear < 0.20 and r_diff > 0.05:
                    r_open_count = 0
                    if not is_dragging:
                        status_text = "RIGHT CLICK (Right Eye)"
                        status_color = (255, 0, 0)
                    if not is_right_winked and (current_ms - last_click_time) > WINK_COOLDOWN_MS:
                        if not is_dragging:
                            pyautogui.click(button='right')
                        is_right_winked = True
                        last_click_time = current_ms
                elif r_ear > 0.22:
                    r_open_count += 1
                    if r_open_count >= WINK_RESET_FRAMES:
                        is_right_winked = False

                cv2.putText(frame, f"L: {l_ear:.2f} R: {r_ear:.2f}  (smoothed)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Profile: {profile_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

            preview.update(frame)

    finally:
        if is_dragging:
            pyautogui.mouseUp(button='left')

        preview.stop()
        mp_executor.shutdown(wait=False)
        cap.release()

        if not stopped_by_user and not failsafe_active:
            update_progress(
                "Completed",
                100,
                "Workflow Completed Successfully.",
                running=False,
                completed=True
            )

# ----------------------
# Main
# ----------------------

if __name__ == "__main__":
    print(get_profiles())

    if WEB_MODE:
        mode = sys.argv[1]
        if mode == "1": mode_alignment()
        elif mode == "2": mode_calibration(num_points=16)
        elif mode == "3": mode_cursor_control()
        elif mode == "4": interactive_profile_menu()
        else: print("Invalid mode.")
    else:
        while True:
            print("""
                GAZE PROTOTYPE MENU
                1 -> Alignment (Camera + Alignment Reference)
                2 -> Calibration (Smooth 4x4 + Slow drags + Edge sampling + Corners)
                3 -> Cursor control (Overlay + OS Cursor Control)
                4 -> Profile Management (Load Saved Calibration Profiles)
                q -> Quit
            """)
            cmd = input("Enter mode: ").strip().lower()
            if cmd == "1":
                mode_alignment()
            elif cmd == "2":
                mode_calibration(num_points=16)
            elif cmd == "3":
                mode_cursor_control()
            elif cmd == "4":
                interactive_profile_menu()
            elif cmd == "q":
                break
            else:
                print("Invalid. Choose 1, 2, 3, 4 or q.")