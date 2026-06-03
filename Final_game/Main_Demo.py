# Silence TensorFlow Lite / MediaPipe / absl log noise.
# On Windows the C++ runtime writes through the Windows API STD_ERROR_HANDLE,
# so we redirect both fd 2 and the Win32 handle to NUL.
import os, sys, ctypes as _ctypes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]      = "3"

class _SuppressStderr:
    """Context manager: silences fd-2-level and (on Windows) Win32 stderr."""
    def __enter__(self):
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._saved_fd2  = os.dup(2)
        os.dup2(self._devnull_fd, 2)
        os.close(self._devnull_fd)
        if sys.platform == "win32":
            k32 = _ctypes.WinDLL("kernel32", use_last_error=True)
            self._nul_handle  = k32.CreateFileW(
                "nul", 0x40000000, 0, None, 3, 0, None)
            self._old_handle  = k32.GetStdHandle(-12)   # STD_ERROR_HANDLE
            k32.SetStdHandle(-12, self._nul_handle)
            self._k32 = k32
        return self
    def __exit__(self, *_):
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd2)
        if sys.platform == "win32":
            self._k32.SetStdHandle(-12, self._old_handle)
            self._k32.CloseHandle(self._nul_handle)

with _SuppressStderr():
    import mediapipe as mp
from collections import defaultdict
import cv2
cv2.setUseOptimized(True)  # Enable OpenCV CPU optimizations
import time
import platform
import numpy as np
import ctypes
import subprocess
import json
import socket
from pathlib import Path
from ultralytics import YOLO
import math
import statistics
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ===== WAYS TO GET PIXEL VALUES FROM HANDTRACKER =====
# 1. get_positions(img, hand_no=0, use_last_known=True)
#    - Returns: List of [landmark_id, x_pixel, y_pixel, z_depth] for all 21 hand landmarks
#    - x_pixel, y_pixel: Pixel coordinates in the image
#    - z_depth: Normalized depth value (-1 to 1, negative = closer to camera)
#
# 2. get_index_finger_pos(img, hand_no=0)
#    - Returns: [8, x_pixel, y_pixel, z_depth] for pege finger tip (landmark 8)
#    - Shortcut for getting just the pege finger position
#
# 3. get_all_finger_tips(img, hand_no=0, use_last_known=True)
#    - Returns: List of [landmark_id, x_pixel, y_pixel, z_depth] for finger tips only
#    - Finger tips: tommel(4), pege(8), lange(12), ringe(16), lille(20)
#
# All methods return pixel coordinates relative to the input image dimensions.
# Use these for touch detection, gesture analysis, or accuracy calculations.

# ===== CONFIGURATION =====
# Select which dimension classes to detect
# Options:
#   None = detect ALL classes
#   [0, 3, 6] = detect only 2x10 (0), 2x4 (3), 3x3 (6)
# Available classes:
#   0: 2x10, 1: 2x2, 2: 2x3, 3: 2x4, 4: 2x6, 5: 2x8, 6: 3x3, 7: 4x4, 8: 4x6, 9: 4x8, 10: 6x6
SELECTED_CLASSES = [3, 8]  # Change this to select specific classes

# NMS settings for YOLO: IoU threshold for non-maximum suppression (higher = less suppression)
YOLO_NMS_IOU = 1.0

# Camera resolution settings (used at all times): width and height in pixels
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Absolute path to the directory containing this script (so the game runs from any cwd)
SCRIPT_DIR = Path(__file__).resolve().parent

# Path to the YOLO model weights (relative to this script's directory)
MODEL_PATH = str(SCRIPT_DIR / 'yolo_results/train_class_gpu_6/weights/best.pt')

# Hand-tracking stability settings
# Minimum confidence for initial hand detection (0.0-1.0, higher = stricter)
HAND_DETECTION_CONFIDENCE = 0.65
# Minimum confidence for tracking hands across frames (0.0-1.0, higher = stricter)
HAND_TRACKING_CONFIDENCE = 0.75
# Model complexity for MediaPipe (0=fastest, 1=more complex but less error)
HAND_MODEL_COMPLEXITY = 1
# Smoothing factor for landmark positions (0.0=no smoothing, 1.0=max smoothing).
# Lower values reduce visual jitter but increase release lag (ghost input after
# lifting a finger). 0.8 gives responsive release (~1 frame lag) with mild smoothing.
LANDMARK_SMOOTHING_ALPHA = 0.8
# Max frames to use last known landmarks when hand is lost
MAX_TRACK_LOST_FRAMES = 2
# Whether to use last known positions for touch detection when hand is lost
TOUCH_USE_LAST_KNOWN = False

# ----- Depth calibration toggle -----
# True  = run the per-finger depth calibration before tracking starts.
#         A touch is only registered when the finger is BOTH inside a bounding
#         box AND its geometry distance exceeds the calibrated threshold.
# False = skip calibration entirely.  A touch is registered as soon as any
#         finger tip enters a bounding box (pure 2-D position, original behaviour).
USE_DEPTH_CALIBRATION = False

# Touch-to-key mapping (left-to-right boxes 1..4 -> a, w, s, d)
BOX_KEY_BINDINGS = ['a', 'w', 's', 'd']

# Finger-to-key constraint: only this finger may trigger each key.
# Pege=index(8), Lange=middle(12), Ringe=ring(16), Lille=pinky(20), Tommel=thumb(4).
KEY_ALLOWED_FINGER = {
    'a': 'Pege',    # yellow  – index
    'w': 'Lange',   # purple  – middle
    's': 'Ringe',   # green   – ring
    'd': 'Lille',   # red     – pinky
    'f': 'Tommel',  # blue    – thumb (activator)
}

# Name of the folder containing game scripts
GAMES_FOLDER_NAME = "games"

# Host and port for UDP control socket to send key presses to the game
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 50555

# Maximum number of LEGO bricks to detect and track
MAX_LEGO_BRICKS = 5
MAX_DETECTION_RETRIES = 5
THUMB_ACTIVATER = False #set to True if thumb activation should be used

CALIB_POINTS  = 5
HOLD_SECONDS  = 1.5
TOUCH_BIAS_PX = 20
CALIB_POINT_POSITIONS = [
    (0.25, 0.35),
    (0.75, 0.35),
    (0.50, 0.55),
    (0.20, 0.72),
    (0.80, 0.72),
]

# ----- Suggest-changes brick-adjustment thresholds -----
# These control when a brick position change is recommended based on the
# averaged region-touch percentages from the last 3 sessions.
# Equations (BR+BL+TR+TL+C = 1, all values in [0, 1]):
#   BR + BL > SUGGEST_A  →  move brick one step down
#   TR + TL > SUGGEST_B  →  move brick one step up
#   TR + BR > SUGGEST_C  →  move brick one step to the right
#   TL + BL > SUGGEST_D  →  move brick one step to the left
SUGGEST_A = 0.3   # bottom-bias threshold
SUGGEST_B = 0.3   # top-bias threshold
SUGGEST_C = 0.3   # right-bias threshold
SUGGEST_D = 0.3   # left-bias threshold
SUGGEST_E = 0.5   # center threshold: if avg center % < E, suggest shrinking brick to 2x4
SUGGEST_COOLDOWN = 2  # sessions a finger is locked after any change is suggested
# ===== END CONFIGURATION =====

# Per-finger correction factors applied when converting playing-session finger
# lengths to cm.  During gameplay the hand is elevated closer to the camera,
# so the same physical finger spans more pixels than it did during the flat-hand
# goal measurement.  These factors (goal_mean / play_mean, averaged across three
# camera distances: 44 cm, 49 cm, 54 cm) rescale the playing measurements back
# down to be comparable with the goal lengths stored at calibration time.
FINGER_PLAY_SCALE = {
    "Pege":  0.933,
    "Lange": 0.909,
    "Ringe": 0.908,
    "Lille": 0.963,
}

FINGER_NAMES   = ["Tommel", "Pege", "Lange", "Ringe", "Lille"]
FINGER_TIP_IDS = [4, 8, 12, 16, 20]
FINGER_JOINT_IDS = {
    "Tommel": [3, 2, 1],
    "Pege":  [7, 6, 5],
    "Lange": [11, 10, 9],
    "Ringe":   [15, 14, 13],
    "Lille":  [19, 18, 17],
}

VK_MAP = {
    'w': 0x57,
    'a': 0x41,
    's': 0x53,
    'd': 0x44,
}

# Dimension classes mapping (must match training order: sorted dimensions)
DIMENSION_CLASSES = {
    0: "2x10",
    1: "2x2",
    2: "2x3",
    3: "2x4",
    4: "2x6",
    5: "2x8",
    6: "3x3",
    7: "4x4",
    8: "4x6",
    9: "4x8",
    10: "6x6"
}


def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two boxes.

    Boxes are in (x1, y1, x2, y2) format.
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    xi1 = max(x1A, x1B)
    yi1 = max(y1A, y1B)
    xi2 = min(x2A, x2B)
    yi2 = min(y2A, y2B)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    areaA = max(0, x2A - x1A) * max(0, y2A - y1A)
    areaB = max(0, x2B - x1B) * max(0, y2B - y1B)
    union = areaA + areaB - intersection

    return intersection / union if union > 0 else 0.0


def send_control_state(control_socket, keys_set, key_accuracy=None):
    """
    Send current pressed key states to pygame game via localhost UDP.
    Optionally includes per-key accuracy scores (0.0–1.0) reflecting how
    centred the touching finger tip was within each bounding box.
    """
    if control_socket is None:
        return

    try:
        payload = {
            "keys": sorted(list(keys_set)),
            "ts": time.time()
        }
        if key_accuracy:
            payload["accuracy"] = key_accuracy
        control_socket.sendto(json.dumps(payload).encode("utf-8"), (CONTROL_HOST, CONTROL_PORT))
    except Exception:
        pass


def set_key_state(key, is_down):
    """
    Set key down/up state for WASD keys.
    """
    if not key or not isinstance(key, str):
        return

    if platform.system() != "Windows":
        return

    try:
        vk_code = VK_MAP.get(key.lower())
        if vk_code is None:
            return

        KEYEVENTF_KEYUP = 0x0002
        flags = 0 if is_down else KEYEVENTF_KEYUP
        ctypes.windll.user32.keybd_event(vk_code, 0, flags, 0)
    except Exception:
        pass


def px_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_finger_geometry(tracker, img, hand_no=0):
    pos = tracker.get_positions(img, hand_no)
    if len(pos) < 21:
        return []
    result = []
    for i, tip_id in enumerate(FINGER_TIP_IDS):
        name = FINGER_NAMES[i]
        tip  = (pos[tip_id][1], pos[tip_id][2])
        dist_sum = sum(px_dist(tip, (pos[j][1], pos[j][2])) for j in FINGER_JOINT_IDS[name])
        result.append((name, pos[tip_id][1], pos[tip_id][2], dist_sum))
    return result


def run_calibration(cap, tracker, lego_bricks):
    """
    Runs the per-finger depth calibration sequence.
    Returns (thresholds, directions) dicts, or (None, None) if the user cancels.
    Only called when USE_DEPTH_CALIBRATION is True.
    """
    all_samples = {name: [] for name in FINGER_NAMES}
    hold_start  = None
    point_samples = {name: [] for name in FINGER_NAMES}
    current_finger_idx = 0  # start with Tommel

    print("=== KALIBRERING ===")
    print("Rør ved en lego-brik med hver finger en ad gangen.")
    print("Start med din TOMMEL, derefter PEGE, LANGE, RINGE, LILLE.\n")

    while current_finger_idx < len(FINGER_NAMES):
        cap.grab()
        success, img = cap.retrieve()
        if not success:
            break

        img  = tracker.find_hands(img, draw=True)
        geom = get_finger_geometry(tracker, img)

        current_finger = FINGER_NAMES[current_finger_idx]

        # Find the current finger in geom
        current_geom = next((g for g in geom if g[0] == current_finger), None)

        # Check if current finger is inside any lego box
        finger_in_box = False
        if current_geom:
            fx, fy = current_geom[1], current_geom[2]
            for (x1, y1, x2, y2, bname) in lego_bricks:
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    finger_in_box = True
                    break

        if finger_in_box:
            if hold_start is None:
                hold_start    = time.time()
                point_samples = {name: [] for name in FINGER_NAMES}
            elapsed = time.time() - hold_start

            # Sample only the current finger
            if current_geom:
                point_samples[current_finger].append(current_geom[3])

            if elapsed >= HOLD_SECONDS:
                s = point_samples[current_finger]
                if s:
                    all_samples[current_finger].extend(s)
                    print(f"  ✓ {current_finger:6s}: dist_avg={sum(s)/len(s):.1f}px")
                current_finger_idx += 1
                hold_start = None
                point_samples = {name: [] for name in FINGER_NAMES}

            msg = f"{current_finger}: calibrating... {elapsed:.1f}/{HOLD_SECONDS}s"
        else:
            hold_start = None
            msg = f"Touch a brick with your {current_finger}"

        # Draw lego boxes
        for (x1, y1, x2, y2, bname) in lego_bricks:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # Progress text e.g. "2/5"
        progress_str = f"{current_finger_idx}/{len(FINGER_NAMES)}"
        cv2.putText(img, f"CALIBRATION  {progress_str}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 55), 2)
        cv2.putText(img, msg, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, None

    thresholds = {}
    directions = {}
    print("=== Calibration summary ===")
    for name in FINGER_NAMES:
        s = all_samples[name]
        mean_dist = sum(s) / len(s) if s else 0.0
        thresh    = mean_dist - TOUCH_BIAS_PX
        thresholds[name] = thresh
        directions[name] = 1
        print(f"  {name:6s}: mean={mean_dist:.1f}px  threshold>={thresh:.1f}px")
    print()
    return thresholds, directions


class HandTracker:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        detection_confidence=HAND_DETECTION_CONFIDENCE,
        tracking_confidence=HAND_TRACKING_CONFIDENCE,
        model_complexity=HAND_MODEL_COMPLEXITY,
        smoothing_alpha=LANDMARK_SMOOTHING_ALPHA,
        max_track_lost_frames=MAX_TRACK_LOST_FRAMES,
    ):
        """
        Initialize hand tracker with MediaPipe.
        
        Args:
            mode: Static image mode (False for video)
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity
        self.smoothing_alpha = smoothing_alpha
        self.max_track_lost_frames = max_track_lost_frames
        
        # Initialize MediaPipe hands (suppress TFLite/absl noise during init)
        self.mp_hands = mp.solutions.hands
        with _SuppressStderr():
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.mode,
                model_complexity=self.model_complexity,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence
            )
        # Warm up: run one dummy inference so TFLite/absl emits all its startup
        # messages NOW (while stderr is still suppressed) instead of on the first
        # real frame (which would print after user-visible output).
        with _SuppressStderr():
            _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.hands.process(_dummy)

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Clickable regions: list of (x1, y1, x2, y2, name)
        self.clickable_regions = []
        self.results = None
        self.prev_landmarks = {}
        self.missed_frames = {}
        
    def find_hands(self, img, draw=True, draw_z_values=False):
        """
        Detect hands in the image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw landmarks on the image
            draw_z_values: Whether to display z-depth values on landmarks
            
        Returns:
            Processed image with landmarks drawn
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Custom per-finger colors (BGR format)
                # Tommel=blå, Pege=gul, Lange=lilla, Ringe=grøn, Lille=rød
                _JOINT_COLOR = (255, 255, 255)  # hvid – håndled & MCP-knoer (0,1,5,9,13,17)
                _FINGER_LANDMARK_COLORS = {
                    0:  _JOINT_COLOR,             # håndled
                    # Tommel (1-4): 1 er MCP-kno
                    1: _JOINT_COLOR, 2: (255, 0, 0), 3: (255, 0, 0), 4: (255, 0, 0),
                    # Pege (5-8): 5 er MCP-kno
                    5: _JOINT_COLOR, 6: (0, 255, 255), 7: (0, 255, 255), 8: (0, 255, 255),
                    # Lange (9-12): 9 er MCP-kno
                    9: _JOINT_COLOR, 10: (128, 0, 128), 11: (128, 0, 128), 12: (128, 0, 128),
                    # Ringe (13-16): 13 er MCP-kno
                    13: _JOINT_COLOR, 14: (0, 255, 0), 15: (0, 255, 0), 16: (0, 255, 0),
                    # Lille (17-20): 17 er MCP-kno
                    17: _JOINT_COLOR, 18: (0, 0, 255), 19: (0, 0, 255), 20: (0, 0, 255),
                }
                # Map each connection to its finger color based on the distal landmark.
                # Connections between the base joints (0,1,5,9,13,17) use _JOINT_COLOR.
                _FINGER_CONNECTION_COLORS = {
                    (0, 1):  _JOINT_COLOR,            # håndled → tommel MCP
                    (1, 2):  (255, 0, 0),             # tommel
                    (2, 3):  (255, 0, 0),
                    (3, 4):  (255, 0, 0),
                    (0, 5):  _JOINT_COLOR,            # håndled → pege MCP
                    (5, 6):  (0, 255, 255),           # pege
                    (6, 7):  (0, 255, 255),
                    (7, 8):  (0, 255, 255),
                    (5, 9):  _JOINT_COLOR,            # pege MCP → lange MCP
                    (9, 10): (128, 0, 128),           # lange
                    (10, 11): (128, 0, 128),
                    (11, 12): (128, 0, 128),
                    (9, 13): _JOINT_COLOR,            # lange MCP → ringe MCP
                    (13, 14): (0, 255, 0),            # ringe
                    (14, 15): (0, 255, 0),
                    (15, 16): (0, 255, 0),
                    (13, 17): _JOINT_COLOR,           # ringe MCP → lille MCP
                    (0, 17):  _JOINT_COLOR,           # håndled → lille MCP
                    (17, 18): (0, 0, 255),            # lille
                    (18, 19): (0, 0, 255),
                    (19, 20): (0, 0, 255),
                }
                _landmark_spec = {
                    idx: self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4)
                    for idx, color in _FINGER_LANDMARK_COLORS.items()
                }
                _connection_spec = {
                    conn: self.mp_draw.DrawingSpec(color=color, thickness=2)
                    for conn, color in _FINGER_CONNECTION_COLORS.items()
                }
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    _landmark_spec,
                    _connection_spec
                )
                
                # Draw z-values on each landmark
                if draw_z_values:
                    h, w, c = img.shape
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cz = landmark.z
                        cv2.putText(img, f"{cz:.2f}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return img
    
    def get_positions(self, img, hand_no=0, use_last_known=True):
        """
        Get landmark positions for a specific hand with depth.
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            
        Returns:
            List of landmark positions [id, x, y, z] where z is normalized depth
        """
        landmark_list = []
        
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]

                h, w, c = img.shape
                for id, landmark in enumerate(hand.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cz = landmark.z
                    landmark_list.append([id, cx, cy, cz])

                previous = self.prev_landmarks.get(hand_no)
                if previous and len(previous) == len(landmark_list):
                    alpha = self.smoothing_alpha
                    smoothed = []
                    for curr, prev in zip(landmark_list, previous):
                        sx = int(alpha * curr[1] + (1 - alpha) * prev[1])
                        sy = int(alpha * curr[2] + (1 - alpha) * prev[2])
                        sz = alpha * curr[3] + (1 - alpha) * prev[3]
                        smoothed.append([curr[0], sx, sy, sz])
                    landmark_list = smoothed

                self.prev_landmarks[hand_no] = landmark_list
                self.missed_frames[hand_no] = 0
                return landmark_list

        if use_last_known:
            last_known = self.prev_landmarks.get(hand_no)
            if last_known is not None:
                missed = self.missed_frames.get(hand_no, 0) + 1
                self.missed_frames[hand_no] = missed
                if missed <= self.max_track_lost_frames:
                    return last_known

                self.prev_landmarks.pop(hand_no, None)
                self.missed_frames.pop(hand_no, None)

        return landmark_list
    
    def get_all_finger_tips(self, img, hand_no=0, use_last_known=True):
        """
        Get all finger tip positions.
        Finger tips are landmarks: 4 (tommel), 8 (pege), 12 (lange), 16 (ringe), 20 (lille)
        
        Args:
            img: Input image
            hand_no: Hand index
            
        Returns:
            List of finger tip positions [[id, x, y, z], ...]
        """
        positions = self.get_positions(img, hand_no=hand_no, use_last_known=use_last_known)
        finger_tip_landmarks = [4, 8, 12, 16, 20]  # Tommel, Pege, Lange, Ringe, Lille
        finger_tips = []
        
        for landmark_id in finger_tip_landmarks:
            if positions and len(positions) > landmark_id:
                finger_tips.append(positions[landmark_id])
        
        return finger_tips
    
    def check_touching_region(self, img, hand_no=0):
        """
        Check which regions (if any) any finger tip is touching.
        
        Args:
            img: Input image
            hand_no: Hand index
            
        Returns:
            List of names of touched regions, or empty list if none
        """
        finger_tips = self.get_all_finger_tips(
            img,
            hand_no=hand_no,
            use_last_known=TOUCH_USE_LAST_KNOWN
        )
        
        if not finger_tips:
            return []
        
        touched_regions = []
        
        # Check all finger tips against all regions
        for finger_pos in finger_tips:
            x, y = finger_pos[1], finger_pos[2]
            
            # Check all regions
            for x1, y1, x2, y2, name in self.clickable_regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Add region if not already in the list (avoid duplicates)
                    if name not in touched_regions:
                        touched_regions.append(name)
        
        return touched_regions


def draw_lego_boxes(img, tracker, region_to_key, touch_states, use_depth,
                    finger_tips_cache=None, key_accuracy=None):
    """
    Draw bounding boxes.

    When use_depth is True the box glows if a depth-touching finger is inside it.
    When use_depth is False the box glows whenever any finger tip is inside it
    (i.e. the same 2-D condition that already triggers the key press).

    finger_tips_cache: pre-computed list from get_all_finger_tips() for this frame.
                       Pass this to avoid an extra smoothing call.
    key_accuracy:      dict mapping key -> float (0.0-1.0) for the current frame.
                       When provided, the accuracy % is drawn inside active boxes.
    """
    # BGR colors assigned left-to-right: blue, yellow, purple, green, red
    BOX_COLORS = [
        (255, 80,  0),    # blue
        (0,   220, 255),  # yellow
        (140, 0,   140),  # purple
        (0,   160, 0),    # green
        (0,   0,   230),  # red
    ]

    finger_tips = finger_tips_cache if finger_tips_cache is not None \
                  else tracker.get_all_finger_tips(img)

    # Sort boxes left-to-right so colors are assigned by position
    sorted_regions = sorted(tracker.clickable_regions, key=lambda r: r[0])

    for color_idx, (x1, y1, x2, y2, bname) in enumerate(sorted_regions):
        box_color = BOX_COLORS[color_idx % len(BOX_COLORS)]
        box_active = False

        if use_depth:
            # Glow only when a depth-confirmed finger is inside the box
            for tip in finger_tips:
                tip_id, tx, ty, tz = tip
                finger_name = (
                    FINGER_NAMES[FINGER_TIP_IDS.index(tip_id)]
                    if tip_id in FINGER_TIP_IDS
                    else None
                )
                if finger_name and touch_states.get(finger_name, False):
                    if x1 <= tx <= x2 and y1 <= ty <= y2:
                        box_active = True
                        break
        else:
            # Glow whenever any finger tip is inside the box (pure 2-D)
            for tip in finger_tips:
                _, tx, ty, _ = tip
                if x1 <= tx <= x2 and y1 <= ty <= y2:
                    box_active = True
                    break

        key = region_to_key.get(bname, "")
        if box_active:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1-6, y1-6), (x2+6, y2+6), box_color, -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)




def launch_game(background_canvas=None):
    """Show a cv2 game-selection screen, wait for a number key press, then launch."""
    script_dir = Path(__file__).resolve().parent
    games_dir  = script_dir / GAMES_FOLDER_NAME

    if not games_dir.exists():
        print(f"Games folder not found: {games_dir}")
        return None, None

    available_games = sorted([f.stem for f in games_dir.glob("*.py") if f.is_file()])

    if not available_games:
        print("No games found in the games folder.")
        return None, None

    print("\nAvailable games:")
    for i, game in enumerate(available_games, 1):
        print(f"  {i}: {game}")

    # ── Build the selection canvas ─────────────────────────────────────────────
    # Start from the background (handplacement image) if provided, else black.
    if background_canvas is not None:
        canvas = background_canvas.copy()
    else:
        canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    n         = len(available_games)
    valid_keys = set(str(i) for i in range(1, n + 1))

    def _draw_menu(highlight=None):
        """Re-draw the menu overlay onto a fresh copy of canvas."""
        img = canvas.copy()

        # Semi-transparent dark panel behind the menu
        panel_x1, panel_y1 = 20, 20
        panel_x2 = CAMERA_WIDTH - 20
        panel_y2 = 55 + n * 38 + 20
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        # Title
        title = f"Vaelg et spil  (tast 1 til {n})"
        (tw, _), _ = cv2.getTextSize(title, font, 0.75, 2)
        cv2.putText(img, title, ((CAMERA_WIDTH - tw) // 2, 52),
                    font, 0.75, (80, 255, 160), 2, cv2.LINE_AA)

        # Quit hint
        quit_hint = "Tryk Q for at afslutte sessionen"
        (qw, _), _ = cv2.getTextSize(quit_hint, font, 0.55, 1)
        cv2.putText(img, quit_hint, ((CAMERA_WIDTH - qw) // 2, panel_y2 - 8),
                    font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # Game list
        for idx, game in enumerate(available_games):
            num   = idx + 1
            y     = 55 + idx * 38 + 30
            col   = (0, 220, 255) if highlight == num else (200, 200, 200)
            thick = 2 if highlight == num else 1
            label = f"  {num}.  {game}"
            cv2.putText(img, label, (40, y), font, 0.65, col, thick, cv2.LINE_AA)

        return img

    # Show initial menu
    cv2.imshow("Hand Tracking", _draw_menu())

    selected_game = None
    while selected_game is None:
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
            return None, None
        if key == 255:          # no key yet
            continue
        ch = chr(key)
        if ch in valid_keys:
            idx = int(ch) - 1
            selected_game = available_games[idx]
            # Flash highlight briefly so the user sees their selection
            cv2.imshow("Hand Tracking", _draw_menu(highlight=int(ch)))
            cv2.waitKey(400)
        elif key == ord('q'):
            print("Game selection cancelled.")
            return None, None

    # ── Launch ────────────────────────────────────────────────────────────────
    game_script = games_dir / f"{selected_game}.py"
    if not game_script.exists():
        print(f"Game not found: {game_script}")
        return None, None

    print(f"Launching game: {game_script.name}")
    game_process = subprocess.Popen(
        [sys.executable, str(game_script)], cwd=str(game_script.parent)
    )
    time.sleep(1.0)

    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return game_process, control_socket


def choose_key_bindings(clickable_regions):
    """Assign WASD keys to detected regions (left-to-right).

    The thumb-activator brick sits at the BOTTOM of the (flipped) image,
    i.e. it has the LARGEST y1 value. We exclude that bottom-most box and
    assign 'a','w','s','d' to the remaining 4 game boxes in left-to-right
    order so that:
        leftmost  -> 'a' -> Pege
        2nd       -> 'w' -> Lange
        3rd       -> 's' -> Ringe
        rightmost -> 'd' -> Lille
    This matches the activation_region selection in run_hand_tracking_loop
    (which also uses max y1) and KEY_TO_FINGER in process_touch_history.
    """
    if not clickable_regions:
        return {}

    candidate_regions = list(clickable_regions)
    # The activator brick is the bottom-most box on the (flipped) image.
    if len(candidate_regions) > len(BOX_KEY_BINDINGS):
        activator_region = max(candidate_regions, key=lambda r: r[1])
        candidate_regions.remove(activator_region)
    sorted_regions = sorted(candidate_regions, key=lambda r: r[0])

    keybind_regions = sorted_regions[:len(BOX_KEY_BINDINGS)]
    return {name: BOX_KEY_BINDINGS[i] for i, (_, _, _, _, name) in enumerate(keybind_regions)}


def detect_legos(cap, model, tracker, max_bricks=MAX_LEGO_BRICKS, detection_duration=2.0):
    """Detect all Lego bricks simultaneously in one detection window.

    All frames captured during ``detection_duration`` seconds are accumulated.
    Overlapping detections across frames are grouped into clusters and averaged,
    so the final bounding box for each brick is based on every frame in the window
    rather than a single snapshot.  The top ``max_bricks`` clusters (ranked by
    average confidence × log-frequency) are registered as clickable regions.

    After detection a preview screen shows the confirmed bricks.
    Press Enter to proceed, R to re-run detection, Q to quit.
    """
    selected_classes = None if SELECTED_CLASSES is None else set(SELECTED_CLASSES)

    # Blue, yellow, purple, green, red  (BGR)
    BOX_COLORS = [
        (255, 80,  0),
        (0,   220, 255),
        (140, 0,   140),
        (0,   160, 0),
        (0,   0,   230),
    ]

    while True:   # outer loop: re-run on R
        all_detections = []
        detection_start_time = time.time()

        print(f"Starting lego brick detection ({detection_duration:.1f}s window)... Press 'q' to quit.")

        # ── Phase 1: accumulate raw detections across all frames ─────────────
        while True:
            cap.grab()
            success, img = cap.retrieve()
            if not success:
                break

            img = cv2.flip(img, -1)
            elapsed = time.time() - detection_start_time

            if elapsed >= detection_duration:
                break

            results = model(img, verbose=False, iou=YOLO_NMS_IOU, classes=SELECTED_CLASSES, conf=0.10)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if selected_classes is not None and class_id not in selected_classes:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        all_detections.append((float(x1), float(y1), float(x2), float(y2), conf, class_id))

            # Countdown overlay
            remaining = detection_duration - elapsed
            msg1 = f"Scanner efter brikker... {remaining:.1f}s"
            msg2 = "Hold haanden vaek fra kameraet"
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, (msg, scale, thick) in enumerate([(msg1, 0.9, 2), (msg2, 0.7, 1)]):
                (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
                tx = (img.shape[1] - tw) // 2
                ty = 50 + i * (th + 22)
                cv2.rectangle(img, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
                cv2.putText(img, msg, (tx, ty), font, scale, (0, 255, 180), thick, cv2.LINE_AA)

            cv2.imshow("Hand Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # ── Phase 2: cluster overlapping detections and pick the top N ───────
        tracker.clickable_regions = []
        selected_boxes = []   # (x1, y1, x2, y2, class_id, avg_conf)

        scored_clusters = []
        if all_detections:
            clusters = []
            cluster_iou_threshold = 0.3

            for detection in all_detections:
                x1, y1, x2, y2, conf, class_id = detection
                matched_cluster = None
                for cluster in clusters:
                    ab = cluster['avg_box']
                    if compute_iou((x1, y1, x2, y2), (ab[0], ab[1], ab[2], ab[3])) > cluster_iou_threshold:
                        matched_cluster = cluster
                        break
                if matched_cluster:
                    matched_cluster['boxes'].append((x1, y1, x2, y2))
                    matched_cluster['confidences'].append(conf)
                    matched_cluster['class_ids'].append(class_id)
                    matched_cluster['avg_box'] = np.array(matched_cluster['boxes']).mean(axis=0)
                else:
                    clusters.append({
                        'boxes': [(x1, y1, x2, y2)],
                        'confidences': [conf],
                        'class_ids': [class_id],
                        'avg_box': np.array([x1, y1, x2, y2])
                    })

            scored_clusters = []
            for cluster in clusters:
                avg_conf = np.mean(cluster['confidences'])
                count    = len(cluster['boxes'])
                majority_class = max(set(cluster['class_ids']), key=cluster['class_ids'].count)
                score = avg_conf * (1 + np.log(count))
                scored_clusters.append((cluster['avg_box'], avg_conf, count, score, majority_class))

            scored_clusters.sort(key=lambda x: x[3], reverse=True)

            # Collect all confident clusters first, then sort left-to-right so
        # Lego1 is always the leftmost brick in the image regardless of which
        # cluster scored highest.
        candidate_boxes = []
        for avg_box, avg_conf, count, score, class_id in scored_clusters[:max_bricks]:
                if avg_conf < 0.60:
                    print(f"  Skipping cluster: confidence={avg_conf*100:.1f}% below 60% threshold")
                    continue
                x1, y1, x2, y2 = avg_box
                candidate_boxes.append((x1, y1, x2, y2, class_id, avg_conf, count))

        # Sort by x1 (left-to-right)
        candidate_boxes.sort(key=lambda b: b[0])

        for brick_num, (x1, y1, x2, y2, class_id, avg_conf, count) in enumerate(candidate_boxes, start=1):
                dimension = DIMENSION_CLASSES.get(class_id, "Unknown")
                selected_boxes.append((x1, y1, x2, y2, class_id, avg_conf))
                tracker.clickable_regions.append(
                    (int(x1), int(y1), int(x2), int(y2), f"Lego{brick_num}({dimension})")
                )
                print(f"  Lego{brick_num}: {dimension}, confidence={avg_conf*100:.1f}%, frames={count}")

        print(f"Detection complete! Found {len(tracker.clickable_regions)}/{max_bricks} bricks.")

        # ── Phase 3: preview – wait for Enter / R / Q ────────────────────────
        no_bricks_found = len(selected_boxes) < max_bricks
        redetect = False
        while True:
            cap.grab()
            success, img = cap.retrieve()
            if not success:
                break
            img = cv2.flip(img, -1)

            for bi, (bx1, by1, bx2, by2, b_class_id, b_conf) in enumerate(selected_boxes):
                color     = BOX_COLORS[bi % len(BOX_COLORS)]
                dimension = DIMENSION_CLASSES.get(b_class_id, "?")
                cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                label = f"{dimension}({b_conf*100:.0f}%)"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.27, 1)
                ly = max(int(by1) - 6, lh + 4)
                cv2.rectangle(img, (int(bx1), ly - lh - 4), (int(bx1) + lw + 4, ly + 2), color, -1)
                cv2.putText(img, label, (int(bx1) + 2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 0, 0), 1, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if no_bricks_found:
                found = len(selected_boxes)
                line1 = f"Fandt {found}/{max_bricks} brikker - skal bruge alle {max_bricks}!" if found > 0 else "Ingen brikker fundet!"
                messages = [
                    (line1, 0.9, 2, (0, 80, 255)),
                    ("R=scan igen   Q=afslut", 0.7, 1, (0, 255, 180)),
                ]
            else:
                messages = [
                    (f"Fandt {len(selected_boxes)} brikker!  Enter=start  R=scan igen", 0.75, 2, (0, 255, 180)),
                ]
            for i, (msg, scale, thick, color) in enumerate(messages):
                (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
                tx = (img.shape[1] - tw) // 2
                ty = 50 + i * (th + 22)
                cv2.rectangle(img, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
                cv2.putText(img, msg, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

            cv2.imshow("Hand Tracking", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10) and len(selected_boxes) >= max_bricks:   # Enter → proceed to game
                return
            if key == ord('r'):   # R → re-run detection
                redetect = True
                break
            if key == ord('q'):   # Q → quit
                return

        if not redetect:
            return


def process_touch_history(touch_history, clickable_regions, region_to_key):
    region_lookup = {r[4]: r[:4] for r in clickable_regions}
    processed = []
    KEY_TO_FINGER = {
        'a': 'Pege',
        'w': 'Lange',
        's': 'Ringe',
        'd': 'Lille'
    }

    # Tommel / aktivator brik reference (for cm <-> pixel scaling).
    # Tommel brikken sidder NEDERST i det (flippede) billede, dvs.
    # den har den STØRSTE y-værdi -- vi finder den via max(y1).
    thumb_region = max(clickable_regions, key=lambda r: r[1])
    thumb_y1 = thumb_region[1]
    thumb_y2 = thumb_region[3]
    thumb_height = thumb_y2 - thumb_y1
    bricklength = 4.7
    finger_factor = bricklength/thumb_height if thumb_height > 0 else 0.0

    # Initialize per-region counters
    region_data = {region_name: {
        "center": 0, "top_left": 0, "bottom_left": 0,
        "top_right": 0, "bottom_right": 0, "misses": 0,
        "touches": 0, "norm_dist": 0.0, "touch_xs": [], "touch_ys": []
    } for region_name in region_lookup}

    for entry in touch_history:
        for region_name, (tx, ty) in entry["touches"].items():
            x1, y1, x2, y2 = region_lookup[region_name]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_square_b = (x1 + (x2 - x1) / 4, y1 + (y2 - y1) / 4)
            center_square_t = (x1 + 3 * (x2 - x1) / 4, y1 + 3 * (y2 - y1) / 4)

            region_data[region_name]["touches"] += 1

            if (center_square_b[0] <= tx <= center_square_t[0] and
                    center_square_b[1] <= ty <= center_square_t[1]):
                region_data[region_name]["center"] += 1
            elif tx <= center_x and ty <= center_y:
                region_data[region_name]["top_left"] += 1
            elif tx <= center_x and ty > center_y:
                region_data[region_name]["bottom_left"] += 1
            elif tx > center_x and ty <= center_y:
                region_data[region_name]["top_right"] += 1
            elif tx > center_x and ty > center_y:
                region_data[region_name]["bottom_right"] += 1
            else:
                region_data[region_name]["misses"] += 1

            dist = math.sqrt((tx - center_x) ** 2 + (ty - center_y) ** 2)
            max_dist = math.sqrt(((x2 - x1) / 2) ** 2 + ((y2 - y1) / 2) ** 2)
            diff_dist = 1 - dist / max_dist 
            region_data[region_name]["norm_dist"] += diff_dist + 0.1 if diff_dist < 0.9 else 1.0 #0.1 bias added to have more touches close to 100%
            region_data[region_name]["touch_xs"].append(tx)
            region_data[region_name]["touch_ys"].append(ty)


    # Build processed output per region
    overall_accuracy = 0.0
    fingers_used = 0
    for region_name, data in region_data.items():
        x1, y1, x2, y2 = region_lookup[region_name]
        key = region_to_key.get(region_name, region_name) if region_to_key else region_name
        display_name = KEY_TO_FINGER.get(key, key)
        total = data["touches"]
        overall_accuracy += data["norm_dist"] / total if total > 0 else 0.0
        if total == 0:
            #if display names starts with lego, dont append
            if not display_name.startswith("Lego"):
                processed.append({
                    "region": display_name,
                    "message": "No touches detected for this region"
                })
        else:
            fingers_used += 1
            precis_x = max(0.0, 1.0 - statistics.stdev(data["touch_xs"]) / ((x2 - x1) / 2)) if total > 1 else 1.0
            precis_y = max(0.0, 1.0 - statistics.stdev(data["touch_ys"]) / ((y2 - y1) / 2)) if total > 1 else 1.0
            processed.append({
                "region": display_name,
                "region_percentages": {
                    "center":       data["center"] / total,
                    "top_left":     data["top_left"] / total,
                    "bottom_left":  data["bottom_left"] / total,
                    "top_right":    data["top_right"] / total,
                    "bottom_right": data["bottom_right"] / total,
                },
                "accuracy_score": data["norm_dist"] / total,
                "precision_score": {
                    "overall": (precis_x + precis_y) / 2,
                    "x": precis_x,
                    "y": precis_y
                }
                
            })

    overall_accuracy = overall_accuracy / fingers_used if fingers_used > 0 else 0.0
    processed.append({
        "overall_accuracy": overall_accuracy,
    })

    # Compute per-finger avg and max across all entries
    finger_stats = {}
    for entry in touch_history:
        touched_regions = entry.get("touches", {}).keys()
        touched_fingers = {KEY_TO_FINGER.get(region_to_key.get(r), None) for r in touched_regions}
        for finger_name, length in entry.get("finger_lengths", {}).items():
            if finger_name in touched_fingers:
                if finger_name not in finger_stats:
                    finger_stats[finger_name] = []
                finger_stats[finger_name].append(finger_factor * length * FINGER_PLAY_SCALE.get(finger_name, 1.0))

    finger_summary = {
        finger: {
            "avg": sum(lengths) / len(lengths),
            "max": max(lengths),
        }
        for finger in FINGER_NAMES
        if finger in finger_stats
        for lengths in [finger_stats[finger]]
    }

    return processed, finger_summary, finger_factor

def region_to_finger_map(clickable_regions):
    """Map each game brick (left-to-right) to a finger name.

    The awsd row reads left-to-right as a,w,s,d, which maps to Pege,
    Lange, Ringe, Lille. The tommel-aktivator brick sits at the BOTTOM of
    the (flipped) image (largest y1) and is excluded so it does not steal
    one of the four finger labels.
    """
    finger_names = ["Pege", "Lange", "Ringe", "Lille"]

    candidate_regions = list(clickable_regions)
    # Drop the activator (bottom-most box) when there are more bricks than fingers.
    if len(candidate_regions) > len(finger_names):
        activator_region = max(candidate_regions, key=lambda r: r[1])
        candidate_regions.remove(activator_region)

    sorted_regions = sorted(candidate_regions, key=lambda r: r[0])
    return {r[4]: finger_names[i] for i, r in enumerate(sorted_regions) if i < len(finger_names)}

def _draw_hand_guide(img, progress_pct, hand_detected, stable, warmup_frames):
    """
    Draws a large top-down hand outline on the live feed (~half the frame height)
    so the user knows to place their hand spread out flat on the surface.
    """
    h, w = img.shape[:2]

    # Hand silhouette height = half the frame height
    S = (h * 0.5) / 220.0   # scale: reference design is 220 units tall

    # Centre the hand in the frame
    cx = w // 2
    # Wrist sits near the bottom so fingers point UP, matching the 180-flipped
    # camera view (where the user's physical hand appears with fingers pointing
    # toward the top of the displayed frame).
    wrist_y = int(h * 0.78)

    # Outline + fill colour depends on whether a hand is currently detected:
    #   red   -> no hand visible, user should place their hand on the board
    #   green -> hand visible, measurement is happening
    # OpenCV uses BGR.
    if hand_detected:
        col       = (60, 220, 60)    # bright green outline
        fill_col  = (40, 160, 40)    # darker green fill
    else:
        col       = (60, 60, 220)    # bright red outline
        fill_col  = (40, 40, 160)    # darker red fill
    thick = max(2, int(3 * S))

    def pt(x, y):
        """Convert hand-local coords (origin = wrist centre, y up from wrist)
        to image coords. x is NOT negated so the hand is mirrored left-to-right
        relative to the original, matching the camera's horizontal flip."""
        return (int(cx + x * S), int(wrist_y - y * S))

    # ── Silhouette polygon ────────────────────────────────────────────────────
    # A single closed contour tracing the outline of a spread right hand
    # viewed from above, fingers pointing up. Coords (x right, y up from wrist).
    contour = np.array([
        # Wrist
        pt(-50,   0), pt(-50,  10),
        # Tommel side of palm → tommel base
        pt(-75,  50), pt(-90,  70),
        # Tommel tip
        pt(-115, 115),
        # Tommel inner edge back to pege web
        pt(-80, 100), pt(-68,  80),
        # Pege finger
        pt(-58,  80), pt(-65, 175),
        # Pege tip → ringe side
        pt(-45, 185), pt(-35, 175),
        # Lange finger
        pt(-28,  90), pt(-12,  90), pt(-15, 200),
        # Lange tip
        pt(  0, 210), pt( 15, 200),
        # Ringe finger
        pt( 12,  90), pt( 28,  90), pt( 22, 185),
        # Ringe tip
        pt( 38, 195), pt( 50, 185),
        # Lille finger
        pt( 45,  78), pt( 60,  78), pt( 55, 160),
        # Lille tip
        pt( 68, 168), pt( 78, 158),
        # Lille outer edge → right palm → wrist
        pt( 75,  55), pt( 65,  20), pt( 50,   0),
    ], dtype=np.int32)

    # Filled semi-transparent silhouette
    overlay = img.copy()
    cv2.fillPoly(overlay, [contour], fill_col)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Solid outline on top
    cv2.polylines(img, [contour], isClosed=True, color=col, thickness=thick,
                  lineType=cv2.LINE_AA)

    # ── Short status line at the top of the frame ─────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not hand_detected:
        msg, mcol = "Spred haanden fladt ud paa braettet", (80, 180, 255)
    elif stable < warmup_frames:
        msg, mcol = "Hold stille...",               (80, 220, 255)
    else:
        msg, mcol = "Maaler nu! Bliv ved med at holde haanden stille",      (80, 240, 120)

    (tw, th), _ = cv2.getTextSize(msg, font, 0.8, 2)
    cv2.putText(img, msg, (w // 2 - tw // 2, 40),
                font, 0.8, mcol, 2, cv2.LINE_AA)


def finger_length_check(cap, tracker):
    # Flush all stale tracking state carried over from lego detection.
    # Without this, prev_landmarks from the previous phase bleeds into the
    # smoothing on the first frames here, giving incorrect positions even
    # though use_last_known=False is set.
    tracker.results = None
    tracker.prev_landmarks.clear()
    tracker.missed_frames.clear()

    samples_needed = 90
    warmup_frames = 90
    FINGER_TIP_KNUCKLE = {
        "Pege": (8, 5), "Lange": (12, 9),
        "Ringe": (16, 13), "Lille": (20, 17)
    }
    samples = {n: [] for n in FINGER_TIP_KNUCKLE}
    stable = 0

    print("\n===== FINGER LÆNGDE MÅLING =====")
    print("Placer din hånd fladt på det hvide lego bræt og hold stille")
    print("tryk på \"q\" for at afslutte måling")

    last_reported_progress = -1
    warmup_announced = False
    hand_was_lost = False

    while min(len(v) for v in samples.values()) < samples_needed:
        cap.grab()
        success, img = cap.retrieve()
        if not success:
            print("Kamera kunne ikke måle.")
            break

        # Match the 180-degree flip applied in detect_legos and run_hand_tracking_loop
        # so MediaPipe sees the same orientation and the hand-guide overlay aligns
        # with the user's actual hand placement.
        img = cv2.flip(img, -1)

        img = tracker.find_hands(img, draw=True)
        pos = tracker.get_positions(img, hand_no=0, use_last_known=False)
        hand_ok = len(pos) > 20

        # Track hand presence
        if not hand_ok:
            if not hand_was_lost:
                print(" Ingen hånd at vise. Placer din hånd på brættet og hold stille...")
                hand_was_lost = True
                warmup_announced = False
            stable = 0
        else:
            if hand_was_lost:
                print(" Hånd fundet. hold den stille...")
                hand_was_lost = False
            stable += 1

            # Announce once when warmup completes
            if stable == warmup_frames and not warmup_announced:
                print("  Måling starter nu! Hold stadigt...")
                warmup_announced = True

            # Record samples only after warmup
            if stable >= warmup_frames:
                for name, (tip_id, knuckle_id) in FINGER_TIP_KNUCKLE.items():
                    samples[name].append(px_dist(
                        (pos[tip_id][1],     pos[tip_id][2]),
                        (pos[knuckle_id][1], pos[knuckle_id][2])
                    ))


        # Progress reporting in 10% increments
        progress_pct = int(min(len(samples["Lange"]) / samples_needed, 1.0) * 100)
        if progress_pct >= last_reported_progress + 10:
            print(f" hånd: {progress_pct}%")
            last_reported_progress = progress_pct - (progress_pct % 10)

        # ── Guide overlay: top-down hand silhouette ───────────────────────────
        _draw_hand_guide(img, progress_pct, hand_ok, stable, warmup_frames)
        # ─────────────────────────────────────────────────────────────────────

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Måling afbrudt af bruger.")
            break

    # Final results
    result = {name: round(sum(v) / len(v), 2) if v else 0.0
              for name, v in samples.items()}
    print("\nFinger laengder (pixels):")
    for name, avg in result.items():
        print(f"  {name:<7}: {avg:.2f} px")
    print("===============================\n")

    # Show completion message on the live feed until Enter is pressed in the window
    done_msg1 = "Din haand er nu maalt.  Fjern haanden."
    done_msg2 = "Tryk Enter her for at fortsaette..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("\nDin haand er nu maalt. Fjern haanden og tryk Enter i kameravinduet for at fortsaette...")
    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.flip(img, -1)
        else:
            img = 255 * np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h // 2 - 65), (w, h // 2 + 65), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        for i, line in enumerate([done_msg1, done_msg2]):
            sz, _ = cv2.getTextSize(line, font, 0.75, 2)
            x = (w - sz[0]) // 2
            y = h // 2 - 12 + i * 46
            cv2.putText(img, line, (x, y), font, 0.75, (80, 255, 160), 2, cv2.LINE_AA)
        cv2.imshow("Hand Tracking", img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter or newline
            break
        if key == ord('q'):
            break
    return result

def run_hand_tracking_loop(cap, tracker, control_socket, region_to_key, model, lego_bricks=(), game_process=None):
    """
    Main hand-tracking loop.

    Behaviour is controlled by the module-level USE_DEPTH_CALIBRATION flag:
      True  – runs calibration first; a touch requires the finger to be inside a
               box AND its geometry distance to exceed the calibrated threshold.
      False – skips calibration; a touch is registered as soon as a finger tip
               enters a bounding box (pure 2-D, original behaviour).
    """
    active_touch_keys = set()

    if region_to_key:
        mapping_text = ", ".join([f"{name}->{key}" for name, key in region_to_key.items()])
        print(f"Key bindings: {mapping_text}")

    # --- Optional depth calibration ---
    thresholds = {}
    directions = {}

    if USE_DEPTH_CALIBRATION:
        thresholds, directions = run_calibration(cap, tracker, list(tracker.clickable_regions))
        if thresholds is None:
            print("Calibration cancelled.")
            return
        print("Calibration complete! Starting tracking...")

    touch_states = {}  # always exists; stays empty when depth calibration is off

    prev_frame_time = time.time()
    frame = 0
    touch_history = []

    while True:
        cap.grab()
        success, img = cap.retrieve()
        if not success:
            break

        img = cv2.flip(img, -1)

        frame += 1
        FRAME_CHECK = (frame % 5 == 0)

        curr_frame_time = time.time()
        fps = 1.0 / (curr_frame_time - prev_frame_time) if curr_frame_time > prev_frame_time else 0.0
        prev_frame_time = curr_frame_time

        img = tracker.find_hands(img)


        touched_keys_this_frame = set()
        # Maps region_name -> (pixel_x, pixel_y) for every box confirmed touched this frame.
        # pixel_x, pixel_y are absolute camera-frame coordinates of the touching finger tip.
        touch_pixel_positions = {}
        cached_finger_tips = []   # populated inside the hand-detection block below

        # --- Depth touch states (only meaningful when USE_DEPTH_CALIBRATION is True) ---
        if USE_DEPTH_CALIBRATION:
            geom = get_finger_geometry(tracker, img)
            touch_states = {}
            for name, x, y, dist_sum in geom:
                boundary  = thresholds.get(name, 0.0)
                direction = directions.get(name, 1)
                touching  = (dist_sum >= boundary) if direction == 1 else (dist_sum <= boundary)
                touch_states[name] = touching

        if not THUMB_ACTIVATER:
            # Thumb activation disabled: buttons always work without thumb
            thumb_active = True
        else:
            # Thumb activation enabled: thumb must be in the activation region
            thumb_active = False

        # --- Determine which keys are touched this frame ---
        if tracker.results and tracker.results.multi_hand_landmarks:
            for hand_idx in range(len(tracker.results.multi_hand_landmarks)):
                # Cache finger tips ONCE per hand per frame to avoid double-smoothing.
                # Calling get_all_finger_tips() multiple times per frame causes
                # get_positions() to re-apply smoothing on already-smoothed data,
                # which can shift tip coordinates enough to miss a box edge on the
                # second scan even though the touch was correctly detected on the first.
                cached_finger_tips = tracker.get_all_finger_tips(img, hand_no=hand_idx)

                touched_regions = tracker.check_touching_region(img, hand_no=hand_idx)

                # Get the activation region (bottom-most box by y1)
                activation_region = max(tracker.clickable_regions, key=lambda r: r[1])
                ax1, ay1, ax2, ay2, _ = activation_region

                # Check if thumb is in the activation region (only required when THUMB_ACTIVATER is True)
                if THUMB_ACTIVATER:
                    if cached_finger_tips:
                        tip_id, tx, ty, _ = cached_finger_tips[0]  # landmark 4 = thumb tip
                        if ax1 <= tx <= ax2 and ay1 <= ty <= ay2:
                            thumb_active = True

                # Always fire 'f' key when thumb tip touches the thumb brick.
                # Games can use this to advance to the next picture / confirm.
                if cached_finger_tips:
                    tip_id, tx, ty, _ = cached_finger_tips[0]  # landmark 4 = thumb tip
                    if ax1 <= tx <= ax2 and ay1 <= ty <= ay2:
                        touched_keys_this_frame.add('f')

                # Only allow keys to be triggered if thumb is active
                for region_name in touched_regions:
                    if region_name in region_to_key and thumb_active:
                        if USE_DEPTH_CALIBRATION:
                            # Only count the touch if the CORRECT finger passes the depth threshold.
                            key = region_to_key[region_name]
                            allowed_finger = KEY_ALLOWED_FINGER.get(key)
                            x1, y1, x2, y2 = next(
                                (r[:4] for r in tracker.clickable_regions if r[4] == region_name),
                                (None, None, None, None)
                            )
                            depth_confirmed = False
                            for tip in cached_finger_tips:  # use cached to avoid double-smoothing
                                tip_id, tx, ty, tz = tip
                                finger_name = (
                                    FINGER_NAMES[FINGER_TIP_IDS.index(tip_id)]
                                    if tip_id in FINGER_TIP_IDS
                                    else None
                                )
                                if (
                                    finger_name == allowed_finger
                                    and touch_states.get(finger_name, False)
                                    and x1 is not None
                                    and x1 <= tx <= x2
                                    and y1 <= ty <= y2
                                ):
                                    depth_confirmed = True
                                    touch_pixel_positions[region_name] = (tx, ty)
                                    break
                            if depth_confirmed:
                                touched_keys_this_frame.add(key)
                        else:
                            # Pure 2-D: only register if the correct finger is inside the box.
                            key = region_to_key[region_name]
                            allowed_finger = KEY_ALLOWED_FINGER.get(key)
                            x1, y1, x2, y2 = next(
                                (r[:4] for r in tracker.clickable_regions if r[4] == region_name),
                                (None, None, None, None)
                            )
                            if x1 is not None:
                                for tip in cached_finger_tips:
                                    tip_id, tx, ty, _ = tip
                                    finger_name = (
                                        FINGER_NAMES[FINGER_TIP_IDS.index(tip_id)]
                                        if tip_id in FINGER_TIP_IDS else None
                                    )
                                    if (finger_name == allowed_finger
                                            and x1 <= tx <= x2 and y1 <= ty <= y2):
                                        touched_keys_this_frame.add(key)
                                        touch_pixel_positions[region_name] = (tx, ty)
                                        break
                                # Fallback accuracy position when correct finger is confirmed
                                # but pixel position wasn't captured (rare jitter edge case).
                                if key in touched_keys_this_frame and region_name not in touch_pixel_positions:
                                    touch_pixel_positions[region_name] = (
                                        (x1 + x2) / 2, (y1 + y2) / 2
                                    )
                                
        if touch_pixel_positions:
            pos = tracker.get_positions(img)
            geom_snapshot = {}
            if len(pos) >= 21:
                finger_tip_knuckle = {
                    "Pege":  (8, 5),
                    "Lange": (12, 9),
                    "Ringe":   (16, 13),
                    "Lille":  (20, 17),
                }
                for finger_name, (tip_id, knuckle_id) in finger_tip_knuckle.items():
                    tip     = (pos[tip_id][1],     pos[tip_id][2])
                    knuckle = (pos[knuckle_id][1], pos[knuckle_id][2])
                    geom_snapshot[finger_name] = px_dist(tip, knuckle)

            touch_history.append({
                "frame":          frame,
                "timestamp":      curr_frame_time,
                "touches":        dict(touch_pixel_positions),
                "finger_lengths": geom_snapshot
            })
        pressed_now  = touched_keys_this_frame - active_touch_keys
        released_now = active_touch_keys - touched_keys_this_frame
        for key in pressed_now:
            set_key_state(key, True)
        for key in released_now:
            set_key_state(key, False)
        active_touch_keys = touched_keys_this_frame

        # Build per-key accuracy: 1.0 = perfect centre, 0.0 = at the edge of the box.
        # Formula: norm_dist = 1 - dist / max_dist, clamped to [0, 1].
        # max_dist uses the shorter half-dimension (min of half-width, half-height)
        # so that touching anywhere within the "inner circle" of the box scores ≥ 0,
        # and touching the very edge of the short side is 0 rather than going negative.
        # The old diagonal max_dist caused edge-of-side touches to score as low as 17 %
        # even when the finger was fully on the plate.
        key_accuracy = {}
        for region_name, (tx, ty) in touch_pixel_positions.items():
            if region_name in region_to_key:
                box = next(
                    (r[:4] for r in tracker.clickable_regions if r[4] == region_name),
                    None
                )
                if box is not None:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    dist     = math.sqrt((tx - center_x) ** 2 + (ty - center_y) ** 2)
                    # Use the shorter half-dimension so any touch that lands inside the
                    # inscribed circle scores > 0, and edge touches are penalised fairly.
                    max_dist = min((x2 - x1) / 2, (y2 - y1) / 2)
                    norm_dist = max(0.0, 1.0 - (dist / max_dist)) if max_dist > 0 else 1.0
                    key_accuracy[region_to_key[region_name]] = round(norm_dist, 3)

        send_control_state(control_socket, active_touch_keys, key_accuracy)

        # Draw boxes after accuracy is computed so we can show live accuracy inside them.
        # Pass cached_finger_tips to avoid an extra smoothing call.
        draw_lego_boxes(img, tracker, region_to_key, touch_states,
                        use_depth=USE_DEPTH_CALIBRATION,
                        finger_tips_cache=cached_finger_tips,
                        key_accuracy=key_accuracy)

        # Small hints at the bottom
        cv2.putText(img, "E: genskanning af brikker", (8, CAMERA_HEIGHT - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Q: afslut session", (8, CAMERA_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow("Hand Tracking", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Treat closing the window (X button) the same as pressing Q
        if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break
        # Also stop tracking when the game window has been closed
        if game_process is not None and game_process.poll() is not None:
            break
        elif key == ord('e'):
            print("Re-detecting lego bricks...")
            tracker.clickable_regions = []     
            detect_legos(cap, model, tracker, detection_duration=2)
            region_to_key = choose_key_bindings(tracker.clickable_regions)


    for key in list(active_touch_keys):
        set_key_state(key, False)
    send_control_state(control_socket, set())

    return touch_history

def generate_feedback(history_file="touch_history.json"):
    with open(history_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    # data[0] is metadata; sessions are all entries that have a "processed" key
    sessions = [s for s in data if "processed" in s]

    if not sessions:
        print("Ingen sessioner fundet.")
        return

    latest = sessions[-1]

    if len(sessions) < 2:
        print("First session recorded — no previous session to compare against yet.")
        previous = None
    else:
        previous = sessions[-2]

    print("\n===== DENNE SESSIONS FEEDBACK =====\n")

    # Translate legacy English finger names to Danish
    _NAME_MAP = {"Thumb": "Tommel", "Index": "Pege", "Middle": "Lange", "Ring": "Ringe", "Pinky": "Lille"}
    def _n(name):
        return _NAME_MAP.get(name, name)

    # --- Per-region accuracy and precision comparison ---
    def get_accuracy_scores(session):
        scores = {}
        for entry in session["processed"]:
            if "region" in entry and "accuracy_score" in entry:
                scores[_n(entry["region"])] = entry["accuracy_score"]
        return scores

    def get_overall_accuracy(session):
        for entry in session["processed"]:
            if "overall_accuracy" in entry:
                return entry["overall_accuracy"]
        return None

    def get_precision_scores(session):
        scores = {}
        for entry in session["processed"]:
            if "region" in entry and "precision_score" in entry:
                scores[_n(entry["region"])] = entry["precision_score"]["overall"]
        return scores

    def get_steadiness_scores(session):
        # New format: steadiness_score embedded in processed entries (0-1 scale)
        scores = {}
        for entry in session.get("processed", []):
            if "region" in entry and "steadiness_score" in entry:
                scores[_n(entry["region"])] = entry["steadiness_score"]
        # Fallback: old format stored as top-level "steadiness" dict (values were 0-100)
        if not scores and session.get("steadiness"):
            for finger, val in session["steadiness"].items():
                scores[_n(finger)] = val / 100.0
        return scores

    def get_overall_steadiness(session):
        # New format: overall_steadiness embedded in processed overall entry (0-1 scale)
        for entry in session.get("processed", []):
            if "overall_steadiness" in entry:
                return entry["overall_steadiness"]
        # Fallback: old format
        vals = list(session.get("steadiness", {}).values())
        return (sum(vals) / len(vals)) / 100.0 if vals else None

    def get_overall_precision(session):
        # New format: overall_precision embedded in processed overall entry
        for entry in session.get("processed", []):
            if "overall_precision" in entry:
                return entry["overall_precision"]
        # Fallback: compute from individual precision scores
        precisions = []
        for entry in session.get("processed", []):
            if "precision_score" in entry:
                precisions.append(entry["precision_score"]["overall"])
        return sum(precisions) / len(precisions) if precisions else None


    latest_a_scores = get_accuracy_scores(latest)
    latest_p_scores = get_precision_scores(latest)
    latest_s_scores = get_steadiness_scores(latest)
    if previous is not None:
        previous_a_scores = get_accuracy_scores(previous)
        previous_p_scores = get_precision_scores(previous)
        previous_s_scores = get_steadiness_scores(previous)
    else:
        previous_a_scores = {}
        previous_p_scores = {}
        previous_s_scores = {}

    # Find all-time best per region
    all_time_best_accuracy = {}
    all_time_best_precision = {}
    all_time_best_steadiness = {}
    for session in sessions:
        for region, score in get_accuracy_scores(session).items():
            if region not in all_time_best_accuracy or score > all_time_best_accuracy[region]:
                all_time_best_accuracy[region] = score
        for region, score in get_precision_scores(session).items():
            if region not in all_time_best_precision or score > all_time_best_precision[region]:
                all_time_best_precision[region] = score
        for finger, score in get_steadiness_scores(session).items():
            if finger not in all_time_best_steadiness or score > all_time_best_steadiness[finger]:
                all_time_best_steadiness[finger] = score

    # --- Per-finger accuracy ---
    print("-- Accuracy for hver finger --")
    for region in set(list(latest_a_scores.keys()) + list(previous_a_scores.keys())):
        latest_val = latest_a_scores.get(region)
        prev_val   = previous_a_scores.get(region)
        if latest_val is None:
            print(f"  {region}: ingen data for denne session")
            continue
        if prev_val is None:
            print(f"  {region}: {100*latest_val:.2f}% (ingen tidligere data at sammenligne)")
            continue
        diff = latest_val - prev_val
        direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
        improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
        record = "  Ny personlig rekord!" if all_time_best_accuracy.get(region) == latest_val else ""
        print(f"  {region}: {100*latest_val:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff)/prev_val:.2f}%){record}")

    # --- Per-finger precision ---
    print("\n-- Precision for hver finger --")
    for region in set(list(latest_p_scores.keys()) + list(previous_p_scores.keys())):
        latest_val = latest_p_scores.get(region)
        prev_val   = previous_p_scores.get(region)
        if latest_val is None:
            print(f"  {region}: ingen data for denne session")
            continue
        if prev_val is None:
            print(f"  {region}: {100*latest_val:.2f}% (ingen tidligere data at sammenligne)")
            continue
        diff = latest_val - prev_val
        direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
        improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
        record = "  Ny personlig rekord!" if all_time_best_precision.get(region) == latest_val else ""
        print(f"  {region}: {100*latest_val:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff)/prev_val:.2f}%){record}")

    # --- Per-finger stabilitets score ---
    finger_order = ["Pege", "Lange", "Ringe", "Lille"]
    ordered_s_regions = [f for f in finger_order if f in set(list(latest_s_scores.keys()) + list(previous_s_scores.keys()))]
    if ordered_s_regions:
        print("\n-- Stabilitets score for hver finger --")
        for finger in ordered_s_regions:
            latest_val = latest_s_scores.get(finger)
            prev_val   = previous_s_scores.get(finger)
            if latest_val is None:
                print(f"  {finger}: ingen data for denne session")
                continue
            if prev_val is None:
                print(f"  {finger}: {100*latest_val:.2f}% (ingen tidligere data at sammenligne)")
                continue
            diff = latest_val - prev_val
            direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
            improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
            record = "  Ny personlig rekord!" if all_time_best_steadiness.get(finger) == latest_val else ""
            print(f"  {finger}: {100*latest_val:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff):.2f}%){record}")

    #noter:
    #formidle forskellen godt på dansk hvordan precision og accuracy blir forbedret
    #gøre brikken mindre hvis høj precision og lav accuracy
    #lave tretris spil måske. justerende hastighed af faldende brik i forhold til hvor meget input der er

    # --- Overall accuracy ---
    print("\n-- Overordnet accuracy --")
    latest_a_overall  = get_overall_accuracy(latest)
    previous_a_overall = get_overall_accuracy(previous) if previous is not None else None
    all_time_a_overall_vals = [get_overall_accuracy(s) for s in sessions if get_overall_accuracy(s) is not None]
    all_time_a_overall = max(all_time_a_overall_vals) if all_time_a_overall_vals else None
    if latest_a_overall is not None and previous_a_overall is not None:
        diff = latest_a_overall - previous_a_overall
        direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
        improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
        record = "  Ny personlig rekord!" if all_time_a_overall == latest_a_overall else ""
        print(f"  {100*latest_a_overall:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff)/previous_a_overall:.2f}%){record}")
    elif latest_a_overall is not None:
        print(f"  {100*latest_a_overall:.2f}% (ingen tidligere data at sammenligne)")

    # --- Overall precision ---
    print("\n-- Overordnet precision --")
    latest_p_overall  = get_overall_precision(latest)
    previous_p_overall = get_overall_precision(previous) if previous is not None else None
    all_time_p_overall_vals = [get_overall_precision(s) for s in sessions if get_overall_precision(s) is not None]
    all_time_p_overall = max(all_time_p_overall_vals) if all_time_p_overall_vals else None
    if latest_p_overall is not None and previous_p_overall is not None:
        diff = latest_p_overall - previous_p_overall
        direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
        improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
        record = "  Ny personlig rekord!" if all_time_p_overall == latest_p_overall else ""
        print(f"  {100*latest_p_overall:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff)/previous_p_overall:.2f}%){record}")
    elif latest_p_overall is not None:
        print(f"  {100*latest_p_overall:.2f}% (ingen tidligere data at sammenligne)")

    # --- Overall stabilitets score ---
    print("\n-- Overordnet stabilitets score --")
    latest_s_overall  = get_overall_steadiness(latest)
    previous_s_overall = get_overall_steadiness(previous) if previous is not None else None
    all_time_s_overall_vals = [get_overall_steadiness(s) for s in sessions if get_overall_steadiness(s) is not None]
    all_time_s_overall = max(all_time_s_overall_vals) if all_time_s_overall_vals else None
    if latest_s_overall is not None and previous_s_overall is not None:
        diff = latest_s_overall - previous_s_overall
        direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
        improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
        record = "  Ny personlig rekord!" if all_time_s_overall == latest_s_overall else ""
        print(f"  {100*latest_s_overall:.2f}% {direction} ({improvement} fra sidste session med {100*abs(diff):.2f}%){record}")
    elif latest_s_overall is not None:
        print(f"  {100*latest_s_overall:.2f}% (ingen tidligere data at sammenligne)")

    # --- Finger length comparison ---
    print("\n-- Finger længde (cm) --")
    latest_fingers   = {_n(k): v for k, v in latest.get("finger_summary", {}).items()}
    previous_fingers = {_n(k): v for k, v in (previous.get("finger_summary", {}) if previous is not None else {}).items()}
    all_time_finger_best = {}
    for session in sessions:
        for finger, stats in session.get("finger_summary", {}).items():
            finger = _n(finger)
            if finger not in all_time_finger_best or stats["max"] > all_time_finger_best[finger]:
                all_time_finger_best[finger] = stats["max"]

    for finger in set(list(latest_fingers.keys()) + list(previous_fingers.keys())):
        latest_avg = latest_fingers.get(finger, {}).get("avg")
        latest_max = latest_fingers.get(finger, {}).get("max")
        prev_avg   = previous_fingers.get(finger, {}).get("avg")
        prev_max   = previous_fingers.get(finger, {}).get("max")

        if latest_avg is None:
            print(f"  {finger}: ingen data for denne session")
            continue

        # Avg line
        if prev_avg is None:
            print(f"  {finger} (gns): {latest_avg:.2f}cm (ingen tidligere data at sammenligne)")
        else:
            diff = latest_avg - prev_avg
            direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
            improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
            print(f"  {finger} (gns): {latest_avg:.2f}cm {direction} ({improvement} fra sidste session med {abs(diff):.2f}cm)")

        # Max line
        if prev_max is None:
            print(f"  {finger} (max): {latest_max:.2f}cm (ingen tidligere data at sammenligne)")
        else:
            diff = latest_max - prev_max
            direction   = "▲" if diff > 0 else "▼" if diff < 0 else "="
            improvement = "forbedret" if diff > 0 else "faldet" if diff < 0 else "uændret"
            record = "   Ny personlig rekord!" if all_time_finger_best.get(finger) == latest_max else ""
            print(f"  {finger} (max): {latest_max:.2f}cm {direction} ({improvement} fra sidste session med {abs(diff):.2f}cm){record}")
    
    # --- All time top results ---
    print()
    print("===== DINE REKORDER =====")
    print()
    print("-- Accuracy for hver finger --")
    for region, score in all_time_best_accuracy.items():
        session_idx = next(i+1 for i, s in enumerate(sessions) if get_accuracy_scores(s).get(region) == score)
        print(f"  {region}: {100*score:.2f}% (session {session_idx})")
    print()
    print("-- Precision for hver finger --")
    for region, score in all_time_best_precision.items():
        session_idx = next(i+1 for i, s in enumerate(sessions) if get_precision_scores(s).get(region) == score)
        print(f"  {region}: {100*score:.2f}% (session {session_idx})")
    print()
    print("-- Stabilitets score for hver finger --")
    for finger, score in all_time_best_steadiness.items():
        session_idx = next(i+1 for i, s in enumerate(sessions) if get_steadiness_scores(s).get(finger) == score)
        print(f"  {finger}: {100*score:.2f}% (session {session_idx})")
    print()
    print(f"-- Overordnet accuracy --")
    print(f"  {100*all_time_a_overall:.2f}% (session {next(i+1 for i, s in enumerate(sessions) if get_overall_accuracy(s) == all_time_a_overall)})")
    print()
    print(f"-- Overordnet precision --")
    print(f"  {100*all_time_p_overall:.2f}% (session {next(i+1 for i, s in enumerate(sessions) if get_overall_precision(s) == all_time_p_overall)})")
    print()
    if all_time_s_overall is not None:
        print(f"-- Overordnet stabilitets score --")
        print(f"  {100*all_time_s_overall:.2f}% (session {next(i+1 for i, s in enumerate(sessions) if get_overall_steadiness(s) == all_time_s_overall)})")
    print()
    print("-- Finger længde (cm) --")
    all_time_finger_best_avg = {}
    for session in sessions:
        for finger, stats in session.get("finger_summary", {}).items():
            finger = _n(finger)
            if finger not in all_time_finger_best_avg or stats["avg"] > all_time_finger_best_avg[finger]:
                all_time_finger_best_avg[finger] = stats["avg"]

    for finger in all_time_finger_best:
        best_avg = all_time_finger_best_avg.get(finger)
        best_max = all_time_finger_best.get(finger)

        max_session = next(i+1 for i, s in enumerate(sessions)
                          if {_n(k): v for k, v in s.get("finger_summary", {}).items()}.get(finger, {}).get("max") == best_max)

        #print(f"  {finger} (gns): {best_avg:.2f}cm (session {avg_session})")
        print(f"  {finger} (max): {best_max:.2f}cm (session {max_session})")

    print()
    print("===== DINE MAL =====")
    print()
    print("-- Dit maal for hver finger laengde er --")
    goal_lengths = {_n(k): v for k, v in (data[0].get("goal_finger_lengths", {}) if len(data) > 0 else {}).items()}
    if goal_lengths:
        print("-- Maal finger laengder (cm) --")
        for finger, goal in goal_lengths.items():
            best_max = all_time_finger_best.get(finger)

            if best_max is None:
                print(f"  {finger}: maal {goal:.2f}cm (ingen data endnu)")
                continue

            diff      = best_max - goal
            direction = "up" if diff > 0 else "down" if diff < 0 else "="
            label     = "over" if diff > 0 else "under" if diff < 0 else "paa"

            print(f"  {finger}: maal {goal:.2f}cm  |  rekord {best_max:.2f}cm {direction} ({abs(diff):.2f}cm {label} maalet)")


    print("\n============================\n")


def show_feedback_window(history_file="touch_history.json"):
    """
    Opens a two-slide matplotlib window.
      Slide 1 – top row: Nøjagtighed / Præcision / Stabilitet graphs.
                bottom row: max finger-length graph per finger.
                Each graph has a Danish description below it.
      Slide 2 – celebration / new records + all-time records + session history.
    Navigation: ← / → buttons at the bottom.  No console output.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Button

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(history_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    sessions = [s for s in data if "processed" in s]
    if not sessions:
        return

    # ── Name / label helpers ──────────────────────────────────────────────────
    _NAME_MAP = {
        "Thumb": "Tommel", "Index": "Pege", "Middle": "Lange",
        "Ring": "Ringe", "Pinky": "Lille"
    }
    def _n(name):
        return _NAME_MAP.get(name, name)

    FINGER_ORDER = ["Pege", "Lange", "Ringe", "Lille"]
    FINGER_LABEL = {
        "Pege": "Pegefinger", "Lange": "Langefinger",
        "Ringe": "Ringfinger", "Lille": "Lillefinger",
    }
    COLORS = {
        "Pege": "#FFD700", "Lange": "#B066FF",
        "Ringe": "#22C55E", "Lille": "#EF4444", "Samlet": "#60A5FA",
    }

    # ── Goal finger lengths from metadata ─────────────────────────────────────
    goal_finger_lengths = {}
    meta = next((d for d in data if "goal_finger_lengths" in d), None)
    if meta:
        for f, v in meta["goal_finger_lengths"].items():
            goal_finger_lengths[_n(f)] = round(v, 1)

    # ── Collect per-session time-series ───────────────────────────────────────
    acc_series   = {f: [] for f in FINGER_ORDER}
    prec_series  = {f: [] for f in FINGER_ORDER}
    stead_series = {f: [] for f in FINGER_ORDER}
    flen_series  = {f: [] for f in FINGER_ORDER}
    overall_acc_s = []; overall_prec_s = []; overall_stead_s = []

    for i, session in enumerate(sessions):
        sx = i + 1
        for entry in session.get("processed", []):
            if "region" in entry:
                finger = _n(entry["region"])
                if finger in acc_series and "accuracy_score" in entry:
                    acc_series[finger].append((sx, entry["accuracy_score"] * 100))
                if finger in prec_series and "precision_score" in entry:
                    prec_series[finger].append((sx, entry["precision_score"]["overall"] * 100))
                if finger in stead_series and "steadiness_score" in entry:
                    stead_series[finger].append((sx, entry["steadiness_score"] * 100))
            if "overall_accuracy" in entry and entry["overall_accuracy"] is not None:
                overall_acc_s.append((sx, entry["overall_accuracy"] * 100))
            if "overall_precision" in entry and entry["overall_precision"] is not None:
                overall_prec_s.append((sx, entry["overall_precision"] * 100))
            if "overall_steadiness" in entry and entry["overall_steadiness"] is not None:
                overall_stead_s.append((sx, entry["overall_steadiness"] * 100))
        for finger, stats in session.get("finger_summary", {}).items():
            fn = _n(finger)
            if fn in flen_series:
                flen_series[fn].append((sx, round(stats["max"], 1)))

    # ── Helper extractors ─────────────────────────────────────────────────────
    def get_accuracy_scores(session):
        return {_n(e["region"]): e["accuracy_score"]
                for e in session["processed"]
                if "region" in e and "accuracy_score" in e}

    def get_precision_scores(session):
        return {_n(e["region"]): e["precision_score"]["overall"]
                for e in session["processed"]
                if "region" in e and "precision_score" in e}

    def get_steadiness_scores(session):
        scores = {_n(e["region"]): e["steadiness_score"]
                  for e in session.get("processed", [])
                  if "region" in e and "steadiness_score" in e}
        if not scores and session.get("steadiness"):
            for f, v in session["steadiness"].items():
                scores[_n(f)] = v / 100.0
        return scores

    def get_overall_accuracy(s):
        for e in s.get("processed", []):
            if "overall_accuracy" in e: return e["overall_accuracy"]
        return None

    def get_overall_precision(s):
        for e in s.get("processed", []):
            if "overall_precision" in e: return e["overall_precision"]
        vals = [e["precision_score"]["overall"]
                for e in s.get("processed", []) if "precision_score" in e]
        return sum(vals) / len(vals) if vals else None

    def get_overall_steadiness(s):
        for e in s.get("processed", []):
            if "overall_steadiness" in e: return e["overall_steadiness"]
        vals = list(s.get("steadiness", {}).values())
        return (sum(vals) / len(vals)) / 100.0 if vals else None

    # ── All-time bests ────────────────────────────────────────────────────────
    best_acc = {}; best_prec = {}; best_stead = {}; best_flen = {}
    for session in sessions:
        for r, s in get_accuracy_scores(session).items():
            if r not in best_acc or s > best_acc[r]: best_acc[r] = s
        for r, s in get_precision_scores(session).items():
            if r not in best_prec or s > best_prec[r]: best_prec[r] = s
        for f, s in get_steadiness_scores(session).items():
            if f not in best_stead or s > best_stead[f]: best_stead[f] = s
        for finger, stats in session.get("finger_summary", {}).items():
            fn = _n(finger)
            if fn not in best_flen or stats["max"] > best_flen[fn]:
                best_flen[fn] = round(stats["max"], 1)

    best_a_overall = max((get_overall_accuracy(s) for s in sessions
                          if get_overall_accuracy(s) is not None), default=None)
    best_p_overall = max((get_overall_precision(s) for s in sessions
                          if get_overall_precision(s) is not None), default=None)
    best_s_overall = max((get_overall_steadiness(s) for s in sessions
                          if get_overall_steadiness(s) is not None), default=None)

    # Which session number holds the best for a given per-finger metric?
    def _best_ses(getter_fn, key, best_val):
        for i, s in enumerate(sessions):
            v = getter_fn(s).get(key)
            if v is not None and abs(v - best_val) < 1e-9:
                return i + 1
        return "?"

    def _best_ses_overall(getter_fn, best_val):
        for i, s in enumerate(sessions):
            v = getter_fn(s)
            if v is not None and abs(v - best_val) < 1e-9:
                return i + 1
        return "?"

    def _best_ses_flen(fn, best_val):
        for i, s in enumerate(sessions):
            for finger, stats in s.get("finger_summary", {}).items():
                if _n(finger) == fn and abs(round(stats["max"], 1) - best_val) < 1e-9:
                    return i + 1
        return "?"

    # ── New records this session ──────────────────────────────────────────────
    n_sessions = len(sessions)
    latest   = sessions[-1]
    previous = sessions[-2] if n_sessions >= 2 else None

    lat_a = get_accuracy_scores(latest)
    lat_p = get_precision_scores(latest)
    lat_s = get_steadiness_scores(latest)
    prv_a = get_accuracy_scores(previous) if previous else {}
    prv_p = get_precision_scores(previous) if previous else {}
    prv_s = get_steadiness_scores(previous) if previous else {}
    lat_flen = {_n(f): round(st["max"], 1) for f, st in latest.get("finger_summary", {}).items()}
    prv_flen = {_n(f): round(st["max"], 1) for f, st in
                (previous.get("finger_summary", {}) if previous else {}).items()}

    # new_records: list of (label, value_str, diff_str)
    new_records = []

    def _add_rec(label, latest_val, prev_val, best_val,
                 fmt_val=lambda v: f"{v*100:.1f}%",
                 diff_scale=100, diff_unit="%"):
        if latest_val is None or best_val is None:
            return
        if abs(latest_val - best_val) > 1e-9:
            return
        val_str = fmt_val(latest_val)
        if prev_val is not None and abs(latest_val - prev_val) > 1e-9:
            diff = (latest_val - prev_val) * diff_scale
            arrow = "▲" if diff > 0 else "▼"
            diff_str = f"{arrow} {abs(diff):.1f}{diff_unit} bedre end sidst"
        elif prev_val is None:
            diff_str = "første session – rekord sat!"
        else:
            diff_str = "matcher tidligere rekord"
        new_records.append((label, val_str, diff_str))

    for region in FINGER_ORDER:
        lbl = FINGER_LABEL[region]
        _add_rec(f"Nøjagtighed – {lbl}",
                 lat_a.get(region), prv_a.get(region), best_acc.get(region))
        _add_rec(f"Præcision – {lbl}",
                 lat_p.get(region), prv_p.get(region), best_prec.get(region))
        _add_rec(f"Stabilitet – {lbl}",
                 lat_s.get(region), prv_s.get(region), best_stead.get(region))
        _add_rec(f"Stræklængde – {lbl}",
                 lat_flen.get(region), prv_flen.get(region), best_flen.get(region),
                 fmt_val=lambda v: f"{v:.2f} cm",
                 diff_scale=1, diff_unit=" cm")

    _add_rec("Overordnet nøjagtighed",
             get_overall_accuracy(latest),
             get_overall_accuracy(previous) if previous else None, best_a_overall)
    _add_rec("Overordnet præcision",
             get_overall_precision(latest),
             get_overall_precision(previous) if previous else None, best_p_overall)
    _add_rec("Overordnet stabilitet",
             get_overall_steadiness(latest),
             get_overall_steadiness(previous) if previous else None, best_s_overall)

    # ── Theme ─────────────────────────────────────────────────────────────────
    BG       = "#12121e"
    PANEL_BG = "#0d0d1a"
    GRID_COL = "#2a2a45"
    TEXT_COL = "#ccccdd"
    GOLD     = "#FFD700"
    DIM      = "#8888aa"
    GREEN    = "#34d399"

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL_BG,
        "text.color":        TEXT_COL,
        "axes.labelcolor":   TEXT_COL,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "axes.edgecolor":    GRID_COL,
        "grid.color":        GRID_COL,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "legend.facecolor":  BG,
        "legend.edgecolor":  GRID_COL,
        "legend.labelcolor": TEXT_COL,
    })

    fig = plt.figure(figsize=(15, 9))
    try:
        fig.canvas.manager.set_window_title("Session Feedback")
    except Exception:
        pass
    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.showMaximized()          # Qt backends (PyQt5 / PySide2)
    except Exception:
        try:
            mgr.window.state("zoomed")      # Tk backend
        except Exception:
            pass

    suptitle_obj = fig.suptitle(
        f"Hånd Rehabilitering  ·  Session {n_sessions}  ·  Slide 1 / 3",
        fontsize=13, fontweight="bold", color="white", y=0.97,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 1  –  top row: 3 metric graphs  |  bottom row: 3 cumulative-average graphs
    # SLIDE 2  –  4 finger-length graphs
    # ═══════════════════════════════════════════════════════════════════════════
    gs_top  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                                left=0.06, right=0.97, top=0.90, bottom=0.57)
    gs_avg  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                                left=0.06, right=0.97, top=0.44, bottom=0.14)
    gs_flen     = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                                    left=0.05, right=0.98, top=0.88, bottom=0.57)
    gs_flen_avg = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                                    left=0.05, right=0.98, top=0.44, bottom=0.14)

    ax_acc   = fig.add_subplot(gs_top[0, 0])
    ax_prec  = fig.add_subplot(gs_top[0, 1])
    ax_stead = fig.add_subplot(gs_top[0, 2])
    ax_avg_acc   = fig.add_subplot(gs_avg[0, 0])
    ax_avg_prec  = fig.add_subplot(gs_avg[0, 1])
    ax_avg_stead = fig.add_subplot(gs_avg[0, 2])
    ax_flen     = [fig.add_subplot(gs_flen[0, i])     for i in range(4)]
    ax_flen_avg = [fig.add_subplot(gs_flen_avg[0, i]) for i in range(4)]

    slide1_axes      = [ax_acc, ax_prec, ax_stead,
                        ax_avg_acc, ax_avg_prec, ax_avg_stead]
    slide2_flen_axes = list(ax_flen) + list(ax_flen_avg)

    x_ticks = (list(range(1, n_sessions + 1)) if n_sessions <= 20
               else list(range(1, n_sessions + 1, max(1, n_sessions // 10))))

    # Danish descriptions shown below each chart
    DESCRIPTIONS = {
        "acc":   "Måler hvor tæt på midten af brikken du rammer.\nHøj score = du rammer præcist midt på.",
        "prec":  "Måler hvor ensartet du rammer det samme sted igen og igen.\nHøj score = konsekvent og stabil placering.",
        "stead": "Måler hvor roligt din finger bevæger sig hen til brikken.\nOver 90% = meget stabil  ·  Under 60% = mulige milde rystelser i fingeren.",
        "Pege":  "Maks. stræklængde for din pegefinger i cm.\nStigende kurve betyder øget bevægelighed.",
        "Lange": "Maks. stræklængde for din langefinger i cm.\nStigende kurve betyder øget bevægelighed.",
        "Ringe": "Maks. stræklængde for din ringfinger i cm.\nStigende kurve betyder øget bevægelighed.",
        "Lille": "Maks. stræklængde for din lillefinger i cm.\nStigende kurve betyder øget bevægelighed.",
    }

    def _style(ax, title, ylabel="Score (%)", desc_key=None, desc_y=-0.30):
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=6)
        ax.set_xlabel("Session", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x_ticks)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if desc_key and desc_key in DESCRIPTIONS:
            ax.text(0.5, desc_y, DESCRIPTIONS[desc_key],
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color=DIM, style="italic",
                    multialignment="center", clip_on=False)

    _style(ax_acc,   "Nøjagtighed", desc_key="acc",   desc_y=-0.14)
    _style(ax_prec,  "Præcision",   desc_key="prec",  desc_y=-0.14)
    _style(ax_stead, "Stabilitet",  desc_key="stead", desc_y=-0.14)

    _style(ax_avg_acc,   "Gns. Nøjagtighed")
    _style(ax_avg_prec,  "Gns. Præcision")
    _style(ax_avg_stead, "Gns. Stabilitet")

    for i, finger in enumerate(FINGER_ORDER):
        _style(ax_flen[i], f"{FINGER_LABEL[finger]} – Stræk",
               ylabel="Længde (cm)", desc_key=finger, desc_y=-0.14)
        _style(ax_flen_avg[i], f"Gns. {FINGER_LABEL[finger]}",
               ylabel="Længde (cm)")

    # Plot helper for multi-finger metric graphs
    def _plot_multi(ax, series_dict, overall_series):
        for finger in FINGER_ORDER:
            pts = series_dict.get(finger, [])
            if not pts: continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", color=COLORS[finger], linewidth=2,
                    markersize=5, label=finger, zorder=3)
            ax.scatter([xs[-1]], [ys[-1]], s=70, color=COLORS[finger],
                       edgecolors="white", linewidths=0.8, zorder=5)
        if overall_series:
            xs = [p[0] for p in overall_series]; ys = [p[1] for p in overall_series]
            ax.plot(xs, ys, marker="s", color=COLORS["Samlet"], linewidth=2.2,
                    markersize=5, linestyle="--", label="Samlet", zorder=3, alpha=0.9)

    _plot_multi(ax_acc,   acc_series,   overall_acc_s)
    _plot_multi(ax_prec,  prec_series,  overall_prec_s)
    _plot_multi(ax_stead, stead_series, overall_stead_s)

    # Shared legend above the top-row graphs (between suptitle and axes)
    handles, labels = ax_acc.get_legend_handles_labels()
    shared_legend = None
    if handles:
        shared_legend = fig.legend(handles, labels,
                                   loc="upper center", bbox_to_anchor=(0.5, 0.945),
                                   ncol=len(handles), fontsize=8,
                                   framealpha=0.25, edgecolor=GRID_COL,
                                   facecolor=PANEL_BG, labelcolor=TEXT_COL)

    # Dynamic y-axis: tight around actual data with a small breathing margin
    def _tight_ylim(ax, per_finger_series, overall_series):
        all_vals = [y for pts in per_finger_series.values() for _, y in pts]
        all_vals += [y for _, y in overall_series]
        if not all_vals:
            ax.set_ylim(0, 100)
            return
        lo, hi   = min(all_vals), max(all_vals)
        span     = hi - lo if hi > lo else 5.0
        pad      = max(2.5, span * 0.18)
        ax.set_ylim(max(0, lo - pad), min(105, hi + pad))

    _tight_ylim(ax_acc,   acc_series,   overall_acc_s)
    _tight_ylim(ax_prec,  prec_series,  overall_prec_s)
    _tight_ylim(ax_stead, stead_series, overall_stead_s)

    # ── Cumulative average graphs (bottom row of slide 1) ─────────────────────
    def _cum_avg(pts):
        """[(sx, val), ...] → [(sx, running_mean), ...] sorted by sx."""
        if not pts:
            return []
        sorted_pts = sorted(pts, key=lambda p: p[0])
        result, total = [], 0.0
        for i, (sx, v) in enumerate(sorted_pts):
            total += v
            result.append((sx, total / (i + 1)))
        return result

    def _plot_cum_avg(ax, series_dict, overall_series):
        for finger in FINGER_ORDER:
            pts = series_dict.get(finger, [])
            if not pts:
                continue
            ca = _cum_avg(pts)
            xs = [p[0] for p in ca]; ys = [p[1] for p in ca]
            ax.plot(xs, ys, marker="o", color=COLORS[finger], linewidth=2,
                    markersize=4, label=finger, zorder=3)
            ax.scatter([xs[-1]], [ys[-1]], s=55, color=COLORS[finger],
                       edgecolors="white", linewidths=0.8, zorder=5)
        if overall_series:
            ca = _cum_avg(overall_series)
            xs = [p[0] for p in ca]; ys = [p[1] for p in ca]
            ax.plot(xs, ys, marker="s", color=COLORS["Samlet"], linewidth=2.2,
                    markersize=4, linestyle="--", label="Samlet", zorder=3, alpha=0.9)

    _plot_cum_avg(ax_avg_acc,   acc_series,   overall_acc_s)
    _plot_cum_avg(ax_avg_prec,  prec_series,  overall_prec_s)
    _plot_cum_avg(ax_avg_stead, stead_series, overall_stead_s)

    # Reuse tight ylim for avg graphs (data is same values, just averaged)
    _tight_ylim(ax_avg_acc,   acc_series,   overall_acc_s)
    _tight_ylim(ax_avg_prec,  prec_series,  overall_prec_s)
    _tight_ylim(ax_avg_stead, stead_series, overall_stead_s)

    # Plot each finger-length graph (single line + optional goal line)
    for i, finger in enumerate(FINGER_ORDER):
        ax = ax_flen[i]
        pts = flen_series.get(finger, [])
        goal = goal_finger_lengths.get(finger)
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", color=COLORS[finger], linewidth=2,
                    markersize=5, zorder=3)
            ax.scatter([xs[-1]], [ys[-1]], s=70, color=COLORS[finger],
                       edgecolors="white", linewidths=0.8, zorder=5)
            # y-axis must show both data and goal, with margin above the higher one
            all_vals = ys + ([goal] if goal is not None else [])
            y_min = max(0, min(all_vals) - 0.2)
            spread = max(all_vals) - min(all_vals)
            y_top  = max(all_vals) + max(0.15, spread * 0.20)
            ax.set_ylim(y_min, y_top)
        # Rehabilitation goal as a dotted white line
        if goal is not None:
            ax.axhline(goal, color="white", linewidth=1.2,
                       linestyle=":", alpha=0.55, label=f"Mål {goal:.1f} cm", zorder=2)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    # Cumulative average finger-length graphs (bottom row of slide 2)
    for i, finger in enumerate(FINGER_ORDER):
        ax  = ax_flen_avg[i]
        pts = flen_series.get(finger, [])
        goal = goal_finger_lengths.get(finger)
        if pts:
            ca = _cum_avg(pts)
            xs = [p[0] for p in ca]; ys = [p[1] for p in ca]
            ax.plot(xs, ys, marker="o", color=COLORS[finger], linewidth=2,
                    markersize=4, zorder=3)
            ax.scatter([xs[-1]], [ys[-1]], s=55, color=COLORS[finger],
                       edgecolors="white", linewidths=0.8, zorder=5)
            all_vals = ys + ([goal] if goal is not None else [])
            y_min = max(0, min(all_vals) - 0.2)
            spread = max(all_vals) - min(all_vals)
            y_top  = max(all_vals) + max(0.15, spread * 0.20)
            ax.set_ylim(y_min, y_top)
        if goal is not None:
            ax.axhline(goal, color="white", linewidth=1.2,
                       linestyle=":", alpha=0.55, label=f"Mål {goal:.1f} cm", zorder=2)
            ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 2  –  celebration + all-time records + session history
    # ═══════════════════════════════════════════════════════════════════════════
    ax_slide2 = fig.add_axes([0.0, 0.08, 1.0, 0.88])
    ax_slide2.set_facecolor(BG)
    ax_slide2.axis("off")

    def _render_slide2():
        ax_slide2.cla()
        ax_slide2.set_facecolor(BG)
        ax_slide2.axis("off")
        ax_slide2.set_xlim(0, 1)
        ax_slide2.set_ylim(0, 1)

        # ── Adaptive line heights ─────────────────────────────────────────────
        # The RIGHT column (all-time records) always uses base sizes.
        # The LEFT column (new records) scales down if its content overflows.
        available_height = 0.97 - 0.02   # usable y range
        SH_BASE = 0.034
        LH_BASE = 0.055
        LH_RATIO = LH_BASE / SH_BASE     # ≈ 1.618 – keeps lh/sh proportional

        goals_to_show_tmp = [(f, goal_finger_lengths[f], best_flen.get(f))
                             for f in FINGER_ORDER if f in goal_finger_lengths]

        # Left column height estimate (sh units)
        if new_records:
            lc_sh = (LH_RATIO * 0.9 + LH_RATIO * 1.1            # GODT GÅET + subtitle
                     + len(new_records) * (1.2 + 1.5)             # label + value per record
                     + 0.5 + 1.4 + 1.4)                           # encouragement lines
        else:
            lc_sh = LH_RATIO + LH_RATIO * 1.1 + 1.4 + 1.4        # no-record variant

        if goals_to_show_tmp:
            lc_sh += 1.2 + 0.7 + LH_RATIO * 0.85 + len(goals_to_show_tmp) * 1.5

        lc_scale = min(1.0, available_height / (SH_BASE * lc_sh))
        sh  = SH_BASE * lc_scale          # left-column line height (may be reduced)
        lh  = sh * LH_RATIO               # left-column heading height (proportional)

        # Right column always uses base sizes
        sh_r = SH_BASE
        lh_r = LH_BASE

        def txt(x, yy, s, color=TEXT_COL, size=9, bold=False, mono=False):
            """Left-column text — font size scales with lc_scale."""
            ax_slide2.text(x, yy, s, transform=ax_slide2.transAxes,
                           fontsize=max(6, round(size * lc_scale)), color=color, va="top",
                           fontweight="bold" if bold else "normal",
                           fontfamily="monospace" if mono else "sans-serif")

        def txt_r(x, yy, s, color=TEXT_COL, size=9, bold=False, mono=False):
            """Right-column text — always base font size."""
            ax_slide2.text(x, yy, s, transform=ax_slide2.transAxes,
                           fontsize=size, color=color, va="top",
                           fontweight="bold" if bold else "normal",
                           fontfamily="monospace" if mono else "sans-serif")

        def vline(alpha=0.25):
            ax_slide2.plot([0.50, 0.50], [0.02, 0.97], color=GRID_COL,
                           linewidth=0.8, alpha=alpha,
                           transform=ax_slide2.transAxes, clip_on=False)

        vline()

        # ── LEFT COLUMN: Celebration / new records ────────────────────────────
        LX  = 0.02   # left column x
        TOP = 0.97

        ly = TOP
        if new_records:
            txt(LX, ly, "GODT GÅET!", color=GOLD, size=20, bold=True)
            ly -= lh * 0.9
            txt(LX, ly, "Du har slået rekord i:", color=GOLD, size=13)
            ly -= lh * 1.1

            for label, val_str, diff_str in new_records:
                txt(LX + 0.01, ly, f"★  {label}",
                    color=GOLD, size=12, bold=True, mono=True)
                ly -= sh * 1.2
                txt(LX + 0.03, ly, f"{val_str}   ({diff_str})",
                    color=TEXT_COL, size=10, mono=True)
                ly -= sh * 1.5

            ly -= sh * 0.5
            txt(LX, ly,
                "Bliv ved med det gode arbejde –",
                color=GREEN, size=13, bold=True)
            ly -= sh * 1.4
            txt(LX, ly,
                "du bliver bedre og bedre!",
                color=GREEN, size=13, bold=True)
        else:
            txt(LX, ly,
                "Godt gået!",
                color=TEXT_COL, size=18, bold=True)
            ly -= lh
            txt(LX, ly,
                "Du er mødt op og har trænet – det tæller!",
                color=TEXT_COL, size=13)
            ly -= lh * 1.1
            txt(LX, ly,
                "Ingen nye rekorder denne gang,",
                color=GREEN, size=13)
            ly -= sh * 1.4
            txt(LX, ly,
                "men bliv ved – fremgangen kommer!",
                color=GREEN, size=13)

        # ── LEFT COLUMN continued: Goals ─────────────────────────────────────
        goals_to_show = [(f, goal_finger_lengths[f], best_flen.get(f))
                         for f in FINGER_ORDER if f in goal_finger_lengths]
        if goals_to_show:
            ly -= sh * 1.2
            ax_slide2.plot([LX, 0.46], [ly, ly], color=GRID_COL,
                           linewidth=0.6, alpha=0.4,
                           transform=ax_slide2.transAxes, clip_on=False)
            ly -= sh * 0.7
            txt(LX, ly, "Mål – Stræklængde", color="white", size=13, bold=True)
            ly -= lh * 0.85

            for finger, goal, current in goals_to_show:
                lbl = FINGER_LABEL[finger]
                col = COLORS.get(finger, TEXT_COL)
                if current is None:
                    txt(LX, ly, f"{lbl}:  ingen data endnu",
                        color=DIM, size=10, mono=True)
                elif current >= goal:
                    txt(LX, ly,
                        f"★  {lbl}:  {current:.1f} cm  –  Mål nået! 🎉",
                        color=GREEN, size=10, bold=True, mono=True)
                else:
                    remaining = goal - current
                    bar_filled = int((current / goal) * 10)
                    bar = "█" * bar_filled + "░" * (10 - bar_filled)
                    txt(LX, ly,
                        f"{lbl}:  {current:.1f} / {goal:.1f} cm  [{bar}]  {remaining:.1f} cm tilbage",
                        color=col, size=10, mono=True)
                ly -= sh * 1.5

        # ── RIGHT COLUMN: All-time records ────────────────────────────────────
        RX  = 0.53   # right column x
        ry  = TOP

        txt_r(RX, ry, "Alle tiders rekorder", color="white", size=14, bold=True)
        ry -= lh_r * 0.85

        sections = [
            ("Nøjagtighed", best_acc,   get_accuracy_scores,
             lambda v: f"{v*100:.1f}%"),
            ("Præcision",   best_prec,  get_precision_scores,
             lambda v: f"{v*100:.1f}%"),
            ("Stabilitet",  best_stead, get_steadiness_scores,
             lambda v: f"{v*100:.1f}%"),
            ("Stræklængde", best_flen,  None,
             lambda v: f"{v:.1f} cm"),
        ]

        for sec_label, best_dict, getter_fn, fmt_fn in sections:
            txt_r(RX, ry, sec_label + ":", color=DIM, size=12, bold=True)
            ry -= sh_r * 1.3
            for finger in FINGER_ORDER:
                bv = best_dict.get(finger)
                if bv is None:
                    continue
                if getter_fn is not None:
                    si = _best_ses(getter_fn, finger, bv)
                else:
                    si = _best_ses_flen(finger, bv)
                lbl = FINGER_LABEL[finger]
                txt_r(RX + 0.02, ry,
                    f"{lbl}:  {fmt_fn(bv)}  (ses. {si})",
                    color=COLORS[finger], size=11, mono=True)
                ry -= sh_r * 1.3
            ry -= sh_r * 0.4


    _render_slide2()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 3  –  suggestive changes (read-only view of sugestive_changes logic)
    # ═══════════════════════════════════════════════════════════════════════════
    ax_slide3 = fig.add_axes([0.0, 0.08, 1.0, 0.88])
    ax_slide3.set_facecolor(BG)
    ax_slide3.axis("off")

    def _render_slide3():
        ax_slide3.cla()
        ax_slide3.set_facecolor(BG)
        ax_slide3.axis("off")
        ax_slide3.set_xlim(0, 1)
        ax_slide3.set_ylim(0, 1)

        y  = 0.97
        lh = 0.042
        sh = 0.026

        _NM = {"Thumb": "Tommel", "Index": "Pege", "Middle": "Lange",
               "Ring":  "Ringe",  "Pinky": "Lille"}

        def txt(x, yy, s, color=TEXT_COL, size=9, bold=False, mono=False):
            ax_slide3.text(x, yy, s, transform=ax_slide3.transAxes,
                           fontsize=size, color=color, va="top",
                           fontweight="bold" if bold else "normal",
                           fontfamily="monospace" if mono else "sans-serif")

        def hline(yy, alpha=0.30):
            ax_slide3.plot([0.02, 0.98], [yy, yy], color=GRID_COL, linewidth=0.8,
                           alpha=alpha, transform=ax_slide3.transAxes, clip_on=False)

        # Re-load data fresh (sugestive_changes may not have run yet)
        with open(history_file, "r") as _f:
            _raw = json.load(_f)
        _raw = [_raw] if isinstance(_raw, dict) else _raw
        all_s3 = [s for s in _raw if "processed" in s]
        n_total = len(all_s3)
        sessions_needed = 3

        txt(0.02, y, "Forslag til justeringer af brikplacering",
            color="white", size=12, bold=True)
        y -= lh * 0.7
        hline(y)
        y -= sh * 0.6

        if n_total < sessions_needed:
            remaining = sessions_needed - n_total
            txt(0.02, y,
                f"Du mangler {remaining} session(er) mere, før forslag kan genereres.",
                color=DIM, size=10)
            y -= sh * 1.6
            txt(0.02, y, f"({n_total} / {sessions_needed} sessioner registreret)",
                color=DIM, size=9)
            return

        raw_brick_state = _raw[0].get("brick_state", {})
        brick_state_s3 = {_NM.get(k, k): v for k, v in raw_brick_state.items()}

        sessions_used = all_s3[-3:]
        n_sess_used   = len(sessions_used)

        finger_data_s3 = defaultdict(lambda: defaultdict(list))
        for _sess in sessions_used:
            for item in _sess.get("processed", []):
                if "region" not in item or "region_percentages" not in item:
                    continue
                fn = _NM.get(item["region"], item["region"])
                for sub, pct in item["region_percentages"].items():
                    finger_data_s3[fn][sub].append(pct)

        averaged_s3 = {
            fn: {sub: sum(vals) / len(vals) for sub, vals in regions.items()}
            for fn, regions in finger_data_s3.items()
        }

        if not averaged_s3:
            txt(0.02, y, "Ingen region-data – kan ikke generere forslag.", color=DIM, size=10)
            return

        txt(0.02, y,
            f"Baseret på de seneste {n_sess_used} sessioner  "
            ,
            color=DIM, size=8)
        y -= lh * 0.85

        finger_order_s3   = ["Pege", "Lange", "Ringe", "Lille"]
        finger_color_name = {"Pege": "gul", "Lange": "lilla",
                              "Ringe": "grøn", "Lille": "rød"}
        fingers_to_show = [f for f in finger_order_s3 if f in averaged_s3]

        # Fingers as columns across the top
        n_cols   = len(fingers_to_show)
        col_w    = 0.96 / max(n_cols, 1)
        ROW_LBL  = 0.02                                      # row-label x
        col_xs   = [0.18 + i * col_w for i in range(n_cols)] # data column x positions

        # ── Column headers (finger names) ─────────────────────────────────────
        txt(ROW_LBL, y, "", color=DIM, size=9)   # blank corner
        for i, finger in enumerate(fingers_to_show):
            farve = finger_color_name.get(finger, "")
            lbl   = f"{finger}\n({farve})" if farve else finger
            txt(col_xs[i], y, f"{finger} ({farve})",
                color=COLORS.get(finger, TEXT_COL), size=10, bold=True)
        y -= sh * 1.6
        hline(y)
        y -= sh * 0.8

        # Pre-compute per-finger data
        finger_info = {}
        for finger in fingers_to_show:
            bs = brick_state_s3.get(finger, {
                "last_change_session": 0, "is_shrunk": False,
                "shrink_check_pending": False, "grow_pending": False,
            })
            last_change    = bs.get("last_change_session", 0)
            is_shrunk      = bs.get("is_shrunk", False)
            shrink_pending = bs.get("shrink_check_pending", False)
            grow_pending   = bs.get("grow_pending", False)
            sessions_since = n_total - last_change
            on_cooldown    = last_change > 0 and sessions_since <= SUGGEST_COOLDOWN
            remaining_cd   = max(0, SUGGEST_COOLDOWN - sessions_since)

            pcts = averaged_s3.get(finger, {})
            TR = pcts.get("top_right",    0.0)
            TL = pcts.get("top_left",     0.0)
            BR = pcts.get("bottom_right", 0.0)
            BL = pcts.get("bottom_left",  0.0)
            C  = pcts.get("center",       0.0)

            finger_info[finger] = dict(
                on_cooldown=on_cooldown, remaining_cd=remaining_cd,
                is_shrunk=is_shrunk, shrink_pending=shrink_pending,
                grow_pending=grow_pending,
                C=C, TR=TR, TL=TL, BR=BR, BL=BL,
            )

        # ── Brick heatmap per finger ──────────────────────────────────────────
        # Each heatmap is a 4-wide × 6-tall grid representing the Lego brick.
        # Sub-region percentages are mapped onto the brick:
        #   rows 0-1 (top)    → TL (cols 0-1) and TR (cols 2-3)
        #   rows 2-3 (middle) → center (all cols)
        #   rows 4-5 (bottom) → BL (cols 0-1) and BR (cols 2-3)
        txt(ROW_LBL, y, "Berørings­varmekort", color=DIM, size=9, bold=True)
        y -= sh * 0.5

        heatmap_h = 0.22   # height in axes coords
        heatmap_w = 0.10   # width  in axes coords (4:6 aspect → ~0.67 ratio)

        # Remove stale heatmap axes from previous renders
        for child in list(fig.axes):
            if getattr(child, "_is_brick_heatmap", False):
                child.remove()

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "brick_hit", ["#0d1117", "#1a2a4a", "#d95f02", "#facc15"])

        for i, finger in enumerate(fingers_to_show):
            d = finger_info[finger]
            cx = col_xs[i]

            x0 = cx - heatmap_w / 2
            y0 = y - heatmap_h
            inset = ax_slide3.inset_axes([x0, y0, heatmap_w, heatmap_h])
            inset._is_brick_heatmap = True

            if True:
                TL, TR = d["TL"], d["TR"]
                BL, BR = d["BL"], d["BR"]
                C      = d["C"]

                # Build 6×4 grid (rows=6, cols=4)
                # Center is a 2×2 square in the middle; quadrants fill corners
                grid = np.array([
                    [TL, TL, TR, TR],   # row 0 – top
                    [TL, TL, TR, TR],   # row 1
                    [TL, C,  C,  TR],   # row 2 – center square
                    [BL, C,  C,  BR],   # row 3
                    [BL, BL, BR, BR],   # row 4 – bottom
                    [BL, BL, BR, BR],   # row 5
                ], dtype=float)

                inset.imshow(grid, cmap=cmap, vmin=0, vmax=0.55,
                             aspect="auto", interpolation="bilinear")

                # Overlay region labels at quadrant centres
                label_coords = [
                    (0.5, 0.5/6,  f"{100*BL:.0f}%"),   # BL centre  (bottom in display = row 5)
                    (2.5, 0.5/6,  f"{100*BR:.0f}%"),   # BR
                    (1.5, 2.5/6,  f"{100*C:.0f}%"),    # C
                    (0.5, 4.5/6,  f"{100*TL:.0f}%"),   # TL
                    (2.5, 4.5/6,  f"{100*TR:.0f}%"),   # TR
                ]
                # imshow puts row 0 at top, so remap y: data_row → display_row = 5 - data_row
                # We annotate in data coords directly (imshow axes are col, row)
                region_label_data = [
                    (0.5, 4.5, f"BL\n{100*BL:.0f}%"),
                    (2.5, 4.5, f"BR\n{100*BR:.0f}%"),
                    (1.5, 2.5, f"C\n{100*C:.0f}%"),    # centre of the 2×2 block
                    (0.5, 0.5, f"TL\n{100*TL:.0f}%"),
                    (2.5, 0.5, f"TR\n{100*TR:.0f}%"),
                ]
                for (xc, yc, lbl) in region_label_data:
                    val = grid[int(round(yc)), int(round(xc))]
                    tc = "white" if val < 0.28 else "black"
                    inset.text(xc, yc, lbl, ha="center", va="center",
                               color=tc, fontsize=5.5, fontweight="bold")

                inset.set_xlim(-0.5, 3.5)
                inset.set_ylim(5.5, -0.5)
                inset.set_xticks([])
                inset.set_yticks([])


            for spine in inset.spines.values():
                spine.set_edgecolor(COLORS.get(finger, "#555"))
                spine.set_linewidth(1.2)

        y -= heatmap_h + sh * 0.6
        hline(y)
        y -= sh * 0.8

        # ── Forslag row ───────────────────────────────────────────────────────
        txt(ROW_LBL, y, "Forslag", color=DIM, size=9, bold=True)
        y -= sh * 1.3
        for i, finger in enumerate(fingers_to_show):
            d = finger_info[finger]
            cx = col_xs[i]
            col_color = COLORS.get(finger, TEXT_COL)
            if d["on_cooldown"]:
                size_note = " [2×4]" if d["is_shrunk"] else ""
                txt(cx, y, f"Hviler", color=DIM, size=8, mono=True)
                y2 = y - sh * 1.3
                txt(cx, y2, f"{d['remaining_cd']} ses.{size_note}",
                    color=DIM, size=8, mono=True)
            elif d["grow_pending"]:
                txt(cx, y, "VOKS", color=GREEN, size=9, bold=True, mono=True)
                txt(cx, y - sh*1.3, "→ 4×6", color=GREEN, size=8, mono=True)
            elif d["shrink_pending"]:
                if d["C"] < SUGGEST_E:
                    txt(cx, y, "SKRUMP", color="#f97316", size=9, bold=True, mono=True)
                    txt(cx, y - sh*1.3, "→ 2×4", color="#f97316", size=8, mono=True)
                else:
                    txt(cx, y, "Størrelse", color=GREEN, size=9, mono=True)
                    txt(cx, y - sh*1.3, "OK", color=GREEN, size=8, mono=True)
            elif d["C"] >= SUGGEST_E:
                txt(cx, y, "Ingen", color=GREEN, size=9, mono=True)
                txt(cx, y - sh*1.3, "flytning", color=GREEN, size=8, mono=True)
            else:
                moves = []
                if d["BR"]+d["BL"] > SUGGEST_A: moves.append("NED ↓")
                if d["TR"]+d["TL"] > SUGGEST_B: moves.append("OP ↑")
                if d["TR"]+d["BR"] > SUGGEST_C: moves.append("HØJRE →")
                if d["TL"]+d["BL"] > SUGGEST_D: moves.append("VENSTRE ←")
                if moves:
                    for mi, mv in enumerate(moves):
                        txt(cx, y - mi*sh*1.3, mv, color=GOLD, size=9, bold=True, mono=True)
                else:
                    txt(cx, y, "Ingen", color=GREEN, size=9, mono=True)
                    txt(cx, y - sh*1.3, "flytning", color=GREEN, size=8, mono=True)

        y -= sh * 3.2

        # ── Placeholder: feedback section ─────────────────────────────────────
        hline(y)
        y -= sh * 0.8

        txt(0.02, y, "Feedback", color="white", size=11, bold=True)
        y -= lh * 0.8

        any_feedback = False
        for finger in ["Pege", "Lange", "Ringe", "Lille"]:
            pts  = flen_series.get(finger, [])
            goal = goal_finger_lengths.get(finger)
            best = best_flen.get(finger)
            lbl  = FINGER_LABEL.get(finger, finger).lower()
            col  = COLORS.get(finger, TEXT_COL)

            if not pts or goal is None or best is None:
                continue

            sorted_pts  = sorted(pts, key=lambda p: p[0])
            latest_val  = sorted_pts[-1][1]
            prev_val    = sorted_pts[-2][1] if len(sorted_pts) >= 2 else None
            improving   = prev_val is not None and latest_val > prev_val
            remaining   = goal - best

            if remaining <= 0:
                txt(0.02, y,
                    f"★  Flot! Du har nået målet for din {lbl}!",
                    color=GREEN, size=10, bold=True)
                y -= sh * 1.5
                any_feedback = True
            elif improving:
                txt(0.02, y,
                    f"Sådan! Du rykker dig frem i at strække din {lbl} –",
                    color=col, size=10, bold=True)
                y -= sh * 1.4
                txt(0.04, y,
                    f"du er kun {remaining:.1f} cm fra at kunne strække den fuldt ud!",
                    color=TEXT_COL, size=10)
                y -= sh * 1.8
                any_feedback = True

        if not any_feedback:
            txt(0.02, y,
                "Bliv ved med at træne – fremgangen vil vise sig!",
                color=DIM, size=10)

    _render_slide3()

    # ═══════════════════════════════════════════════════════════════════════════
    # Navigation buttons  (prev / next, labels update per slide)
    # ═══════════════════════════════════════════════════════════════════════════
    ax_btn_prev = fig.add_axes([0.36, 0.02, 0.13, 0.045])
    ax_btn_next = fig.add_axes([0.51, 0.02, 0.13, 0.045])

    btn_prev = Button(ax_btn_prev, "◄  Grafer",
                      color="#1e1e32", hovercolor="#2e2e4a")
    btn_next = Button(ax_btn_next, "Stræklængde  ►",
                      color="#1e1e32", hovercolor="#2e2e4a")
    for btn in (btn_prev, btn_next):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)

    state = {"page": 0}

    def _show_page(page):
        state["page"] = page
        for ax in slide1_axes:
            ax.set_visible(page == 0)
        for ax in slide2_flen_axes:
            ax.set_visible(page == 1)
        ax_slide2.set_visible(page == 2)
        ax_slide3.set_visible(page == 3)
        if shared_legend is not None:
            shared_legend.set_visible(page == 0)

        if page == 2:
            _render_slide2()
        elif page == 3:
            _render_slide3()

        suptitle_obj.set_text(
            f"Hånd Rehabilitering  ·  Session {n_sessions}  ·  Slide {page + 1} / 4"
        )

        # Update button labels and visibility
        btn_prev.ax.set_visible(page > 0)
        btn_next.ax.set_visible(page < 3)
        if page == 0:
            btn_next.label.set_text("Stræklængde  ►")
        elif page == 1:
            btn_prev.label.set_text("◄  Grafer")
            btn_next.label.set_text("Rekorder  ►")
        elif page == 2:
            btn_prev.label.set_text("◄  Stræklængde")
            btn_next.label.set_text("Justeringer  ►")
        elif page == 3:
            btn_prev.label.set_text("◄  Rekorder")

        fig.canvas.draw_idle()

    btn_prev.on_clicked(lambda e: _show_page(state["page"] - 1))
    btn_next.on_clicked(lambda e: _show_page(state["page"] + 1))

    _show_page(0)
    plt.show()


def sugestive_changes(history_file="touch_history.json"):
    """Suggest brick position adjustments based on averaged touch regions
    from the last 3 sessions.

    Movement rules (BR+BL+TR+TL+C = 1):
        BR + BL > SUGGEST_A  →  move brick one step DOWN
        TR + TL > SUGGEST_B  →  move brick one step UP
        TR + BR > SUGGEST_C  →  move brick one step to the RIGHT
        TL + BL > SUGGEST_D  →  move brick one step to the LEFT

    Shrink rule (checked when finger is not on cooldown):
        center < SUGGEST_E   →  suggest shrinking brick to 2x4

    Cooldown rule: once any change is suggested for a finger it is locked
    for SUGGEST_COOLDOWN sessions before it can receive another suggestion.

    Per-finger state is stored in touch_history.json under "brick_state":
        last_change_session     – session number when the last change was made
        on_cooldown             – True if still within the cooldown window
        cooldown_sessions_remaining – how many more sessions until unlocked
        is_shrunk               – True once a shrink has been suggested
    """

    _NAME_MAP = {"Thumb": "Tommel", "Index": "Pege", "Middle": "Lange", "Ring": "Ringe", "Pinky": "Lille"}

    # --- Load history (dict on first session ever, list afterwards) ---
    with open(history_file, "r") as f:
        raw = json.load(f)

    raw_was_dict = isinstance(raw, dict)
    data = [raw] if raw_was_dict else raw

    # Keep only entries that are actual play sessions (have 'processed' data)
    all_sessions = [s for s in data if "processed" in s]

    sessions_needed = 3
    n_total = len(all_sessions)

    if n_total == 0:
        print("Ingen sessioner fundet – ingen forslag kan genereres.")
        print(f"Fuldfør {sessions_needed} sessioner for at aktivere forslag.")
        return

    if n_total < sessions_needed:
        remaining = sessions_needed - n_total
        print(f"\n===== FORSLAG TIL JUSTERINGER =====")
        print(f"  {remaining} session(er) mangler, før forslag kan genereres.")
        print(f"  ({n_total}/{sessions_needed} sessioner registreret)")
        print("============================\n")
        return

    # --- Load / migrate brick_state from data[0] ---
    # Supports migration from the old "suggest_cooldown" dict format.
    raw_brick_state = data[0].get("brick_state", None)
    if raw_brick_state is None:
        # Migrate from old suggest_cooldown format if present
        old_cooldown = data[0].get("suggest_cooldown", {})
        raw_brick_state = {
            _NAME_MAP.get(k, k): {
                "last_change_session": v,
                "is_shrunk": False,
            }
            for k, v in old_cooldown.items()
        }

    # Normalise any English keys that may have been stored previously
    brick_state = {_NAME_MAP.get(k, k): v for k, v in raw_brick_state.items()}

    # --- Accumulate region percentages per finger across the last 3 sessions ---
    sessions_used = all_sessions[-3:]
    n_sessions    = len(sessions_used)

    finger_data = defaultdict(lambda: defaultdict(list))
    for session in sessions_used:
        for item in session.get("processed", []):
            if "region" not in item or "region_percentages" not in item:
                continue
            finger = _NAME_MAP.get(item["region"], item["region"])
            for sub_region, pct in item["region_percentages"].items():
                finger_data[finger][sub_region].append(pct)

    if not finger_data:
        print("Ingen region-data i de seneste sessioner – ingen forslag kan genereres.")
        return

    # --- Average across sessions ---
    averaged = {
        finger: {sub: sum(vals) / len(vals) for sub, vals in regions.items()}
        for finger, regions in finger_data.items()
    }

    # --- Print header ---
    print(f"\n===== FORSLAG TIL JUSTERINGER (baseret på de seneste {n_sessions} sessioner) =====\n")

    finger_order   = ["Tommel", "Pege", "Lange", "Ringe", "Lille"]
    finger_colors  = {"Pege": "gul", "Lange": "lilla", "Ringe": "grøn", "Lille": "rød"}
    fingers_to_show = [f for f in finger_order if f in averaged] + \
                      [f for f in averaged  if f not in finger_order]

    any_change_recorded = False

    for finger in fingers_to_show:
        farve = finger_colors.get(finger, "")
        label = f"{finger} ({farve})" if farve else finger

        # Ensure this finger has an entry in brick_state
        if finger not in brick_state:
            brick_state[finger] = {
                "last_change_session": 0,
                "is_shrunk": False,
                "shrink_check_pending": False,
                "grow_pending": False,
            }

        state = brick_state[finger]
        last_change_session   = state.get("last_change_session", 0)
        is_shrunk             = state.get("is_shrunk", False)
        shrink_check_pending  = state.get("shrink_check_pending", False)
        grow_pending          = state.get("grow_pending", False)
        sessions_since_change = n_total - last_change_session

        # --- Cooldown check ---
        on_cooldown = last_change_session > 0 and sessions_since_change <= SUGGEST_COOLDOWN
        remaining   = max(0, SUGGEST_COOLDOWN - sessions_since_change)

        if on_cooldown:
            size_note = "  [brik er skrumpet til 2x4]" if is_shrunk else ""
            print(f"  {label}:")
            print(f"    Hviler – {remaining} session(er) tilbage af pause "
                  f"(justering foretaget ved session {last_change_session}){size_note}")
            print()
            state["on_cooldown"]                 = True
            state["cooldown_sessions_remaining"] = remaining
            continue

        # --- Not on cooldown: evaluate pending size changes first, then direction ---
        pcts = averaged[finger]
        TR = pcts.get("top_right",    0.0)
        TL = pcts.get("top_left",     0.0)
        BR = pcts.get("bottom_right", 0.0)
        BL = pcts.get("bottom_left",  0.0)
        C  = pcts.get("center",       0.0)

        breakdown = (f"C={100*C:.0f}%  TR={100*TR:.0f}%  TL={100*TL:.0f}%  "
                     f"BR={100*BR:.0f}%  BL={100*BL:.0f}%")

        print(f"  {label}:")
        print(f"    {breakdown}")

        # Grow rule — fires on the first session after the shrink cooldown ends.
        # The brick is always grown back to 4x6 regardless of centre percentage.
        if grow_pending:
            print(f"    →  VOKS brik tilbage til 4x6 – cooldown for skrumpet brik er slut")
            state["is_shrunk"]            = False
            state["grow_pending"]         = False
            state["last_change_session"]  = n_total
            state["on_cooldown"]          = True
            state["cooldown_sessions_remaining"] = SUGGEST_COOLDOWN
            any_change_recorded = True
            print()
            continue   # no further rules this session for this finger

        # Shrink rule — only on the first session after a movement cooldown ends.
        if shrink_check_pending:
            if C < SUGGEST_E:
                print(f"    →  SKRUMP brik til 2x4 – spilleren rammer stadig ikke "
                      f"midten (center {100*C:.0f}% < tærskel {100*SUGGEST_E:.0f}%)")
                state["is_shrunk"]            = True
                state["shrink_check_pending"] = False
                state["grow_pending"]         = True   # grow back after next cooldown
                state["last_change_session"]  = n_total
                state["on_cooldown"]          = True
                state["cooldown_sessions_remaining"] = SUGGEST_COOLDOWN
                any_change_recorded = True
            else:
                print(f"    →  Brikstørrelse OK – spilleren rammer midten nu "
                      f"(center {100*C:.0f}% ≥ {100*SUGGEST_E:.0f}%)")
                state["shrink_check_pending"] = False
            print()
            continue   # no direction rules on a shrink-check session

        # Direction rules — only evaluated when centre is below threshold.
        # If center >= SUGGEST_E the brick position is already good enough.
        if C >= SUGGEST_E:
            print(f"    →  Ingen flytning nødvendig – center ramt tilstrækkeligt "
                  f"({100*C:.0f}% ≥ tærskel {100*SUGGEST_E:.0f}%)")
            print()
            state["on_cooldown"]                 = False
            state["cooldown_sessions_remaining"] = 0
            continue

        moves = []
        if BR + BL > SUGGEST_A:
            moves.append("NED  ↓")
        if TR + TL > SUGGEST_B:
            moves.append("OP   ↑")
        if TR + BR > SUGGEST_C:
            moves.append("HØJRE →")
        if TL + BL > SUGGEST_D:
            moves.append("VENSTRE ←")

        if moves:
            print(f"    →  Flyt brik: " + ",  ".join(moves))
        else:
            print(f"    →  Ingen flytning nødvendig (berøring tilpas centreret)")

        print()

        if moves:
            state["last_change_session"]         = n_total
            state["on_cooldown"]                 = True
            state["cooldown_sessions_remaining"] = SUGGEST_COOLDOWN
            state["shrink_check_pending"]        = True
            any_change_recorded = True
        else:
            state["on_cooldown"]                 = False
            state["cooldown_sessions_remaining"] = 0

    print("Tærskelværdier:  a={a}  b={b}  c={c}  d={d}  e={e}  cooldown={cd} sessioner".format(
        a=SUGGEST_A, b=SUGGEST_B, c=SUGGEST_C, d=SUGGEST_D,
        e=SUGGEST_E, cd=SUGGEST_COOLDOWN))
    print("============================\n")

    # --- Always persist brick_state so cooldown_sessions_remaining stays current ---
    data[0]["brick_state"] = brick_state
    # Remove the old key if it was migrated
    data[0].pop("suggest_cooldown", None)
    with open(history_file, "w") as f:
        json.dump(data[0] if raw_was_dict else data, f, indent=4)

def trim_touch_history(touch_history, keep_frames=3):
    from collections import defaultdict

    region_events  = defaultdict(list)
    region_current = defaultdict(list)
    region_last_frame = {}

    for i, entry in enumerate(touch_history):
        frame_num      = entry["frame"]
        active_regions = set(entry["touches"].keys())

        for region in active_regions:
            if region in region_last_frame:
                if frame_num - region_last_frame[region] > 1:
                    region_events[region].append(region_current[region][:])
                    region_current[region] = []
            region_current[region].append(i)
            region_last_frame[region] = frame_num

    for region, run in region_current.items():
        if run:
            region_events[region].append(run)

    # Build a keep set per region independently
    region_keep = defaultdict(set)
    for region, events in region_events.items():
        for run in events:
            if len(run) <= 2:
                region_keep[region].update(run)
            else:
                middle = run[1:-1]
                start  = max(0, (len(middle) - keep_frames) // 2)
                region_keep[region].update(middle[start : start + keep_frames])

    # Rebuild entries keeping only touches selected for that specific region
    trimmed = []
    for i, entry in enumerate(touch_history):
        kept_touches = {
            region: pos
            for region, pos in entry["touches"].items()
            if i in region_keep[region]
        }
        if kept_touches:
            trimmed.append({**entry, "touches": kept_touches})

    return trimmed

def compute_steadiness(touch_history, clickable_regions, k=1.0):

    finger_map = region_to_finger_map(clickable_regions)

    # Build per-region runs of consecutive frames
    region_runs    = defaultdict(list)
    region_current = defaultdict(list)
    region_last_frame = {}

    for entry in touch_history:
        frame_num      = entry["frame"]
        active_regions = set(entry["touches"].keys())

        for region in active_regions:
            if region in region_last_frame:
                if frame_num - region_last_frame[region] > 1:
                    region_runs[region].append(region_current[region][:])
                    region_current[region] = []
            region_current[region].append(entry["touches"][region])
            region_last_frame[region] = frame_num

    for region, run in region_current.items():
        if run:
            region_runs[region].append(run)

    # Compute steadiness score S per run, then average across runs
    steadiness = {}
    for region, runs in region_runs.items():
        label = finger_map.get(region, region)
        scores = []

        for points in runs:
            if len(points) < 2:
                scores.append(100.0)
                continue

            pts   = np.array(points, dtype=float)
            x, y  = pts[:, 0], pts[:, 1]
            x_bar, y_bar = np.mean(x), np.mean(y)

            # Least-squares slope a and intercept b
            denom = np.sum((x - x_bar) ** 2)

            if denom == 0:
                # Degenerate: all points share the same x (perfectly vertical)
                scores.append(100.0)
                continue

            a = np.sum((x - x_bar) * (y - y_bar)) / denom
            b = y_bar - a * x_bar

            # Perpendicular distance from each point to the line ax - y + b = 0
            d_i = np.abs(a * x - y + b) / np.sqrt(a ** 2 + 1)

            # RMS perpendicular distance
            d_rms = np.sqrt(np.mean(d_i ** 2))

            # Normalisation: half the bounding-box diagonal
            W = np.max(x) - np.min(x)
            H = np.max(y) - np.min(y)
            D = 0.5 * np.sqrt(W ** 2 + H ** 2)

            if D == 0:
                scores.append(100.0)
                continue

            d_norm = k * D
            S = 100.0 * (1.0 - d_rms / d_norm)
            scores.append(float(np.clip(S, 0.0, 100.0)))

        steadiness[label] = round(float(np.mean(scores)), 1)
    # Sort so Lille appears last
    finger_order = ["Pege", "Lange", "Ringe", "Lille"]
    steadiness = {f: steadiness[f] for f in finger_order if f in steadiness}

    return steadiness

def _setup_camera_window():
    """
    Create and position the 'Hand Tracking' cv2 window so it occupies the
    same proportional area it does on a 1920×1080 reference display.

    Reference layout (1920×1080):
      - Camera window : 640 × 480, anchored at x=0, vertically centred
      - Game window   : 1280 × 720, anchored at x=640, vertically centred

    On any other resolution the camera window is scaled by
      scale = min(screen_w / 1920, screen_h / 1080)
    so both windows keep the same relative footprint.
    """
    if platform.system() == "Windows":
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    else:
        try:
            import subprocess as _sp, re as _re
            out = _sp.check_output(["xrandr"]).decode()
            m = _re.search(r"current (\d+) x (\d+)", out)
            screen_w, screen_h = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
        except Exception:
            screen_w, screen_h = 1920, 1080

    scale  = min(screen_w / 1920.0, screen_h / 1080.0)
    win_w  = max(320, int(CAMERA_WIDTH  * scale))
    win_h  = max(240, int(CAMERA_HEIGHT * scale))
    win_y  = max(0, (screen_h - win_h) // 2)

    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracking", win_w, win_h)
    cv2.moveWindow("Hand Tracking", 0, win_y)


def main():
    """
    Main function: detect lego bricks, launch game, then hand tracking loop.
    """
    print("Starting hand rehabilitation session...")
    _setup_camera_window()

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded!")

    tracker = HandTracker(max_hands=1)

    if not (SCRIPT_DIR / "touch_history.json").exists():
        goal_lengths = finger_length_check(cap, tracker)
    else:
        goal_lengths = None

    # Detect lego bricks
    for attempt in range(MAX_DETECTION_RETRIES + 1):
        tracker.clickable_regions = []
        detect_legos(cap, model, tracker, detection_duration=2.0)

        if len(tracker.clickable_regions) >= MAX_LEGO_BRICKS:
            break

        if attempt < MAX_DETECTION_RETRIES:
            print(f"Only {len(tracker.clickable_regions)}/{MAX_LEGO_BRICKS} bricks detected. Retrying ({attempt + 1}/{MAX_DETECTION_RETRIES})...")
        else:
            if len(tracker.clickable_regions) == 0:
                print(f"No bricks detected after {MAX_DETECTION_RETRIES} retries. Closing.")
                cap.release()
                cv2.destroyAllWindows()
                return
            print(f"Only {len(tracker.clickable_regions)}/{MAX_LEGO_BRICKS} bricks detected after all retries. Continuing anyway.")

    Launch_the_game = True

    if Launch_the_game:
        # introduction to placing hand
        script_dir = Path(__file__).resolve().parent
        handplacement_img = cv2.imread(str(script_dir / "Handplacement.png"))
        canvas = None
        if handplacement_img is not None:
            # Scale image to fit the camera window while keeping aspect ratio
            hp_h, hp_w = handplacement_img.shape[:2]
            scale = min(CAMERA_WIDTH / hp_w, CAMERA_HEIGHT / hp_h)
            new_w, new_h = int(hp_w * scale), int(hp_h * scale)
            handplacement_img = cv2.resize(handplacement_img, (new_w, new_h))

            # Centre image on a black canvas the same size as the camera window
            canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            y_off = (CAMERA_HEIGHT - new_h) // 2
            x_off = (CAMERA_WIDTH  - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = handplacement_img

            # Instruction bar at the bottom
            font = cv2.FONT_HERSHEY_SIMPLEX
            msg = "Placer haanden som vist - vaelg et spil for at fortsaette"
            (tw, _), _ = cv2.getTextSize(msg, font, 0.65, 2)
            cv2.rectangle(canvas, (0, CAMERA_HEIGHT - 50), (CAMERA_WIDTH, CAMERA_HEIGHT), (30, 30, 30), -1)
            cv2.putText(canvas, msg, ((CAMERA_WIDTH - tw) // 2, CAMERA_HEIGHT - 15),
                        font, 0.65, (80, 255, 160), 2, cv2.LINE_AA)

            cv2.imshow("Hand Tracking", canvas)
            cv2.waitKey(1)   # ensure window is rendered before menu draws on top

        # ── Game selection loop: keep asking until the user presses Q ──────────
        touch_history = []
        region_to_key = {}
        while True:
            game_process, control_socket = launch_game(background_canvas=canvas)
            if game_process is None:
                # User pressed Q in the game menu → end session
                print("Session ended by user.")
                break

            region_to_key = choose_key_bindings(tracker.clickable_regions)
            touch_history = run_hand_tracking_loop(cap, tracker, control_socket, region_to_key, model, tracker.clickable_regions, game_process=game_process)

            control_socket.close()
            if game_process.poll() is None:
                game_process.terminate()

            print("Game ended. Returning to game selection...")
    else:
        region_to_key = choose_key_bindings(tracker.clickable_regions)
        print("Skipping game launch. Running hand tracking loop without game control.")
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        touch_history = run_hand_tracking_loop(cap, tracker, control_socket, region_to_key, model, tracker.clickable_regions)
        control_socket.close()

    #print("\nRaw touch history:")
    #for entry in touch_history:
    #    print(f"  Frame {entry['frame']}: {list(entry['touches'].keys())}")

    #Touch_history cleanup
    touch_history_trimmed = trim_touch_history(touch_history, keep_frames=3)

    #print("\nTrimmed touch history:")
    #for entry in touch_history_trimmed:
    #    print(f"  Frame {entry['frame']}: {list(entry['touches'].keys())}")

    steadiness = compute_steadiness(touch_history, tracker.clickable_regions)

    processed, finger_summary, finger_factor = process_touch_history(touch_history_trimmed, tracker.clickable_regions, region_to_key)

    # Inject steadiness into processed entries (0-1 scale to match accuracy/precision)
    for entry in processed:
        if "region" in entry:
            s_val = steadiness.get(entry["region"])
            if s_val is not None:
                entry["steadiness_score"] = round(s_val / 100.0, 10)
        if "overall_accuracy" in entry:
            prec_vals = [e["precision_score"]["overall"] for e in processed if "precision_score" in e]
            entry["overall_precision"] = sum(prec_vals) / len(prec_vals) if prec_vals else None
            s_vals = list(steadiness.values())
            entry["overall_steadiness"] = round((sum(s_vals) / len(s_vals)) / 100.0, 10) if s_vals else None

    # Sort processed entries by FINGER_NAMES order; keep the overall summary entry last
    _finger_order = {name: i for i, name in enumerate(FINGER_NAMES)}
    _finger_entries  = [e for e in processed if "region" in e]
    _overall_entries = [e for e in processed if "region" not in e]
    _finger_entries.sort(key=lambda e: _finger_order.get(e["region"], len(FINGER_NAMES)))
    processed = _finger_entries + _overall_entries

    # Ask save/discard in the camera window (Y / N)
    save_msg1 = "Tryk  Y  for at gemme data fra session"
    save_msg2 = "Tryk  N  for at slette session"
    font_save = cv2.FONT_HERSHEY_SIMPLEX
    save_history = None
    while save_history is None:
        ret, save_frame = cap.read()
        if ret:
            save_frame = cv2.flip(save_frame, -1)
        else:
            save_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        h_sf, w_sf = save_frame.shape[:2]
        ov = save_frame.copy()
        cv2.rectangle(ov, (0, h_sf // 2 - 70), (w_sf, h_sf // 2 + 70), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.75, save_frame, 0.25, 0, save_frame)
        for i, line in enumerate([save_msg1, save_msg2]):
            sz, _ = cv2.getTextSize(line, font_save, 0.8, 2)
            x = (w_sf - sz[0]) // 2
            y = h_sf // 2 - 14 + i * 50
            cv2.putText(save_frame, line, (x, y), font_save, 0.8, (80, 255, 160), 2, cv2.LINE_AA)
        cv2.imshow("Hand Tracking", save_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('y'):
            save_history = 'y'
        elif key == ord('n'):
            save_history = 'n'
        elif cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
            save_history = 'n'

    cap.release()
    cv2.destroyAllWindows()

    _SESSION_KEYS = {"processed", "finger_summary", "session_time", "session_number"}

    if save_history == 'n':
        print("Session history not saved.")
    elif save_history == 'y':
        _history_path = str(SCRIPT_DIR / "touch_history.json")
        if (SCRIPT_DIR / "touch_history.json").exists():
            with open(_history_path, "r") as f:
                existing_data = json.load(f)

            # --- Migrate old formats so data[0] is always pure metadata ---
            if isinstance(existing_data, dict):
                # Very old: single dict = metadata + session 1 merged
                meta = {k: v for k, v in existing_data.items() if k not in _SESSION_KEYS}
                sess1 = {k: v for k, v in existing_data.items() if k in _SESSION_KEYS}
                existing_data = [meta, sess1]
            elif "processed" in existing_data[0]:
                # Old list: session 1 data was mixed into data[0]
                meta = {k: v for k, v in existing_data[0].items() if k not in _SESSION_KEYS}
                sess1 = {k: v for k, v in existing_data[0].items() if k in _SESSION_KEYS}
                existing_data = [meta] + [sess1] + existing_data[1:]

            # data[0] is now metadata; sessions are data[1:]
            all_sessions = [s for s in existing_data if "processed" in s]
            new_session = {
                "processed": processed,
                "finger_summary": finger_summary,
                "session_time": datetime.now().isoformat(),
                "session_number": len(all_sessions) + 1
            }
            existing_data.append(new_session)
            all_sessions.append(new_session)

            # Recompute personal best across all sessions (including the new one)
            personal_best_updated = {
                "finger_lengths": {},
                "accuracy": {},
                "precision": {},
                "steadiness": {},
                "overall_accuracy": None,
                "overall_precision": None,
                "overall_steadiness": None,
            }
            for session in all_sessions:
                for finger, stats in session.get("finger_summary", {}).items():
                    fl = personal_best_updated["finger_lengths"]
                    if finger not in fl or stats["max"] > fl[finger]:
                        fl[finger] = stats["max"]
                for entry in session.get("processed", []):
                    if "region" in entry and "accuracy_score" in entry:
                        r, s = entry["region"], entry["accuracy_score"]
                        if r not in personal_best_updated["accuracy"] or s > personal_best_updated["accuracy"][r]:
                            personal_best_updated["accuracy"][r] = s
                    if "region" in entry and "precision_score" in entry:
                        r, s = entry["region"], entry["precision_score"]["overall"]
                        if r not in personal_best_updated["precision"] or s > personal_best_updated["precision"][r]:
                            personal_best_updated["precision"][r] = s
                    if "region" in entry and "steadiness_score" in entry:
                        r, s = entry["region"], entry["steadiness_score"]
                        if r not in personal_best_updated["steadiness"] or s > personal_best_updated["steadiness"][r]:
                            personal_best_updated["steadiness"][r] = s
                    if "overall_accuracy" in entry:
                        s = entry["overall_accuracy"]
                        if s is not None and (personal_best_updated["overall_accuracy"] is None or s > personal_best_updated["overall_accuracy"]):
                            personal_best_updated["overall_accuracy"] = s
                    if "overall_steadiness" in entry:
                        s = entry["overall_steadiness"]
                        if s is not None and (personal_best_updated["overall_steadiness"] is None or s > personal_best_updated["overall_steadiness"]):
                            personal_best_updated["overall_steadiness"] = s
                prec_vals = [e["precision_score"]["overall"] for e in session.get("processed", []) if "precision_score" in e]
                if prec_vals:
                    op = sum(prec_vals) / len(prec_vals)
                    if op is not None and (personal_best_updated["overall_precision"] is None or op > personal_best_updated["overall_precision"]):
                        personal_best_updated["overall_precision"] = op
            # Re-order sub-dicts to follow FINGER_NAMES order
            for _key in ("finger_lengths", "accuracy", "precision", "steadiness"):
                _d = personal_best_updated[_key]
                personal_best_updated[_key] = {f: _d[f] for f in FINGER_NAMES if f in _d}
            existing_data[0]["Personal_best"] = personal_best_updated
            with open(_history_path, "w") as f:
                json.dump(existing_data, f, indent=4)
        else:
            _pb_acc  = {e["region"]: e["accuracy_score"]            for e in processed if "region" in e and "accuracy_score"   in e}
            _pb_prec = {e["region"]: e["precision_score"]["overall"] for e in processed if "region" in e and "precision_score"  in e}
            _pb_sted = {e["region"]: e["steadiness_score"]           for e in processed if "region" in e and "steadiness_score" in e}
            personal_best_init = {
                "finger_lengths": {f: finger_summary[f]["max"] for f in FINGER_NAMES if f in finger_summary},
                "accuracy":   {f: _pb_acc[f]  for f in FINGER_NAMES if f in _pb_acc},
                "precision":  {f: _pb_prec[f] for f in FINGER_NAMES if f in _pb_prec},
                "steadiness": {f: _pb_sted[f] for f in FINGER_NAMES if f in _pb_sted},
                "overall_accuracy":   next((e["overall_accuracy"]   for e in processed if "overall_accuracy"   in e), None),
                "overall_precision":  next((e["overall_precision"]  for e in processed if "overall_precision"  in e), None),
                "overall_steadiness": next((e["overall_steadiness"] for e in processed if "overall_steadiness" in e), None),
            }
            # First-ever save: data[0] = pure metadata, data[1] = session 1
            with open(str(SCRIPT_DIR / "touch_history.json"), "w") as f:
                json.dump([
                    {
                        "brick_state": {},
                        "goal_finger_lengths": {finger: finger_factor * length for finger, length in goal_lengths.items()},
                        "Personal_best": personal_best_init,
                    },
                    {
                        "processed": processed,
                        "finger_summary": finger_summary,
                        "session_time": datetime.now().isoformat(),
                        "session_number": 1
                    }
                ], f, indent=4)

    _history_path = str(SCRIPT_DIR / "touch_history.json")
    if (SCRIPT_DIR / "touch_history.json").exists():
        print("Saved session history to touch_history.json")
    else:
        print("Ingen tidligere data at vise")

    if (SCRIPT_DIR / "touch_history.json").exists():
        generate_feedback(history_file=_history_path)
        show_feedback_window(history_file=_history_path)
        sugestive_changes(history_file=_history_path)


if __name__ == "__main__":
    main()

   