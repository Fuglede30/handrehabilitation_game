import pygame
import sys
import os
import math
import time
import ctypes as _ctypes
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml_control_bridge import MLControlBridge

# ── Screen-aware scaling ──────────────────────────────────────────────────────
# Match 1_Intro layout: game window is 1280×720 on a 1920×1080 reference screen,
# positioned at the right, vertically centred.
try:
    _sw = _ctypes.windll.user32.GetSystemMetrics(0)
    _sh = _ctypes.windll.user32.GetSystemMetrics(1)
except Exception:
    _sw, _sh = 1920, 1080

SCALE  = min(_sw / 1920.0, _sh / 1080.0)
WIDTH  = max(640, round(1280 * SCALE))
HEIGHT = max(360, round(720  * SCALE))

def _s(v: float) -> int:
    return max(1, round(v * SCALE))

# TILE reference is 64 px in a 1280-wide window (96 px was for 1920 fullscreen)
TILE = _s(64)
FPS  = 60

DEFAULT_PRESS_MODE = "single"
MOVE_DELAY = 0.18

# Position window identically to 1_Intro: right side of screen, vertically centred
_win_x = max(0, _sw - WIDTH)
_win_y = max(0, (_sh - HEIGHT) // 2)
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{_win_x},{_win_y}"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sokoban - 3 Levels")
clock = pygame.time.Clock()

font       = pygame.font.SysFont("Arial", _s(52), bold=True)
font_med   = pygame.font.SysFont("Arial", _s(30))
small_font = pygame.font.SysFont("Arial", _s(22))

pygame.mixer.init()
AUDIO_DIR        = Path(__file__).resolve().parent / "audio"
SPEECH_INTRO     = AUDIO_DIR / "tts_3_sokoban_first_page.mp3"
SPEECH_REMINDER  = AUDIO_DIR / "tts_husk_hånden.mp3"

REMINDER_SECONDS = 30   # play reminder 30 seconds after gameplay starts

def _play_speech(path: Path):
    """Play an MP3 via the music channel. Fails silently if file is missing."""
    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
    except Exception:
        pass

# ── Colours ───────────────────────────────────────────────────────────────────
COL_BG          = (245, 247, 252)
COL_WALL        = (54,  74,  99)
COL_FLOOR       = (226, 232, 240)
COL_GOAL        = (255, 205,  86)
COL_BOX         = (176, 120,  74)
COL_BOX_ON_GOAL = (101, 163,  13)
COL_PLAYER      = (220,  38,  38)
COL_TEXT        = (15,   23,  42)

# Finger colours (matching 1_Intro)
FINGER_COLORS = {
    "a": (230, 210,  45),   # Pegefinger  – gul
    "w": (145,  55, 210),   # Langefinger – lilla
    "s": ( 55, 185,  75),   # Ringfinger  – grøn
    "d": (210,  45,  45),   # Lillefinger – rød
}
FINGER_NAMES = {
    "a": "Pegefinger",
    "w": "Langefinger",
    "s": "Ringfinger",
    "d": "Lillefinger",
}
DIR_ARROWS = {
    "a": "←",
    "w": "↑",
    "s": "↓",
    "d": "→",
}

LEVELS = [
    [
        "##########",
        "#        #",
        "#  $ .   #",
        "#  @     #",
        "#        #",
        "##########",
    ],
    [
        "############",
        "#      .   #",
        "#  $$ ##   #",
        "#  @       #",
        "#      .   #",
        "############",
    ],
    [
        "#############",
        "#   .   .   #",
        "# ### ###   #",
        "#  $   $    #",
        "#   @   $   #",
        "#       .   #",
        "#############",
    ],
]


# ── Intro screen ──────────────────────────────────────────────────────────────

def draw_intro(tick, hand_img, anim_t):
    screen.fill((18, 22, 38))

    # Title
    title = font.render("Sokoban", True, (255, 255, 255))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, _s(40)))
    pygame.draw.rect(screen, (100, 120, 220),
                     (WIDTH // 2 - _s(80), _s(40) + title.get_height() + _s(4),
                      _s(160), _s(3)), border_radius=_s(2))

    # Description
    lines = [
        "I Sokoban er du en lagerarbejder, der skal skubbe kasser hen på de rigtige felter.",
        "Du er den røde mand, der skal skubbe de brune kasser hen på de gule felter.",
        "Imens du spiller, vil du kunne se i bunden, hvordan du bevæger din mand fra side til side og op og ned.",
        "I spillet vil det kunne være, at din kasse sidder fast, og du ikke kan få den ud, ",
        "da du kun kan skubbe i spillet og ikke trække. Hvis det sker, skal du trykke på R for at genstarte banen.",
        "Husk at placere hånden i udgangsstillingen som vist nedenfor, og forsøg så vidt muligt at have håndroden ",
        "på den multifarvede glatte overflade."
    ]
    y = _s(130)
    for line in lines:
        surf = font_med.render(line, True, (200, 210, 235))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
        y += _s(38)

    # Direction colour guide
    y += _s(10)
    dirs = [("w", "↑  Op"), ("a", "←  Venstre"), ("s", "↓  Ned"), ("d", "→  Højre")]
    btn_w, btn_h = _s(220), _s(44)
    gap = _s(12)
    total_w = len(dirs) * btn_w + (len(dirs) - 1) * gap
    bx = WIDTH // 2 - total_w // 2
    for key, label in dirs:
        col = FINGER_COLORS[key]
        rect = pygame.Rect(bx, y, btn_w, btn_h)
        pygame.draw.rect(screen, col, rect, border_radius=_s(10))
        pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=_s(10))
        name_surf = small_font.render(f"{label}  –  {FINGER_NAMES[key]}", True, (20, 20, 20))
        screen.blit(name_surf, (rect.centerx - name_surf.get_width() // 2,
                                rect.centery - name_surf.get_height() // 2))
        bx += btn_w + gap

    y += btn_h + _s(18)

    # Hand placement image
    if hand_img is not None:
        y += _s(22)
        screen.blit(hand_img, (WIDTH // 2 - hand_img.get_width() // 2, y))

    # Blinking continue prompt
    alpha = int(180 + 75 * math.sin(anim_t * 3.5))
    cont = font_med.render("► Tryk Enter for at starte", True, (100, min(255, alpha), 120))
    screen.blit(cont, (WIDTH // 2 - cont.get_width() // 2, HEIGHT - _s(50)))

    replay = small_font.render("R = gentag tale", True, (90, 100, 120))
    screen.blit(replay, (WIDTH // 2 - replay.get_width() // 2, HEIGHT - _s(24)))


# ── Direction HUD drawn during gameplay ───────────────────────────────────────

def draw_direction_hud():
    """
    Draws a compact colour-coded direction guide at the bottom of the screen
    so players always know which finger to use for each direction.
    """
    dirs = [("a", "← Venstre"), ("w", "↑ Op"), ("s", "↓ Ned"), ("d", "→ Højre")]
    btn_w, btn_h = _s(230), _s(36)
    gap = _s(10)
    total_w = len(dirs) * btn_w + (len(dirs) - 1) * gap
    bx = WIDTH // 2 - total_w // 2
    by = HEIGHT - btn_h - _s(8)

    for key, label in dirs:
        col = FINGER_COLORS[key]
        rect = pygame.Rect(bx, by, btn_w, btn_h)
        pygame.draw.rect(screen, col, rect, border_radius=_s(8))
        pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=_s(8))
        txt = small_font.render(f"{label}  –  {FINGER_NAMES[key]}", True, (20, 20, 20))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                          rect.centery - txt.get_height() // 2))
        bx += btn_w + gap


# ── Game logic ────────────────────────────────────────────────────────────────

class SokobanGame:
    def __init__(self):
        self.ml_controls = MLControlBridge(press_mode=DEFAULT_PRESS_MODE)
        self.level_index = 0
        self.won_all_levels = False
        self.load_level(self.level_index)

    def close(self):
        self.ml_controls.close()

    def load_level(self, index):
        self.level_index = index
        raw = LEVELS[index]
        self.grid_h = len(raw)
        self.grid_w = max(len(row) for row in raw)

        self.walls = set()
        self.goals = set()
        self.boxes = set()
        self.player = (1, 1)
        self.moves = 0
        self.last_move_key = None
        self.move_timer = 0.0
        self.level_clear_timer = 0.0   # counts down after a level is solved

        for y, row in enumerate(raw):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch == "#":
                    self.walls.add(pos)
                elif ch == ".":
                    self.goals.add(pos)
                elif ch == "$":
                    self.boxes.add(pos)
                elif ch == "*":
                    self.boxes.add(pos)
                    self.goals.add(pos)
                elif ch == "@":
                    self.player = pos
                elif ch == "+":
                    self.player = pos
                    self.goals.add(pos)

    def inside(self, pos):
        x, y = pos
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h

    def try_move(self, dx, dy):
        px, py = self.player
        next_pos = (px + dx, py + dy)

        if not self.inside(next_pos) or next_pos in self.walls:
            return False

        if next_pos in self.boxes:
            box_target = (next_pos[0] + dx, next_pos[1] + dy)
            if (not self.inside(box_target)) or (box_target in self.walls) or (box_target in self.boxes):
                return False
            self.boxes.remove(next_pos)
            self.boxes.add(box_target)

        self.player = next_pos
        self.moves += 1
        return True

    def level_complete(self):
        return all(goal in self.boxes for goal in self.goals)

    def next_level(self):
        if self.level_index + 1 < len(LEVELS):
            self.load_level(self.level_index + 1)
            return
        self.won_all_levels = True

    def restart_level(self):
        self.load_level(self.level_index)
        self.last_move_key = None

    def read_move_input(self):
        mapping = [
            ("w", (0, -1)),
            ("s", (0,  1)),
            ("a", (-1, 0)),
            ("d", (1,  0)),
        ]
        for name, delta in mapping:
            if self.ml_controls.get_input(name):
                return name, delta
        return None, (0, 0)

    def update(self, dt=0.0):
        self.ml_controls.pump()
        keys = pygame.key.get_pressed()
        self.move_timer = max(0.0, self.move_timer - dt)

        # Level-clear pause: freeze input, count down, then advance
        if self.level_clear_timer > 0:
            self.level_clear_timer -= dt
            if self.level_clear_timer <= 0:
                self.next_level()
            return

        if self.won_all_levels:
            if keys[pygame.K_w] or self.ml_controls.is_pressed("w"):
                self.won_all_levels = False
                self.load_level(0)
            return

        if keys[pygame.K_r]:
            self.restart_level()
            return

        move_key, (dx, dy) = self.read_move_input()
        if move_key is not None:
            self.try_move(dx, dy)
            if self.level_complete():
                self.level_clear_timer = 5.0   # 5-second pause before next level

    def _draw_player(self):
        """Draw a stick figure centred on the player's tile."""
        cx = self.offset_x + self.player[0] * TILE + TILE // 2
        cy = self.offset_y + self.player[1] * TILE + TILE // 2

        col      = COL_PLAYER           # red
        col_dark = (160, 20, 20)        # darker red for limbs
        lw       = max(2, TILE // 16)   # line width scales with tile

        # Proportions (all relative to TILE)
        head_r   = TILE // 7
        head_cy  = cy - TILE // 4
        neck_y   = head_cy + head_r
        waist_y  = cy + TILE // 8
        foot_y   = cy + TILE // 2 - _s(4)
        arm_y    = neck_y + (waist_y - neck_y) // 3
        arm_dx   = TILE // 4

        # Head (filled circle + outline)
        pygame.draw.circle(screen, col,      (cx, head_cy), head_r)
        pygame.draw.circle(screen, col_dark, (cx, head_cy), head_r, max(1, lw - 1))

        # Eyes
        eye_r  = max(1, head_r // 4)
        eye_dx = head_r // 3
        pygame.draw.circle(screen, (255, 255, 255), (cx - eye_dx, head_cy - 1), eye_r)
        pygame.draw.circle(screen, (255, 255, 255), (cx + eye_dx, head_cy - 1), eye_r)

        # Body
        pygame.draw.line(screen, col_dark, (cx, neck_y), (cx, waist_y), lw)

        # Arms
        pygame.draw.line(screen, col_dark, (cx - arm_dx, arm_y + _s(4)),
                         (cx, arm_y), lw)
        pygame.draw.line(screen, col_dark, (cx, arm_y),
                         (cx + arm_dx, arm_y + _s(4)), lw)

        # Legs
        pygame.draw.line(screen, col_dark, (cx, waist_y),
                         (cx - arm_dx + _s(2), foot_y), lw)
        pygame.draw.line(screen, col_dark, (cx, waist_y),
                         (cx + arm_dx - _s(2), foot_y), lw)

    def draw_tile(self, x, y):
        px = self.offset_x + x * TILE
        py = self.offset_y + y * TILE
        rect = pygame.Rect(px, py, TILE, TILE)
        pygame.draw.rect(screen, COL_FLOOR, rect)
        pygame.draw.rect(screen, (203, 213, 225), rect, 1)

    def draw(self):
        screen.fill(COL_BG)

        board_w_px = self.grid_w * TILE
        board_h_px = self.grid_h * TILE
        self.offset_x = (WIDTH  - board_w_px) // 2 - TILE
        self.offset_y = (HEIGHT - board_h_px) // 2 - TILE

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                self.draw_tile(x, y)

        for x, y in self.goals:
            cx = self.offset_x + x * TILE + TILE // 2
            cy = self.offset_y + y * TILE + TILE // 2
            pygame.draw.circle(screen, COL_GOAL, (cx, cy), TILE // 6)

        for x, y in self.walls:
            px = self.offset_x + x * TILE
            py = self.offset_y + y * TILE
            wall_rect = pygame.Rect(px, py, TILE, TILE)
            pygame.draw.rect(screen, COL_WALL, wall_rect)
            pygame.draw.rect(screen, (30, 41, 59), wall_rect, 2)

        for x, y in self.boxes:
            px = self.offset_x + x * TILE + _s(6)
            py = self.offset_y + y * TILE + _s(6)
            box_rect = pygame.Rect(px, py, TILE - _s(12), TILE - _s(12))
            color = COL_BOX_ON_GOAL if (x, y) in self.goals else COL_BOX
            pygame.draw.rect(screen, color, box_rect, border_radius=_s(6))
            pygame.draw.rect(screen, (71, 85, 105), box_rect, 2, border_radius=_s(6))

        self._draw_player()

        # Header
        header = font_med.render(
            f"Sokoban  |  Level {self.level_index + 1}/{len(LEVELS)}  |  Træk: {self.moves}",
            True, COL_TEXT)
        screen.blit(header, (_s(20), _s(12)))

        reset_hint = small_font.render("R = nulstil niveau", True, (120, 130, 150))
        screen.blit(reset_hint, (WIDTH - reset_hint.get_width() - _s(20), _s(14)))

        # Direction colour HUD
        draw_direction_hud()

        # Level-clear overlay
        if self.level_clear_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 30, 10, 160))
            screen.blit(overlay, (0, 0))
            t1 = font.render("Niveau løst!", True, (120, 255, 140))
            t2 = font_med.render("Næste bane starter om lidt…", True, (200, 235, 200))
            secs = math.ceil(self.level_clear_timer)
            t3 = small_font.render(f"{secs}", True, (160, 220, 160))
            screen.blit(t1, (WIDTH // 2 - t1.get_width() // 2, HEIGHT // 2 - _s(55)))
            screen.blit(t2, (WIDTH // 2 - t2.get_width() // 2, HEIGHT // 2 + _s(4)))
            screen.blit(t3, (WIDTH // 2 - t3.get_width() // 2, HEIGHT // 2 + _s(46)))

        # Win overlay
        if self.won_all_levels:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 130))
            screen.blit(overlay, (0, 0))
            t1 = font.render("Du klarede alle 3 niveauer!", True, (255, 255, 255))
            t2 = font_med.render("Tryk ↑ (Langefinger) for at spille igen", True, (255, 255, 255))
            screen.blit(t1, (WIDTH // 2 - t1.get_width() // 2, HEIGHT // 2 - _s(30)))
            screen.blit(t2, (WIDTH // 2 - t2.get_width() // 2, HEIGHT // 2 + _s(18)))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    # Load hand placement image for intro
    hand_img = None
    img_path = Path(__file__).resolve().parent / "Handplacement.png"
    if img_path.exists():
        raw = pygame.image.load(str(img_path)).convert_alpha()
        rw, rh = raw.get_size()
        max_w, max_h = _s(300), _s(160)
        scale = min(max_w / rw, max_h / rh)
        hand_img = pygame.transform.smoothscale(raw, (int(rw * scale), int(rh * scale)))

    state  = "intro"
    anim_t = 0.0
    game   = SokobanGame()
    game_start_time = None   # set when gameplay begins
    reminder_played = False  # True once the 2-min reminder has fired

    _play_speech(SPEECH_INTRO)

    while True:
        dt = clock.tick(FPS) / 1000.0
        anim_t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game.close()
                    pygame.quit()
                    sys.exit()
                if state == "intro" and event.key == pygame.K_r:
                    _play_speech(SPEECH_INTRO)
                if state == "intro" and event.key == pygame.K_RETURN:
                    state = "playing"
                    game_start_time = time.time()
                    reminder_played = False

        if state == "intro":
            draw_intro(0, hand_img, anim_t)
        else:
            game.update(dt)
            game.draw()
            # Play the hand-reminder after 2 minutes of gameplay
            if (not reminder_played and game_start_time is not None
                    and time.time() - game_start_time >= REMINDER_SECONDS):
                _play_speech(SPEECH_REMINDER)
                reminder_played = True

        pygame.display.flip()


if __name__ == "__main__":
    main()
