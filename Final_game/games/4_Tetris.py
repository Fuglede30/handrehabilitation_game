import pygame
import random
import sys
import math
import os
import ctypes as _ctypes
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml_control_bridge import MLControlBridge

# ── Screen-aware scaling ──────────────────────────────────────────────────────
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

_win_x = max(0, _sw - WIDTH)
_win_y = max(0, (_sh - HEIGHT) // 2)
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{_win_x},{_win_y}"

pygame.init()
pygame.mixer.init()

# ── Layout ────────────────────────────────────────────────────────────────────
BLOCK_SIZE    = _s(28)
GRID_WIDTH    = 10
GRID_HEIGHT   = 20
PANEL_W       = _s(160)
GRID_PIXEL_W  = GRID_WIDTH  * BLOCK_SIZE
GRID_PIXEL_H  = GRID_HEIGHT * BLOCK_SIZE
GRID_OFFSET_X = (WIDTH  - GRID_PIXEL_W) // 2
GRID_OFFSET_Y = (HEIGHT - GRID_PIXEL_H) // 2

FPS          = 60
FALL_SPEED   = 1500   # ms between auto-drops

DEFAULT_PRESS_MODE = "single"

AUDIO_DIR      = Path(__file__).resolve().parent / "audio"
SPEECH_INTRO   = AUDIO_DIR / "tts_4_tetris_first_page.mp3"
REMINDER_SECONDS = 30

HIGHSCORE_FILE = Path(__file__).resolve().parent / "tetris_highscore.txt"

def _load_highscore() -> int:
    try:
        return int(HIGHSCORE_FILE.read_text().strip())
    except Exception:
        return 0

def _save_highscore(score: int):
    try:
        HIGHSCORE_FILE.write_text(str(score))
    except Exception:
        pass

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
GRAY   = (80,  85,  95)
DARK   = (18,  22,  38)
PANEL  = (28,  32,  50)

CYAN    = (0,   240, 240)
BLUE    = (0,   0,   240)
ORANGE  = (240, 160,   0)
YELLOW  = (240, 240,   0)
GREEN   = (0,   240,   0)
MAGENTA = (200,   0, 200)
RED     = (240,   0,   0)

# Finger colours (same across all games)
FINGER_COLORS = {
    "a": (230, 210,  45),   # Pegefinger  – gul   → move left
    "d": (210,  45,  45),   # Lillefinger – rød   → move right
    "w": (145,  55, 210),   # Langefinger – lilla → rotate right
    "s": ( 55, 185,  75),   # Ringfinger  – grøn  → rotate left
}
FINGER_NAMES = {
    "a": "Pegefinger",
    "d": "Lillefinger",
    "w": "Langefinger",
    "s": "Ringfinger",
}

# ── Controls reference used in intro + in-game HUD ───────────────────────────
# Each entry: (key, arrow_symbol, action_label, finger_name)
CONTROLS = [
    ("a", "←", "Venstre",  "Pegefinger"),
    ("d", "→", "Højre",    "Lillefinger"),
    ("w", "",  "Drej",     "Langefinger"),
    ("s", "↓", "Ned",      "Ringfinger"),
]

# ── Tetromino shapes ──────────────────────────────────────────────────────────
SHAPES = [
    # I
    [['.....',
      '.....',
      '.....',
      'OOOO.',
      '.....'],
     ['.....',
      '..O..',
      '..O..',
      '..O..',
      '..O..']],
    # T
    [['.....',
      '.....',
      '..O..',
      '.OOO.',
      '.....'],
     ['.....',
      '..O..',
      '..OO.',
      '..O..',
      '.....'],
     ['.....',
      '.....',
      '.OOO.',
      '..O..',
      '.....'],
     ['.....',
      '..O..',
      '.OO..',
      '..O..',
      '.....']],
    # S
    [['.....',
      '.....',
      '..OO.',
      '.OO..',
      '.....'],
     ['.....',
      '..O..',
      '..OO.',
      '...O.',
      '.....']],
    # Z
    [['.....',
      '.....',
      '.OO..',
      '..OO.',
      '.....'],
     ['.....',
      '...O.',
      '..OO.',
      '..O..',
      '.....']],
    # O
    [['.....',
      '.....',
      '.OO..',
      '.OO..',
      '.....']],
    # J
    [['.....',
      '..O..',
      '..O..',
      '.OO..',
      '.....'],
     ['.....',
      '.O...',
      '.OOO.',
      '.....',
      '.....'],
     ['.....',
      '..OO.',
      '..O..',
      '..O..',
      '.....'],
     ['.....',
      '.....',
      '.OOO.',
      '...O.',
      '.....']],
    # L
    [['.....',
      '.....',
      '.OOO.',
      '.O...',
      '.....'],
     ['.....',
      '.OO..',
      '..O..',
      '..O..',
      '.....'],
     ['.....',
      '...O.',
      '.OOO.',
      '.....',
      '.....'],
     ['.....',
      '..O..',
      '..O..',
      '..OO.',
      '.....']]
]

SHAPE_COLORS = [CYAN, BLUE, ORANGE, YELLOW, GREEN, MAGENTA, RED]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _play_speech(path: Path):
    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
    except Exception:
        pass


def _draw_text_centered(surf, msg, font, color, cx, y):
    s = font.render(msg, True, color)
    surf.blit(s, (cx - s.get_width() // 2, y))


def _draw_piece_preview(surf, shape_idx, rotation, cx, cy, block_size):
    """Draw a tetromino centred at (cx, cy)."""
    shape = SHAPES[shape_idx][rotation % len(SHAPES[shape_idx])]
    color = SHAPE_COLORS[shape_idx]
    cells = [(j, i) for i, row in enumerate(shape) for j, cell in enumerate(row) if cell == 'O']
    if not cells:
        return
    min_j = min(c[0] for c in cells)
    min_i = min(c[1] for c in cells)
    max_j = max(c[0] for c in cells)
    max_i = max(c[1] for c in cells)
    w = (max_j - min_j + 1) * block_size
    h = (max_i - min_i + 1) * block_size
    ox = cx - w // 2
    oy = cy - h // 2
    for (j, i) in cells:
        rx = ox + (j - min_j) * block_size
        ry = oy + (i - min_i) * block_size
        pygame.draw.rect(surf, color,
                         (rx, ry, block_size - 1, block_size - 1), border_radius=_s(3))
        pygame.draw.rect(surf, WHITE,
                         (rx, ry, block_size - 1, block_size - 1), 1, border_radius=_s(3))


# ── Phases ────────────────────────────────────────────────────────────────────
PHASE_INTRO    = 0
PHASE_PLAYING  = 1
PHASE_GAMEOVER = 2


# ══════════════════════════════════════════════════════════════════════════════
class Tetris:
# ══════════════════════════════════════════════════════════════════════════════

    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tetris – Håndrehabilitering")
        self.clock  = pygame.time.Clock()

        self.f_title = pygame.font.SysFont("Arial", _s(52), bold=True)
        self.f_large = pygame.font.SysFont("Arial", _s(36), bold=True)
        self.f_med   = pygame.font.SysFont("Arial", _s(26))
        self.f_small = pygame.font.SysFont("Arial", _s(20))
        self.f_hud   = pygame.font.SysFont("Arial", _s(20), bold=True)
        self.f_icon  = pygame.font.SysFont("Arial", _s(24), bold=True)

        img_path = Path(__file__).resolve().parent / "Handplacement.png"
        try:
            raw = pygame.image.load(str(img_path)).convert_alpha()
            max_w, max_h = _s(260), _s(160)
            sc = min(max_w / raw.get_width(), max_h / raw.get_height())
            self.hand_img = pygame.transform.smoothscale(
                raw, (int(raw.get_width() * sc), int(raw.get_height() * sc)))
        except Exception:
            self.hand_img = None

        self.ml_controls = MLControlBridge(press_mode=DEFAULT_PRESS_MODE)

        self.phase   = PHASE_INTRO
        self.anim_t  = 0.0
        self.gameplay_start_time = None
        self.reminder_played     = False
        self.highscore           = _load_highscore()

        _play_speech(SPEECH_INTRO)
        self._init_game()

    def _update_highscore(self):
        """Save highscore if current score beats it. Safe to call at any time."""
        if self.score > self.highscore:
            self.highscore = self.score
            _save_highscore(self.highscore)

    # ── Game state ────────────────────────────────────────────────────────────

    def _init_game(self):
        self.grid          = [[None] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.score         = 0
        self.lines_cleared = 0
        self.last_fall     = pygame.time.get_ticks()

        self.piece_idx   = random.randint(0, len(SHAPES) - 1)
        self.piece_x     = GRID_WIDTH // 2 - 2
        self.piece_y     = 0
        self.piece_rot   = 0
        self.piece_color = SHAPE_COLORS[self.piece_idx]

        self.next_idx  = random.randint(0, len(SHAPES) - 1)
        self.hold_idx  = None
        self.hold_used = False

    # ── Piece logic ───────────────────────────────────────────────────────────

    def _get_shape(self, idx, rot):
        return SHAPES[idx][rot % len(SHAPES[idx])]

    def _check_collision(self, x, y, idx, rot):
        for i, row in enumerate(self._get_shape(idx, rot)):
            for j, cell in enumerate(row):
                if cell == 'O':
                    nx, ny = x + j, y + i
                    if nx < 0 or nx >= GRID_WIDTH or ny >= GRID_HEIGHT:
                        return True
                    if ny >= 0 and self.grid[ny][nx] is not None:
                        return True
        return False

    def _spawn_next(self):
        self.piece_idx   = self.next_idx
        self.piece_x     = GRID_WIDTH // 2 - 2
        self.piece_y     = 0
        self.piece_rot   = 0
        self.piece_color = SHAPE_COLORS[self.piece_idx]
        self.hold_used   = False
        self.next_idx    = random.randint(0, len(SHAPES) - 1)
        if self._check_collision(self.piece_x, self.piece_y,
                                  self.piece_idx, self.piece_rot):
            # Update and save highscore before showing game over
            if self.score > self.highscore:
                self.highscore = self.score
                _save_highscore(self.highscore)
            self.phase = PHASE_GAMEOVER

    def _lock_piece(self):
        for i, row in enumerate(self._get_shape(self.piece_idx, self.piece_rot)):
            for j, cell in enumerate(row):
                if cell == 'O':
                    ny, nx = self.piece_y + i, self.piece_x + j
                    if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                        self.grid[ny][nx] = self.piece_color
        self._clear_lines()
        self._spawn_next()

    def _clear_lines(self):
        full = [r for r in range(GRID_HEIGHT) if all(self.grid[r])]
        for r in full:
            del self.grid[r]
            self.grid.insert(0, [None] * GRID_WIDTH)
        n = len(full)
        self.lines_cleared += n
        self.score += [0, 100, 300, 500, 800][min(n, 4)]
        self._update_highscore()

    def _move(self, dx, dy):
        nx, ny = self.piece_x + dx, self.piece_y + dy
        if not self._check_collision(nx, ny, self.piece_idx, self.piece_rot):
            self.piece_x, self.piece_y = nx, ny
            return True
        return False

    def _rotate_right(self):
        """Rotate clockwise (W key / Langefinger)."""
        new_rot = (self.piece_rot + 1) % len(SHAPES[self.piece_idx])
        for kick in (0, 1, -1, 2, -2):
            if not self._check_collision(self.piece_x + kick, self.piece_y,
                                         self.piece_idx, new_rot):
                self.piece_x  += kick
                self.piece_rot = new_rot
                return

    def _rotate_left(self):
        """Rotate counter-clockwise (S key / Ringfinger)."""
        new_rot = (self.piece_rot - 1) % len(SHAPES[self.piece_idx])
        for kick in (0, 1, -1, 2, -2):
            if not self._check_collision(self.piece_x + kick, self.piece_y,
                                         self.piece_idx, new_rot):
                self.piece_x  += kick
                self.piece_rot = new_rot
                return

    def _hold_piece(self):
        if self.hold_used:
            return
        if self.hold_idx is None:
            self.hold_idx = self.piece_idx
            self._spawn_next()
        else:
            self.hold_idx, self.piece_idx = self.piece_idx, self.hold_idx
            self.piece_x     = GRID_WIDTH // 2 - 2
            self.piece_y     = 0
            self.piece_rot   = 0
            self.piece_color = SHAPE_COLORS[self.piece_idx]
            if self._check_collision(self.piece_x, self.piece_y,
                                      self.piece_idx, self.piece_rot):
                self.hold_idx, self.piece_idx = self.piece_idx, self.hold_idx
                self.piece_color = SHAPE_COLORS[self.piece_idx]
                return
        self.hold_used = True

    def _ghost_y(self):
        y = self.piece_y
        while not self._check_collision(self.piece_x, y + 1,
                                         self.piece_idx, self.piece_rot):
            y += 1
        return y

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_block(self, x, y, color, alpha=255):
        px = GRID_OFFSET_X + x * BLOCK_SIZE
        py = GRID_OFFSET_Y + y * BLOCK_SIZE
        if alpha < 255:
            s = pygame.Surface((BLOCK_SIZE - 1, BLOCK_SIZE - 1), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.screen.blit(s, (px, py))
        else:
            pygame.draw.rect(self.screen, color,
                             (px, py, BLOCK_SIZE - 1, BLOCK_SIZE - 1),
                             border_radius=_s(3))
            pygame.draw.rect(self.screen, WHITE,
                             (px, py, BLOCK_SIZE - 1, BLOCK_SIZE - 1),
                             1, border_radius=_s(3))

    def _draw_panel_box(self, label, cx, cy, w, h):
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=_s(8))
        pygame.draw.rect(self.screen, GRAY,  rect, 1, border_radius=_s(8))
        lbl = self.f_hud.render(label, True, (160, 170, 200))
        self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                                rect.y + _s(6)))
        return rect

    def _draw_control_pill(self, key, arrow, action, cy, panel_cx, pill_w, pill_h):
        """Draw one color-coded control button on a side panel."""
        col  = FINGER_COLORS[key]
        rect = pygame.Rect(panel_cx - pill_w // 2, cy - pill_h // 2, pill_w, pill_h)
        pygame.draw.rect(self.screen, col, rect, border_radius=_s(8))
        pygame.draw.rect(self.screen, WHITE, rect, 1, border_radius=_s(8))

        # Action text centered on the pill
        label = f"{arrow}  {action}" if arrow else action
        lbl_s = self.f_small.render(label, True, (20, 20, 20))
        self.screen.blit(lbl_s, (rect.centerx - lbl_s.get_width() // 2,
                                  rect.centery - lbl_s.get_height() // 2))

    # ── Intro screen ──────────────────────────────────────────────────────────

    def _draw_intro(self):
        self.screen.fill(DARK)

        title = self.f_title.render("Tetris", True, WHITE)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, _s(32)))
        pygame.draw.rect(self.screen, (100, 120, 220),
                         (WIDTH // 2 - _s(60), _s(32) + title.get_height() + _s(4),
                          _s(120), _s(3)), border_radius=_s(2))

        lines = [
            "I Tetris falder brikker ned fra toppen – stable dem uden huller.",
            "Bevæg brikken til venstre og højre, og drej den så den passer ind.",
            "Når en hel vandret række er fyldt, forsvinder den og du får point.",
            "Gem en brik til senere ved at trykke på den blå brik – byt den ind igen når du vil.",
            "Spillet er slut, hvis brikkerne når op til toppen.",
        ]
        y = _s(118)
        for line in lines:
            surf = self.f_med.render(line, True, (200, 210, 235))
            self.screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
            y += _s(38)

        y += _s(14)

        # Color-coded control buttons — 4 fingers
        btn_w, btn_h = _s(230), _s(50)
        gap   = _s(10)
        total = len(CONTROLS) * btn_w + (len(CONTROLS) - 1) * gap
        bx    = WIDTH // 2 - total // 2

        for key, arrow, action, finger in CONTROLS:
            col  = FINGER_COLORS[key]
            rect = pygame.Rect(bx, y, btn_w, btn_h)
            pygame.draw.rect(self.screen, col, rect, border_radius=_s(10))
            pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=_s(10))

            # Action text centered on button
            label = f"{arrow}  {action}" if arrow else action
            lbl_s = self.f_med.render(label, True, (20, 20, 20))
            self.screen.blit(lbl_s, (rect.centerx - lbl_s.get_width() // 2,
                                      rect.centery - lbl_s.get_height() // 2))

            bx += btn_w + gap

        y += btn_h + _s(12)

        # Hold button — blue, centered
        hold_rect = pygame.Rect(WIDTH // 2 - _s(150), y, _s(300), btn_h)
        pygame.draw.rect(self.screen, (60, 110, 210), hold_rect, border_radius=_s(10))
        pygame.draw.rect(self.screen, WHITE, hold_rect, 2, border_radius=_s(10))
        act_s = self.f_med.render("Gem / hent brik", True, (230, 235, 255))
        self.screen.blit(act_s, (hold_rect.centerx - act_s.get_width() // 2,
                                  hold_rect.centery - act_s.get_height() // 2))

        y += btn_h + _s(16)

        if self.hand_img:
            self.screen.blit(self.hand_img,
                              (WIDTH // 2 - self.hand_img.get_width() // 2, y))

        alpha = int(180 + 75 * math.sin(self.anim_t * 3.5))
        cont  = self.f_med.render("Tryk Enter for at starte", True,
                                   (100, min(255, alpha), 120))
        self.screen.blit(cont, (WIDTH // 2 - cont.get_width() // 2, HEIGHT - _s(46)))

        replay = self.f_small.render("R = gentag tale", True, (90, 100, 120))
        self.screen.blit(replay, (WIDTH // 2 - replay.get_width() // 2, HEIGHT - _s(22)))

    # ── Gameplay ──────────────────────────────────────────────────────────────

    def _draw_game(self):
        self.screen.fill(DARK)

        # Grid cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(self.screen, (30, 34, 50),
                                 (GRID_OFFSET_X + x * BLOCK_SIZE,
                                  GRID_OFFSET_Y + y * BLOCK_SIZE,
                                  BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        pygame.draw.rect(self.screen, GRAY,
                         (GRID_OFFSET_X - 1, GRID_OFFSET_Y - 1,
                          GRID_PIXEL_W + 2, GRID_PIXEL_H + 2), 1)

        # Locked pieces
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                col = self.grid[y][x]
                if col:
                    self._draw_block(x, y, col)

        # Ghost
        ghost_y = self._ghost_y()
        if ghost_y != self.piece_y:
            for i, row in enumerate(self._get_shape(self.piece_idx, self.piece_rot)):
                for j, cell in enumerate(row):
                    if cell == 'O':
                        self._draw_block(self.piece_x + j, ghost_y + i,
                                         self.piece_color, alpha=55)

        # Current piece
        if self.phase == PHASE_PLAYING:
            for i, row in enumerate(self._get_shape(self.piece_idx, self.piece_rot)):
                for j, cell in enumerate(row):
                    if cell == 'O':
                        self._draw_block(self.piece_x + j, self.piece_y + i,
                                         self.piece_color)

        # ── Left panel: HOLD + control guide ─────────────────────────────────
        left_cx  = GRID_OFFSET_X - PANEL_W // 2 - _s(10)
        panel_h  = _s(110)

        # HOLD box
        hold_rect = self._draw_panel_box("HOLD", left_cx,
                                          GRID_OFFSET_Y + _s(60), PANEL_W, panel_h)
        if self.hold_idx is not None:
            draw_col = SHAPE_COLORS[self.hold_idx]
            if self.hold_used:
                draw_col = tuple(max(0, c - 100) for c in draw_col)
            _draw_piece_preview(self.screen, self.hold_idx, 0,
                                 hold_rect.centerx,
                                 hold_rect.centery + _s(12), _s(18))
        if self.hold_used:
            used_s = self.f_small.render("Låst", True, (180, 80, 80))
            self.screen.blit(used_s, (hold_rect.centerx - used_s.get_width() // 2,
                                       hold_rect.bottom - _s(18)))

        # Color-coded control guide (left panel, below HOLD)
        pill_w  = PANEL_W - _s(8)
        pill_h  = _s(38)
        gap     = _s(6)
        guide_y = hold_rect.bottom + _s(14)

        for key, arrow, action, finger in CONTROLS:
            self._draw_control_pill(key, arrow, action,
                                     guide_y + pill_h // 2, left_cx,
                                     pill_w, pill_h)
            guide_y += pill_h + gap

        # Hold pill
        f_rect = pygame.Rect(left_cx - pill_w // 2, guide_y, pill_w, pill_h)
        pygame.draw.rect(self.screen, (60, 110, 210), f_rect, border_radius=_s(8))
        pygame.draw.rect(self.screen, WHITE, f_rect, 1, border_radius=_s(8))
        save_s = self.f_small.render("Gem / hent brik", True, (200, 215, 255))
        self.screen.blit(save_s, (f_rect.centerx - save_s.get_width() // 2,
                                   f_rect.centery - save_s.get_height() // 2))

        # ── Right panel: NEXT + SCORE + LINES ────────────────────────────────
        right_cx = GRID_OFFSET_X + GRID_PIXEL_W + PANEL_W // 2 + _s(10)

        next_rect = self._draw_panel_box("NÆSTE", right_cx,
                                          GRID_OFFSET_Y + _s(60), PANEL_W, panel_h)
        _draw_piece_preview(self.screen, self.next_idx, 0,
                             next_rect.centerx, next_rect.centery + _s(12), _s(18))

        score_rect = self._draw_panel_box("SCORE", right_cx,
                                           GRID_OFFSET_Y + panel_h + _s(30),
                                           PANEL_W, _s(80))
        sc_s = self.f_large.render(str(self.score), True, (255, 230, 100))
        self.screen.blit(sc_s, (score_rect.centerx - sc_s.get_width() // 2,
                                  score_rect.centery - sc_s.get_height() // 2 + _s(8)))

        lines_rect = self._draw_panel_box("LINJER", right_cx,
                                           GRID_OFFSET_Y + panel_h + _s(130),
                                           PANEL_W, _s(70))
        lc_s = self.f_large.render(str(self.lines_cleared), True, (140, 210, 255))
        self.screen.blit(lc_s, (lines_rect.centerx - lc_s.get_width() // 2,
                                  lines_rect.centery - lc_s.get_height() // 2 + _s(6)))

        hs_rect = self._draw_panel_box("HIGHSCORE", right_cx,
                                        GRID_OFFSET_Y + panel_h + _s(230),
                                        PANEL_W, _s(70))
        hs_col = (255, 200, 50) if self.score >= self.highscore and self.score > 0 else (200, 170, 80)
        hs_s = self.f_large.render(str(self.highscore), True, hs_col)
        self.screen.blit(hs_s, (hs_rect.centerx - hs_s.get_width() // 2,
                                  hs_rect.centery - hs_s.get_height() // 2 + _s(6)))

    # ── Game-over overlay ─────────────────────────────────────────────────────

    def _draw_gameover(self):
        self._draw_game()
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        _draw_text_centered(self.screen, "GAME OVER",    self.f_title,
                             (240, 60, 60),   WIDTH // 2, HEIGHT // 2 - _s(80))
        _draw_text_centered(self.screen, f"Score: {self.score}", self.f_large,
                             (255, 230, 100), WIDTH // 2, HEIGHT // 2 - _s(20))
        _draw_text_centered(self.screen, f"Linjer: {self.lines_cleared}", self.f_med,
                             (180, 220, 255), WIDTH // 2, HEIGHT // 2 + _s(28))
        if self.score > 0 and self.score >= self.highscore:
            _draw_text_centered(self.screen, "Ny highscore!", self.f_med,
                                 (255, 210, 40), WIDTH // 2, HEIGHT // 2 + _s(64))
        else:
            _draw_text_centered(self.screen, f"Highscore: {self.highscore}", self.f_med,
                                 (200, 170, 80), WIDTH // 2, HEIGHT // 2 + _s(64))
        _draw_text_centered(self.screen,
                             "Tryk Enter for at spille igen  |  Q for at afslutte",
                             self.f_small, (160, 170, 200),
                             WIDTH // 2, HEIGHT // 2 + _s(105))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.anim_t += dt
            self.ml_controls.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._update_highscore()
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if self.phase == PHASE_INTRO:
                        if event.key == pygame.K_RETURN:
                            self._init_game()
                            self.phase = PHASE_PLAYING
                            self.gameplay_start_time = pygame.time.get_ticks()
                            self.reminder_played = False
                        elif event.key == pygame.K_r:
                            _play_speech(SPEECH_INTRO)
                        elif event.key == pygame.K_ESCAPE:
                            self._update_highscore()
                            running = False

                    elif self.phase == PHASE_PLAYING:
                        if   event.key in (pygame.K_LEFT,  pygame.K_a):
                            self._move(-1, 0)
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            self._move(1, 0)
                        elif event.key in (pygame.K_UP,    pygame.K_w):
                            self._rotate_right()
                        elif event.key in (pygame.K_DOWN,  pygame.K_s):
                            self._move(0, 1)
                        elif event.key == pygame.K_f:
                            self._hold_piece()
                        elif event.key == pygame.K_q:
                            self._update_highscore()
                            running = False
                        elif event.key == pygame.K_ESCAPE:
                            self._update_highscore()
                            running = False

                    elif self.phase == PHASE_GAMEOVER:
                        if event.key == pygame.K_RETURN:
                            self._init_game()
                            self.phase = PHASE_PLAYING
                            self.gameplay_start_time = pygame.time.get_ticks()
                            self.reminder_played = False
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

            # ML hand inputs
            if self.phase == PHASE_PLAYING:
                if self.ml_controls.get_input('a'):
                    self._move(-1, 0)
                if self.ml_controls.get_input('d'):
                    self._move(1, 0)
                if self.ml_controls.get_input('w'):
                    self._rotate_right()
                if self.ml_controls.get_input('s'):
                    self._move(0, 1)
                if self.ml_controls.get_input('f'):
                    self._hold_piece()

                # Hand-placement reminder
                if self.gameplay_start_time and not self.reminder_played:
                    elapsed = (pygame.time.get_ticks() - self.gameplay_start_time) / 1000.0
                    if elapsed >= REMINDER_SECONDS:
                        _play_speech(AUDIO_DIR / "tts_husk_hånden.mp3")
                        self.reminder_played = True

                # Auto-drop
                now = pygame.time.get_ticks()
                if now - self.last_fall > FALL_SPEED:
                    if not self._move(0, 1):
                        self._lock_piece()
                    self.last_fall = now

            if self.phase == PHASE_INTRO:
                self._draw_intro()
            elif self.phase == PHASE_PLAYING:
                self._draw_game()
            elif self.phase == PHASE_GAMEOVER:
                self._draw_gameover()

            pygame.display.flip()

        self.ml_controls.close()
        pygame.quit()


def main():
    game = Tetris()
    game.run()


if __name__ == "__main__":
    main()
