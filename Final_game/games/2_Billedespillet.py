import os
import random
import pygame
import sys
import time
import traceback
import ctypes as _ctypes
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml_control_bridge import MLControlBridge

# ─────────────────────────────────────────────
#  Screen-aware scaling
#  Reference: W=700, H=800 at 1920×1080.
#  Scale to fill the full screen height on any display.
# ─────────────────────────────────────────────
try:
    _sw = _ctypes.windll.user32.GetSystemMetrics(0)
    _sh = _ctypes.windll.user32.GetSystemMetrics(1)
except Exception:
    _sw, _sh = 1920, 1080

_TARGET_H = _sh - 80   # leave ~80 px for taskbar + title bar
SCALE = _TARGET_H / 800.0

def _s(v: float) -> int:
    """Scale a reference-resolution pixel value to the current screen size."""
    return max(1, round(v * SCALE))

# ─────────────────────────────────────────────
#  Window & layout config
# ─────────────────────────────────────────────
W, H       = _s(700), _TARGET_H    # slightly less than screen height
FPS        = 60
PICS_DIR   = Path(__file__).resolve().parent / "pictures"

GRID_COLS  = 4
GRID_ROWS  = 4

PIC_MAX_W  = _s(580)
PIC_MAX_H  = _s(420)
PIC_Y      = _s(100)
PIC_BOX_X  = (W - PIC_MAX_W) // 2

# Camera window width from Main_Demo (640 × min(sw/1920, sh/1080))
HAND_WIN_W = round(640 * min(_sw / 1920.0, _sh / 1080.0))

# Audio
AUDIO_DIR  = Path(__file__).resolve().parent / "audio"
SPEECH_P1       = AUDIO_DIR / "tts_2_picture_game_first_page.mp3"
SPEECH_P2       = AUDIO_DIR / "tts_2_picture_game_second_page.mp3"
SPEECH_REMINDER = AUDIO_DIR / "tts_husk_hånden.mp3"

REMINDER_SECONDS = 30   # play reminder 30 seconds after gameplay starts

def _play_speech(path: Path):
    """Play an MP3 via the music channel. Fails silently if file is missing."""
    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
    except Exception:
        pass

SAMPLE_RATE = 44100

# Musikalske frekvenser – én per finger (samme som i 1_Intro.py)
KEY_FREQS = {
    'a': 784,   # G5  – Pegefinger
    'w': 659,   # E5  – Langefinger
    's': 523,   # C5  – Ringefinger
    'd': 392,   # G4  – Lillefinger
}

def _make_tone(freq: float, duration: float = 0.18, volume: float = 0.55) -> pygame.mixer.Sound:
    """Returnerer en pygame-lyd som en sinusbølgetone med kort udtoning."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    fade_start = int(n * 0.75)
    fade = np.linspace(1.0, 0.0, n - fade_start)
    wave[fade_start:] *= fade
    pcm = (wave * volume * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)

def _make_chord(freq1: float, freq2: float, duration: float = 0.22, volume: float = 0.50) -> pygame.mixer.Sound:
    """Laver en behagelig akkordlyd ved at blande to frekvenser i én bølge."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    fade_start = int(n * 0.70)
    fade = np.linspace(1.0, 0.0, n - fade_start)
    wave[fade_start:] *= fade
    pcm = (wave * volume * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)

# Combo-toner: hvert par får sin egen akkord (rene intervaller, ét kald afspiller begge frekvenser)
#   ('a','w')  Pegefinger+Langefinger  → D5 + A5   (ren kvint, lys og åben)
#   ('a','d')  Pegefinger+Lillefinger  → F4 + A4   (stor terts, varm)
#   ('w','s')  Langefinger+Ringefinger → E4 + B4   (ren kvint, dyb)
#   ('w','d')  Langefinger+Lillefinger → G4 + D5   (ren kvint, mid)
#   ('s','d')  Ringefinger+Lillefinger → C4 + G4   (ren kvint, lav og rund)
COMBO_FREQS = {
    ('a', 'w'): (587.3, 880.0),   # D5 + A5
    ('a', 'd'): (349.2, 440.0),   # F4 + A4
    ('w', 's'): (329.6, 493.9),   # E4 + B4
    ('w', 'd'): (392.0, 587.3),   # G4 + D5
    ('s', 'd'): (261.6, 392.0),   # C4 + G4
}

KEYS = ['w', 'a', 's', 'd']

KEY_LABEL = {
    'a': 'Pegefinger',
    'w': 'Langefinger',
    's': 'Ringefinger',
    'd': 'Lillefinger',
}

KEY_COLOR = {
    'a': (255, 210,   0),   # gul   - Pegefinger
    'w': (160,  30, 255),   # lilla - Langefinger
    's': ( 20, 200,  50),   # groen - Ringefinger
    'd': (230,  25,  25),   # roed  - Lillefinger
}

# The thumb key – sent via UDP when thumb brick is pressed
THUMB_KEY   = 'f'
THUMB_COLOR = ( 40, 120, 255)   # blaa

# All valid prompts: 4 singles + 5 combos, chosen with equal probability
ALL_PROMPTS = [
    'a', 'w', 's', 'd',
    ('a', 'w'), ('a', 'd'), ('w', 's'), ('w', 'd'), ('s', 'd'),
]

def generate_prompt_pool(n=16):
    return [random.choice(ALL_PROMPTS) for _ in range(n)]

C_BG          = (18,  20,  36)
C_TILE        = (45,  48,  70)
C_TILE_BORDER = (70,  74, 105)
C_TEXT        = (235, 235, 245)
C_DIM         = (140, 140, 165)
C_CORRECT     = ( 80, 230, 120)
C_WIN         = (255, 215,   0)
C_HIGHLIGHT   = ( 60,  80, 180)
C_HIGHLIGHT_B = (120, 150, 255)

# Module-level handles - set inside main()
screen     = None
clock      = None
ml         = None
font_big   = None
font_med   = None
font_small = None
hand_img   = None   # Handplacement image shown on intro screen
sounds       = {}   # key -> pygame.mixer.Sound (sine-wave toner, enkelt-tryk)
combo_sounds = {}   # tuple -> pygame.mixer.Sound (akkordtoner, combo-tryk)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t

def lerp_color(a, b, t):
    return tuple(int(lerp(a[i], b[i], t)) for i in range(3))

def avg_color(*colors):
    n = len(colors)
    return tuple(sum(c[i] for c in colors) // n for i in range(3))

def draw_rrect(surf, color, rect, radius=12, border=0, border_col=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_col:
        pygame.draw.rect(surf, border_col, rect, border, border_radius=radius)

def prompt_color(prompt):
    if isinstance(prompt, str):
        return KEY_COLOR[prompt]
    return avg_color(*[KEY_COLOR[k] for k in prompt])

def scale_fit(raw_surface):
    rw, rh = raw_surface.get_size()
    scale  = min(PIC_MAX_W / rw, PIC_MAX_H / rh)
    fit_w  = max(GRID_COLS, (int(rw * scale) // GRID_COLS) * GRID_COLS)
    fit_h  = max(GRID_ROWS, (int(rh * scale) // GRID_ROWS) * GRID_ROWS)
    return pygame.transform.scale(raw_surface, (fit_w, fit_h)), fit_w, fit_h

def load_images_from(directory):
    imgs = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for p in sorted(directory.glob(pat)):
            try:
                raw  = pygame.image.load(str(p)).convert()
                surf, fw, fh = scale_fit(raw)
                imgs.append((surf, p.stem, fw, fh))
            except Exception:
                pass
    return imgs

def get_picture_folders():
    entries = []
    for d in sorted(PICS_DIR.iterdir()):
        if d.is_dir():
            has_img = any(
                next(d.glob(pat), None) is not None
                for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            )
            if has_img:
                entries.append((d.name, d))
    return entries


# ─────────────────────────────────────────────
#  Input detection
# ─────────────────────────────────────────────
def combo_triggered(prompt):
    if isinstance(prompt, str):
        return ml.get_input(prompt)
    all_held = all(k in ml.current_keys for k in prompt)
    any_new  = any(k not in ml.previous_keys for k in prompt)
    return all_held and any_new


# ─────────────────────────────────────────────
#  Tile reveal animation
# ─────────────────────────────────────────────
class TileAnim:
    FRAMES = 18

    def __init__(self, tile_idx, prompt, tile_w, tile_h, pic_x, pic_y):
        self.tile   = tile_idx
        self.color  = prompt_color(prompt)
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.pic_x  = pic_x
        self.pic_y  = pic_y
        self.t      = 0.0

    @property
    def done(self):
        return self.t >= 1.0

    def update(self):
        self.t = min(1.0, self.t + 1.0 / self.FRAMES)

    def draw(self, surf):
        col    = self.tile % GRID_COLS
        row    = self.tile // GRID_COLS
        tx     = self.pic_x + col * self.tile_w
        ty     = self.pic_y + row * self.tile_h
        alpha  = int(255 * (1.0 - self.t))
        shrink = int(self.t * (self.tile_w // 2))
        rect   = pygame.Rect(tx + shrink, ty + shrink,
                             self.tile_w - 2*shrink, self.tile_h - 2*shrink)
        if rect.width > 0 and rect.height > 0:
            s = pygame.Surface((rect.width, rect.height))
            s.fill(self.color)
            s.set_alpha(alpha)
            surf.blit(s, rect)


# ─────────────────────────────────────────────
#  Game state
# ─────────────────────────────────────────────
class PictureGame:

    def __init__(self, images):
        self.images    = images
        self.pics_done = 0
        self.finished  = False
        self.anims     = []
        self.img_idx   = random.randrange(len(images))
        self._load_picture()

    def _load_picture(self):
        self.pic, self.pic_name, fw, fh = self.images[self.img_idx]
        self.pic_x  = PIC_BOX_X + (PIC_MAX_W - fw) // 2
        self.pic_y  = PIC_Y     + (PIC_MAX_H - fh) // 2
        self.tile_w = fw // GRID_COLS
        self.tile_h = fh // GRID_ROWS
        self.pic_w  = fw
        self.pic_h  = fh

        pool       = generate_prompt_pool(GRID_COLS * GRID_ROWS)
        tile_order = list(range(GRID_COLS * GRID_ROWS))
        random.shuffle(tile_order)
        self.seq       = list(zip(pool, tile_order))
        self.step      = 0
        self.covered   = set(range(GRID_COLS * GRID_ROWS))
        self.state     = "playing"
        self.fb_timer  = 0

    def _advance_picture(self):
        self.pics_done += 1
        if self.pics_done >= len(self.images):
            self.finished = True
            return
        self.img_idx = (self.img_idx + 1) % len(self.images)
        self._load_picture()

    @property
    def current_prompt(self):
        if self.step < len(self.seq):
            return self.seq[self.step][0]
        return None

    def _on_correct(self):
        prompt, tile = self.seq[self.step]
        self.covered.discard(tile)
        self.anims.append(TileAnim(
            tile, prompt,
            self.tile_w, self.tile_h, self.pic_x, self.pic_y))
        self.fb_timer = 30
        # Afspil lyd: akkordtone for kombinationer, enkelt-tone for enkelt-tryk
        if isinstance(prompt, tuple) and prompt in combo_sounds:
            combo_sounds[prompt].play()
        elif isinstance(prompt, str) and prompt in sounds:
            sounds[prompt].play()
        self.step += 1
        if not self.covered:
            self.state = "win"

    def check_input(self):
        if self.state == "playing":
            prompt = self.current_prompt
            if prompt is None:
                return
            if combo_triggered(prompt):
                self._on_correct()
        elif self.state == "win":
            # Advance to next picture when thumb key is pressed
            if ml.was_pressed(THUMB_KEY):
                self._advance_picture()

    def update(self):
        if self.fb_timer > 0:
            self.fb_timer -= 1
        for a in self.anims:
            a.update()
        self.anims = [a for a in self.anims if not a.done]

    def draw(self, surf):
        surf.fill(C_BG)

        t = font_med.render("Billedspil", True, (180, 200, 255))
        surf.blit(t, (W // 2 - t.get_width() // 2, _s(18)))

        pics_left = max(0, len(self.images) - self.pics_done)
        sc = font_small.render("Billeder tilbage: {}".format(pics_left), True, C_DIM)
        surf.blit(sc, (W // 2 - sc.get_width() // 2, _s(58)))

        surf.blit(self.pic, (self.pic_x, self.pic_y))

        for tile in self.covered:
            col = tile % GRID_COLS
            row = tile // GRID_COLS
            tx  = self.pic_x + col * self.tile_w
            ty  = self.pic_y + row * self.tile_h
            draw_rrect(surf, C_TILE,
                       pygame.Rect(tx+1, ty+1, self.tile_w-2, self.tile_h-2), radius=4)
            draw_rrect(surf, C_TILE_BORDER,
                       pygame.Rect(tx+1, ty+1, self.tile_w-2, self.tile_h-2),
                       radius=4, border=2, border_col=C_TILE_BORDER)

        for a in self.anims:
            a.draw(surf)

        bar_y = PIC_Y + PIC_MAX_H + _s(8)
        pct   = 1.0 - len(self.covered) / (GRID_COLS * GRID_ROWS)
        pygame.draw.rect(surf, C_TILE,
                         (PIC_BOX_X, bar_y, PIC_MAX_W, _s(8)), border_radius=_s(4))
        pygame.draw.rect(surf, C_CORRECT,
                         (PIC_BOX_X, bar_y, int(PIC_MAX_W * pct), _s(8)), border_radius=_s(4))

        prompt_y = bar_y + _s(8) + _s(24)
        if self.state == "win":
            self._draw_win(surf, prompt_y)
        else:
            self._draw_prompt(surf, prompt_y)

    def _draw_prompt(self, surf, y):
        prompt = self.current_prompt
        if prompt is None:
            return
        is_combo = isinstance(prompt, tuple)

        if self.fb_timer > 0:
            msg = font_big.render("Korrekt!", True, C_CORRECT)
            s   = pygame.Surface(msg.get_size(), pygame.SRCALPHA)
            s.blit(msg, (0, 0))
            s.set_alpha(int(255 * self.fb_timer / 30))
            surf.blit(s, (W // 2 - msg.get_width() // 2, y))
            y += _s(54)
        else:
            lbl_txt = "Tryk begge på en gang:" if is_combo else "Tryk med denne finger:"
            lbl = font_med.render(lbl_txt, True, C_TEXT)
            surf.blit(lbl, (W // 2 - lbl.get_width() // 2, y))
            y += _s(46)

        if is_combo:
            self._draw_combo_buttons(surf, prompt, y)
        else:
            self._draw_single_button(surf, prompt, y)
        y += _s(112)

        n  = len(self.covered)
        tl = font_small.render("{} felt tilbage".format(n), True, C_DIM)
        surf.blit(tl, (W // 2 - tl.get_width() // 2, y))

    def _draw_single_button(self, surf, key, y):
        btn = pygame.Rect(W // 2 - _s(140), y, _s(280), _s(90))
        pygame.draw.rect(surf, KEY_COLOR[key], btn, border_radius=_s(18))
        pygame.draw.rect(surf, (255, 255, 255), btn, 3, border_radius=_s(18))
        lbl = font_med.render(KEY_LABEL[key], True, (20, 20, 20))
        surf.blit(lbl, (btn.centerx - lbl.get_width()//2,
                        btn.centery - lbl.get_height()//2))

    def _draw_combo_buttons(self, surf, combo, y):
        BTN_W, BTN_H = _s(190), _s(90)
        GAP     = _s(30)
        start_x = W // 2 - (BTN_W * 2 + GAP) // 2
        for i, key in enumerate(combo):
            bx  = start_x + i * (BTN_W + GAP)
            btn = pygame.Rect(bx, y, BTN_W, BTN_H)
            pygame.draw.rect(surf, KEY_COLOR[key], btn, border_radius=_s(16))
            pygame.draw.rect(surf, (255, 255, 255), btn, 3, border_radius=_s(16))
            lbl = font_small.render(KEY_LABEL[key], True, (20, 20, 20))
            surf.blit(lbl, (btn.centerx - lbl.get_width()//2,
                            btn.centery - lbl.get_height()//2))
        plus   = font_big.render("+", True, (255, 255, 255))
        plus_x = start_x + BTN_W + GAP//2 - plus.get_width()//2
        surf.blit(plus, (plus_x, y + BTN_H//2 - plus.get_height()//2))

    def _draw_win(self, surf, y):
        txt = font_big.render("Billede afsløret!", True, C_WIN)
        surf.blit(txt, (W//2 - txt.get_width()//2, y + _s(10)))

        lbl = font_med.render("Tryk tommelfingeren for at fortsætte:", True, C_TEXT)
        surf.blit(lbl, (W//2 - lbl.get_width()//2, y + _s(70)))

        btn = pygame.Rect(W//2 - _s(140), y + _s(112), _s(280), _s(80))
        pygame.draw.rect(surf, THUMB_COLOR, btn, border_radius=_s(18))
        pygame.draw.rect(surf, (255, 255, 255), btn, 3, border_radius=_s(18))
        blbl = font_med.render("Tommelfinger", True, (20, 20, 20))
        surf.blit(blbl, (btn.centerx - blbl.get_width()//2,
                         btn.centery - blbl.get_height()//2))


# ─────────────────────────────────────────────
#  Intro screen
# ─────────────────────────────────────────────
def draw_intro(tick):
    screen.fill(C_BG)
    title = font_big.render("Billedspil!", True, (180, 200, 255))
    screen.blit(title, (W//2 - title.get_width()//2, _s(80)))
    pygame.draw.rect(screen, (100, 120, 220),
                     (W//2-_s(120), _s(80)+title.get_height()+_s(6), _s(240), _s(3)),
                     border_radius=_s(2))

    lines = [
        "Afslør et skjult billede ved at trykke",
        "på de rigtige brikker!",
        "",
        "Du skal i dette spil lave en serie af tryk",
        "for at afsløre et billede.",
        "I spillet vil du til tidspunkter skulle trykke",
        "på to brikker på samme tid.",
        "Stil din hånd i udgangsstillingen som vist nedenfor,",
        "Husk at holde håndroden på den multifarvede glatte overflade."
    ]
    y = _s(155)
    for line in lines:
        if line == "":
            y += _s(12)
            continue
        txt = font_small.render(line, True, C_TEXT)
        screen.blit(txt, (W//2 - txt.get_width()//2, y))
        y += _s(28)

    # Hand placement image
    if hand_img is not None:
        img_x = W // 2 - hand_img.get_width() // 2
        screen.blit(hand_img, (img_x, y + _s(16)))

    pulse     = abs((tick % 60) / 30 - 1)
    enter_col = lerp_color((120, 140, 200), (220, 235, 255), pulse)
    enter_txt = font_med.render("Tryk  Enter  for  at  fortsætte", True, enter_col)
    screen.blit(enter_txt, (W//2 - enter_txt.get_width()//2, H - _s(60)))

    replay = font_small.render("R = gentag tale", True, (90, 100, 130))
    screen.blit(replay, (W//2 - replay.get_width()//2, H - _s(30)))


# ─────────────────────────────────────────────
#  Folder selector screen
# ─────────────────────────────────────────────
def draw_folder_select(folders, selected_idx, tick):
    screen.fill(C_BG)
    title = font_big.render("Vælg en kategori", True, (180, 200, 255))
    screen.blit(title, (W//2 - title.get_width()//2, _s(60)))
    pygame.draw.rect(screen, (100, 120, 220),
                     (W//2-_s(140), _s(60)+title.get_height()+_s(6), _s(280), _s(3)), border_radius=_s(2))

    sub = font_small.render("Hvilken billedmappe vil du bruge?", True, C_DIM)
    screen.blit(sub, (W//2 - sub.get_width()//2, _s(130)))

    BTN_H, BTN_W, GAP = _s(70), _s(460), _s(18)
    for i, (label, _) in enumerate(folders):
        bx   = W//2 - BTN_W//2
        by   = _s(180) + i * (BTN_H + GAP)
        rect = pygame.Rect(bx, by, BTN_W, BTN_H)
        if i == selected_idx:
            pulse   = abs((tick % 60) / 30 - 1)
            bg_col  = lerp_color(C_HIGHLIGHT, C_HIGHLIGHT_B, pulse * 0.5)
            bdr_col = lerp_color((140, 170, 255), (200, 220, 255), pulse)
            draw_rrect(screen, bg_col, rect, radius=_s(14))
            draw_rrect(screen, bdr_col, rect, radius=_s(14), border=3, border_col=bdr_col)
            txt_col = (255, 255, 255)
        else:
            draw_rrect(screen, C_TILE, rect, radius=_s(14))
            draw_rrect(screen, C_TILE_BORDER, rect, radius=_s(14), border=2, border_col=C_TILE_BORDER)
            txt_col = C_DIM
        lbl = font_med.render(label, True, txt_col)
        screen.blit(lbl, (rect.centerx - lbl.get_width()//2,
                          rect.centery - lbl.get_height()//2))

    nav = font_small.render("Pile op/ned for at vælge    Enter for at bekræfte", True, C_DIM)
    screen.blit(nav, (W//2 - nav.get_width()//2, H-_s(60)))

    replay = font_small.render("R = gentag tale", True, (90, 100, 130))
    screen.blit(replay, (W//2 - replay.get_width()//2, H - _s(30)))


# ─────────────────────────────────────────────
#  Congratulations screen
# ─────────────────────────────────────────────
def draw_congrats(tick):
    screen.fill(C_BG)

    title = font_big.render("Tillykke!", True, C_WIN)
    screen.blit(title, (W//2 - title.get_width()//2, _s(120)))
    pygame.draw.rect(screen, (180, 160, 0),
                     (W//2 - _s(100), _s(120) + title.get_height() + _s(6), _s(200), _s(3)), border_radius=_s(2))

    lines = [
        "Du har klaret alle billeder i mappen!",
        "",
        "Tryk med tommelfingeren for at",
        "vende tilbage til billedmapperne.",
    ]
    y = _s(230)
    for line in lines:
        if line == "":
            y += _s(14)
            continue
        txt = font_med.render(line, True, C_TEXT)
        screen.blit(txt, (W//2 - txt.get_width()//2, y))
        y += _s(42)

    # Pulsing thumb button
    pulse   = abs((tick % 60) / 30 - 1)
    btn_col = lerp_color(THUMB_COLOR, (120, 180, 255), pulse * 0.4)
    btn     = pygame.Rect(W//2 - _s(160), y + _s(30), _s(320), _s(80))
    pygame.draw.rect(screen, btn_col, btn, border_radius=_s(18))
    pygame.draw.rect(screen, (255, 255, 255), btn, 3, border_radius=_s(18))
    blbl = font_med.render("Tommelfinger", True, (20, 20, 20))
    screen.blit(blbl, (btn.centerx - blbl.get_width()//2,
                       btn.centery - blbl.get_height()//2))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    global screen, clock, ml, font_big, font_med, font_small, sounds, combo_sounds

    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    # Centre the window vertically on the screen
    _info    = pygame.display.Info()
    win_x    = HAND_WIN_W
    win_y    = max(0, (_info.current_h - H) // 2)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "{},{}".format(win_x, win_y)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Billedspil")
    clock  = pygame.time.Clock()
    ml     = MLControlBridge(press_mode="single")

    font_big   = pygame.font.SysFont("arialrounded", _s(46)) or pygame.font.SysFont(None, _s(46))
    font_med   = pygame.font.SysFont("arialrounded", _s(30)) or pygame.font.SysFont(None, _s(30))
    font_small = pygame.font.SysFont("arialrounded", _s(22)) or pygame.font.SysFont(None, _s(22))

    # Byg sine-bølge toner (én per finger-tast)
    sounds = {k: _make_tone(freq) for k, freq in KEY_FREQS.items()}
    # Byg akkordtoner for de 5 kombinationer
    combo_sounds = {combo: _make_chord(f1, f2) for combo, (f1, f2) in COMBO_FREQS.items()}

    # Load hand placement image for intro screen
    global hand_img
    _img_path = Path(__file__).resolve().parent / "Handplacement.png"
    if _img_path.exists():
        _raw = pygame.image.load(str(_img_path)).convert_alpha()
        _rw, _rh = _raw.get_size()
        _max_w, _max_h = W - _s(80), H - _s(340)   # leave room for text + enter prompt
        _scale = min(_max_w / _rw, _max_h / _rh, 1.0) * 0.5
        hand_img = pygame.transform.smoothscale(
            _raw, (int(_rw * _scale), int(_rh * _scale)))

    state        = "intro"
    tick         = 0
    running      = True
    game         = None
    folders      = get_picture_folders()
    selected_idx = 0
    keyboard_held = set()
    game_start_time  = None   # set when gameplay begins
    reminder_played  = False  # True once the 2-min reminder has fired

    # Auto-play intro speech on startup
    _play_speech(SPEECH_P1)

    while running:
        ml.pump()
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif state == "intro" and event.key == pygame.K_r:
                    _play_speech(SPEECH_P1)

                elif state == "intro" and event.key == pygame.K_RETURN:
                    if folders:
                        state = "folder"
                        _play_speech(SPEECH_P2)
                    else:
                        images = load_images_from(PICS_DIR)
                        if images:
                            game  = PictureGame(images)
                            state = "playing"
                            game_start_time = time.time()
                            reminder_played = False

                elif state == "congrats" and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    ml.previous_keys = set(ml.current_keys)
                    ml.current_keys  = set(ml.current_keys) | {THUMB_KEY}

                elif state == "folder" and event.key == pygame.K_r:
                    _play_speech(SPEECH_P2)

                elif state == "folder":
                    if event.key == pygame.K_UP:
                        selected_idx = (selected_idx - 1) % len(folders)
                    elif event.key == pygame.K_DOWN:
                        selected_idx = (selected_idx + 1) % len(folders)
                    elif event.key == pygame.K_RETURN:
                        _, chosen_dir = folders[selected_idx]
                        images = load_images_from(chosen_dir)
                        if images:
                            game  = PictureGame(images)
                            state = "playing"
                            game_start_time = time.time()
                            reminder_played = False

                elif state == "playing":
                    key_map = {
                        pygame.K_w: 'w', pygame.K_a: 'a',
                        pygame.K_s: 's', pygame.K_d: 'd',
                    }
                    k = key_map.get(event.key)
                    if k:
                        keyboard_held.add(k)
                        ml.previous_keys = set(ml.current_keys)
                        ml.current_keys  = set(keyboard_held)
                    # Space / Enter acts as thumb key (for testing without hardware)
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        ml.previous_keys = set(ml.current_keys)
                        ml.current_keys  = set(ml.current_keys) | {THUMB_KEY}

            elif event.type == pygame.KEYUP and state == "playing":
                key_map = {
                    pygame.K_w: 'w', pygame.K_a: 'a',
                    pygame.K_s: 's', pygame.K_d: 'd',
                }
                k = key_map.get(event.key)
                if k:
                    keyboard_held.discard(k)
                    ml.previous_keys = set(ml.current_keys)
                    ml.current_keys  = set(keyboard_held)
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    ml.current_keys.discard(THUMB_KEY)

        if state == "intro":
            draw_intro(tick)
        elif state == "folder":
            draw_folder_select(folders, selected_idx, tick)
        elif state == "congrats":
            draw_congrats(tick)
            if ml.was_pressed(THUMB_KEY):
                state        = "folder"
                selected_idx = 0
        elif state == "playing" and game:
            game.check_input()
            game.update()
            if game.finished:
                game  = None
                state = "congrats"
            else:
                game.draw(screen)
                # Play the hand-reminder after 2 minutes of gameplay
                if (not reminder_played and game_start_time is not None
                        and time.time() - game_start_time >= REMINDER_SECONDS):
                    _play_speech(SPEECH_REMINDER)
                    reminder_played = True

        pygame.display.flip()
        clock.tick(FPS)

    ml.close()
    pygame.quit()


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parent / "2_Picture_Game_error.log"
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        with open(log_path, "w") as f:
            f.write(tb)
        try:
            f2 = pygame.font.SysFont(None, 24)
            screen.fill((20, 0, 0))
            for i, line in enumerate(tb.splitlines()[-22:]):
                t = f2.render(line[:90], True, (255, 100, 100))
                screen.blit(t, (8, 8 + i*26))
            hint = f2.render("Tryk en tast for at afslutte", True, (180, 180, 180))
            screen.blit(hint, (8, H-30))
            pygame.display.flip()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                        waiting = False
        except Exception:
            pass
        raise
