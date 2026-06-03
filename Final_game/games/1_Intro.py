"""
0_Intro.py  –  Håndrehabilitering – Introduktionsspil
======================================================
Lærer patienten at interagere med LEGO-plader ved hjælp af fingrene.

Faser
-----
1. INTRO      – To instruktionsskærme der forklarer frem-og-tilbage-bevægelsen.
2. OPVARMNING – Rør ved hver plade to gange (én gang på, én gang af og tilbage)
                så patienten oplever frem-og-tilbage-bevægelsen.
3. ØVELSE     – 15 runder; kræver gennemsnitlig nøjagtighed ≥ 30 % for at bestå.
                Nøjagtighed = hvor centreret fingerspidsen lander inden i bounding boxen.
                En unik lyd afspilles ved hvert tryk med nøjagtighed > 30 %.
4. RESULTAT   – Viser nøjagtighed, bestået/ikke bestået og opmuntring.

Tastebindinger (fra Main_Demo)
------------------------------
  d → Lillefinger   s → Ringfinger   w → Langfinger   a → Pegefinger
  (venstre til højre på tværs af de fire LEGO-plader)
"""

import math
import random
import os
import sys
from pathlib import Path

import numpy as np
import pygame
import ctypes as _ctypes

import json
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml_control_bridge import MLControlBridge


class AccuracyBridge(MLControlBridge):
    """
    Extends MLControlBridge to also capture per-key accuracy scores sent by
    Main_Demo in the UDP payload under the "accuracy" key.

    accuracy score: 0.0 = finger at edge of box, 1.0 = perfect centre.
    Scores are only present when Main_Demo includes them (i.e. a touch is active).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_accuracy: dict = {}   # key -> float (0.0–1.0)

    def pump(self):
        self.previous_keys = set(self.current_keys)
        latest_keys     = None
        latest_accuracy = {}

        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except Exception:
                break

            try:
                payload = json.loads(data.decode("utf-8"))
                keys = payload.get("keys", [])
                latest_keys = {str(k).lower() for k in keys if isinstance(k, str)}
                if "accuracy" in payload:
                    latest_accuracy = {str(k).lower(): float(v)
                                       for k, v in payload["accuracy"].items()}
            except Exception:
                continue

        if latest_keys is not None:
            self.current_keys   = latest_keys
            self.last_accuracy  = latest_accuracy

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
# ── Screen-aware scaling ──────────────────────────────────────────────────────
# Reference layout is 1920×1080: game window 1280×720 (right two-thirds),
# live-stream window 640×480 (left third).  On any other resolution we apply a
# uniform scale factor so both windows tile identically.
try:
    _sw = _ctypes.windll.user32.GetSystemMetrics(0)   # screen width
    _sh = _ctypes.windll.user32.GetSystemMetrics(1)   # screen height
except Exception:
    _sw, _sh = 1920, 1080

SCALE = min(_sw / 1920.0, _sh / 1080.0)
W = max(640, round(1280 * SCALE))
H = max(360, round(720  * SCALE))

def _s(v: float) -> int:
    """Scale a reference-resolution pixel value to the current screen size."""
    return max(1, round(v * SCALE))
# ─────────────────────────────────────────────────────────────────────────────

FPS  = 60
DEFAULT_PRESS_MODE = "single"

# Lyd-filer (tale)
AUDIO_DIR         = Path(__file__).resolve().parent / "audio"
SPEECH_INTRO_P1   = AUDIO_DIR / "tts_1_intro_first_page.mp3"
SPEECH_INTRO_P2   = AUDIO_DIR / "tts_1_intro_second_page.mp3"
SPEECH_INTRO_P3   = AUDIO_DIR / "tts_1_intro_third_page.mp3"
SPEECH_INTRO_P4   = AUDIO_DIR / "tts_1_intro_fourth_page.mp3"
SPEECH_HIT_SADAN  = AUDIO_DIR / "tts_sådan.mp3"
SPEECH_HIT_GODT   = AUDIO_DIR / "tts_godt_gået.mp3"

# Pladerækkefølge (venstre → højre på skærmen)
FINGER_ORDER = ["Pegefinger", "Langefinger", "Ringfinger", "Lillefinger"]
KEY_TO_FINGER = {"d": "Lillefinger", "s": "Ringfinger", "w": "Langefinger", "a": "Pegefinger"}
FINGER_TO_KEY = {v: k for k, v in KEY_TO_FINGER.items()}
ALL_KEYS = ["d", "s", "a", "w"]

# Farver per finger
FINGER_COLORS = {
    "Lillefinger": (210,  45,  45),   # rød
    "Ringfinger":  ( 55, 185,  75),   # grøn
    "Langefinger": (145,  55, 210),   # lilla
    "Pegefinger":  (230, 210,  45),   # gul
}

# Musikalske frekvenser (lette at skelne fra hinanden)
FINGER_FREQS = {
    "Lillefinger": 392,   # G4
    "Ringfinger":  523,   # C5
    "Langefinger": 659,   # E5
    "Pegefinger":  784,   # G5
}

SAMPLE_RATE = 44100

# ──────────────────────────────────────────────────────────────────────────────
# Lydhjælpere
# ──────────────────────────────────────────────────────────────────────────────

def _make_tone(freq: float, duration: float = 0.18, volume: float = 0.55) -> pygame.mixer.Sound:
    """Returnerer en pygame-lyd som en sinusbølgetone med kort udtoning."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)

    # Jævn udtoning over de sidste 25 %
    fade_start = int(n * 0.75)
    fade = np.linspace(1.0, 0.0, n - fade_start)
    wave[fade_start:] *= fade

    pcm = (wave * volume * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])   # 2-kanal
    return pygame.sndarray.make_sound(stereo)




# ──────────────────────────────────────────────────────────────────────────────
# Fasekonstanter
# ──────────────────────────────────────────────────────────────────────────────
PHASE_INTRO    = 0
PHASE_WARMUP   = 1
PHASE_EXERCISE = 2
PHASE_RESULTS  = 3

EXERCISE_TOTAL     = 15
PASS_THRESHOLD     = 0.50     # 50 %
WARMUP_TOUCHES     = 2        # hver finger skal røres dette antal gange



# ──────────────────────────────────────────────────────────────────────────────
# Tegnhjælpere
# ──────────────────────────────────────────────────────────────────────────────

def text(surf: pygame.Surface, msg: str, font: pygame.font.Font,
         color, cx: int, y: int) -> None:
    """Tegner centreret tekst."""
    s = font.render(msg, True, color)
    surf.blit(s, (cx - s.get_width() // 2, y))


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ──────────────────────────────────────────────────────────────────────────────
# Hoved-spilklasse
# ──────────────────────────────────────────────────────────────────────────────

class IntroGame:

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

        # Position window as far right as possible, vertically centred
        info = pygame.display.Info()
        win_x = max(0, info.current_w - W)
        win_y = max(0, (info.current_h - H) // 2)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{win_x},{win_y}'

        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Håndrehabilitering – Introduktion")
        self.clock = pygame.time.Clock()
        self.ml    = AccuracyBridge(press_mode=DEFAULT_PRESS_MODE)

        # Skrifttyper – skaleret med skærmstørrelsen
        self.f_title  = pygame.font.SysFont("Arial", _s(54), bold=True)
        self.f_large  = pygame.font.SysFont("Arial", _s(40), bold=True)
        self.f_med    = pygame.font.SysFont("Arial", _s(30))
        self.f_small  = pygame.font.SysFont("Arial", _s(22))
        self.f_key    = pygame.font.SysFont("Courier New", _s(20), bold=True)

        # Lyde
        self.sounds = {f: _make_tone(freq) for f, freq in FINGER_FREQS.items()}

        # Håndplaceringsbillede (vises nederst til venstre på første introside)
        img_path = Path(__file__).resolve().parent / "Handplacement.png"
        try:
            raw = pygame.image.load(str(img_path)).convert_alpha()
            # Skaler så billedet er max 280px bredt / 220px højt (ved 1920×1080)
            max_w, max_h = _s(280), _s(220)
            scale = min(max_w / raw.get_width(), max_h / raw.get_height())
            new_size = (int(raw.get_width() * scale), int(raw.get_height() * scale))
            self.hand_img = pygame.transform.smoothscale(raw, new_size)
        except Exception:
            self.hand_img = None   # ingen fil = intet billede

        self._reset_state()

        # Afspil velkomsttale automatisk ved opstart
        self._play_speech(SPEECH_INTRO_P1)

    # ── tale ──────────────────────────────────────────────────────────────────

    def _play_speech(self, path: Path):
        """Afspiller en MP3-talefil via music-kanalen. Fejler lydløst hvis filen mangler."""
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
        except Exception:
            pass   # ingen fil = ingen lyd, spillet fortsætter normalt

    # ── tilstand ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.phase       = PHASE_INTRO
        self.intro_page  = 0
        self.anim_t      = 0.0

        # opvarmning
        self.wu_idx      = 0                              # nuværende fingerindeks
        self.wu_touches  = {f: 0 for f in FINGER_ORDER}
        self.wu_done_at  = None                           # anim_t når alt er færdigt

        # øvelse
        self.ex_round       = 0          # antal godkendte tryk (accuracy ≥ tærskel)
        self.ex_press_count = 0          # totalt antal godkendte tryk (til "godt gået"-kadence)
        self.ex_live_acc    = {}         # key -> float: live accuracy fra seneste pump()
        self.ex_target      = None       # fremhævet plade (vejledende, ikke krav)
        self.ex_hit_timer   = 0.0        # kort flash-timer efter godkendt tryk
        self.ex_hit_label   = "Sådan!"  # næste flash-tekst (skifter mellem to sætninger)
        self.ex_hit_display = ""        # flash-tekst der vises lige nu
        self.ex_scored_keys = set()      # taster der allerede har scoret i dette tryk
        self.ex_done        = False
        self._pick_target()

    def _pick_target(self):
        """Vælger et nyt tilfældigt øvelsesmål (undgår at gentage samme finger to gange)."""
        others = [f for f in FINGER_ORDER if f != self.ex_target]
        self.ex_target = random.choice(others if others else FINGER_ORDER)

    # ── pladelayout ───────────────────────────────────────────────────────────

    def _plate_rects(self) -> dict:
        pw, ph     = _s(170), _s(115)
        gap        = _s(36)
        total_w    = len(FINGER_ORDER) * pw + (len(FINGER_ORDER) - 1) * gap
        x0         = (W - total_w) // 2
        y0         = H // 2 + _s(40)
        return {
            f: pygame.Rect(x0 + i * (pw + gap), y0, pw, ph)
            for i, f in enumerate(FINGER_ORDER)
        }

    # ── fælles plade-renderer ─────────────────────────────────────────────────

    def _draw_plates(self, highlight: str = None, glow_set: set = None,
                     show_warmup_count: bool = False):
        rects = self._plate_rects()
        t = self.anim_t

        for finger, rect in rects.items():
            base = FINGER_COLORS[finger]
            is_hl   = (finger == highlight)
            is_glow = glow_set and finger in glow_set

            if is_hl:
                pulse = 0.55 + 0.45 * math.sin(t * 5.0)
                face  = lerp_color(base, (255, 255, 255), pulse * 0.45)
                border_col = (255, 255, 255)
                bw = 4
            elif is_glow:
                face       = lerp_color(base, (255, 255, 255), 0.35)
                border_col = (200, 255, 200)
                bw = 3
            else:
                face       = lerp_color(base, (0, 0, 0), 0.30)
                border_col = (80, 80, 90)
                bw = 2

            pygame.draw.rect(self.screen, face, rect, border_radius=_s(14))
            pygame.draw.rect(self.screen, border_col, rect, bw, border_radius=_s(14))

            # Knopnitter
            for dx, dy in [(-_s(30), -_s(18)), (0, -_s(18)), (_s(30), -_s(18)),
                           (-_s(30),  _s(18)), (0,  _s(18)), (_s(30),  _s(18))]:
                cx, cy = rect.centerx + dx, rect.centery + dy
                pygame.draw.circle(self.screen, lerp_color(face, (0,0,0), 0.15), (cx, cy), _s(7))

            # Fingernavn
            lbl = self.f_small.render(finger, True, (255, 255, 255))
            self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                                   rect.centery - _s(16)))

            # Opvarmningstæller
            if show_warmup_count:
                count   = self.wu_touches[finger]
                needed  = WARMUP_TOUCHES
                done    = count >= needed
                c_color = (100, 255, 140) if done else (220, 220, 220)
                c_surf  = self.f_small.render(f"{'Klaret!' if done else f'{count}/{needed}'}", True, c_color)
                self.screen.blit(c_surf, (rect.centerx - c_surf.get_width() // 2,
                                          rect.bottom + 8))

    # ── live nøjagtighed under pladerne ──────────────────────────────────────

    def _draw_accuracy_bars_under_plates(self, target: str = None):
        """
        Tegner en vandret bjælke under målet-pladen (target).
        Bjælken er altid synlig under target-pladen og viser live nøjagtighed
        når fingeren hviler på pladen, ellers er den tom (grå baggrund).
        Farve: grøn når ≥ tærskel, orange når under.
        En lodret hvid streg markerer tærsklen (30 %).
        """
        if target is None:
            return

        rects = self._plate_rects()
        rect = rects.get(target)
        if rect is None:
            return

        key    = FINGER_TO_KEY[target]
        bar_h  = _s(14)
        bar_gap = _s(10)   # afstand fra pladens bund til bjælkens top

        bx = rect.x
        by = rect.bottom + bar_gap
        bw = rect.width

        # Baggrund (altid synlig)
        pygame.draw.rect(self.screen, (45, 48, 65),
                         pygame.Rect(bx, by, bw, bar_h), border_radius=_s(6))

        # Fyld kun når fingeren hviler på pladen og vi har accuracy-data.
        # Bjælken går fra 0 → fuld ved tærsklen (30 %), så fuld bjælke = godkendt tryk.
        if key in self.ml.current_keys and key in self.ml.last_accuracy:
            acc    = self.ml.last_accuracy[key]
            ratio  = min(max(acc, 0.0), PASS_THRESHOLD) / PASS_THRESHOLD
            fill_w = int(bw * ratio)
            if fill_w > 0:
                col = (80, 220, 100)
                pygame.draw.rect(self.screen, col,
                                 pygame.Rect(bx, by, fill_w, bar_h), border_radius=_s(6))

    # ── fremgangsbjælke ───────────────────────────────────────────────────────

    def _draw_bar(self, x: int, y: int, w: int, h: int,
                  value: float, threshold: float = None, label: str = ""):
        bg = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (45, 48, 65), bg, border_radius=_s(8))
        fill_w = int(w * min(max(value, 0.0), 1.0))
        if fill_w > 0:
            col = (75, 200, 100) if (threshold is None or value >= threshold) else (210, 110, 50)
            pygame.draw.rect(self.screen, col,
                             pygame.Rect(x, y, fill_w, h), border_radius=_s(8))
        if threshold is not None:
            mx = x + int(w * threshold)
            pygame.draw.line(self.screen, (255, 255, 255), (mx, y - _s(5)), (mx, y + h + _s(5)), 2)
            thr_s = self.f_small.render(f"{threshold:.0%}", True, (255, 250, 200))
            self.screen.blit(thr_s, (mx - thr_s.get_width() // 2, y - _s(26)))
        if label:
            ls = self.f_small.render(label, True, (190, 200, 220))
            self.screen.blit(ls, (x + w // 2 - ls.get_width() // 2, y + h + _s(6)))

    # ══════════════════════════════════════════════════════════════════════════
    # Fase-renderere
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_intro(self):
        self.screen.fill((18, 22, 38))

        if self.intro_page == 0:
            # ── Velkomstside ──
            text(self.screen, "Velkommen til introspillet!",
                 self.f_title, (255, 255, 255), W // 2, _s(100))

            lines = [
                "Dette er en guide til at lære dig at spille med LEGO-brættet.",
                "Start med at stille din hånd i udgangspositionen, som vises nede til venstre.",
                "Det er vigtigt at have håndroden på den multifarvede glatte overflade nederst på brættet.",
                "Det gør, at du fokuserer på at strække dine fingre.",
                "For at lave et tryk, skal du strække fingeren ud og ramme LEGO brikken.",
                "Efter et tryk, skal du trække fingeren tilbage til hånden igen.",
                "Det er alt!"
            ]
            y = _s(230)
            for line in lines:
                surf = self.f_med.render(line, True, (200, 210, 235))
                self.screen.blit(surf, (W // 2 - surf.get_width() // 2, y))
                y += _s(52)

            # Blinkende fortsæt-prompt
            alpha = int(180 + 75 * math.sin(self.anim_t * 3.5))
            cont = self.f_med.render("► Tryk Enter for at fortsætte", True,
                                     (100, min(255, alpha), 120))
            self.screen.blit(cont, (W // 2 - cont.get_width() // 2, H - _s(70)))
            replay = self.f_small.render("R = gentag tale", True, (90, 100, 120))
            self.screen.blit(replay, (W // 2 - replay.get_width() // 2, H - _s(35)))

            # Håndplaceringsbillede – nederst til venstre
            if self.hand_img:
                margin = _s(18)
                self.screen.blit(self.hand_img,
                                 (margin, H - self.hand_img.get_height() - margin))

        elif self.intro_page == 1:
            # ── Pladeoversigt ──
            text(self.screen, "LEGO-brikkerne",
                 self.f_title, (255, 255, 255), W // 2, _s(80))

            text(self.screen, "Her ses alle LEGO-brikkerne, som du skal ramme i øvelsen.",
                 self.f_med, (200, 215, 240), W // 2, _s(180))
            text(self.screen, "I rehabiliteringen øver vi os på at bruge pegefinger, langefinger, ringfinger og lillefinger.",
                 self.f_med, (200, 215, 240), W // 2, _s(220))
            text(self.screen, "Vi vil nu varme op ved at prøve at trykke på nogle af LEGO-brikkerne.",
                 self.f_med, (200, 215, 240), W // 2, _s(260))
            text(self.screen, "Det er helt okay hvis det er svært i starten, det vigtigste er at have det sjovt!",
                 self.f_med, (200, 215, 240), W // 2, _s(300))
            text(self.screen, "Husk at trække fingeren tilbage efter et tryk!",
                 self.f_med, (200, 215, 240), W // 2, _s(340))

            self._draw_plates()

            cont = self.f_med.render("► Tryk Enter for at starte",
                                     True, (100, 220, 130))
            self.screen.blit(cont, (W // 2 - cont.get_width() // 2, H - _s(70)))
            replay = self.f_small.render("R = gentag tale", True, (90, 100, 120))
            self.screen.blit(replay, (W // 2 - replay.get_width() // 2, H - _s(35)))

    # ──────────────────────────────────────────────────────────────────────────

    def _draw_warmup(self):
        self.screen.fill((14, 28, 22))

        finished   = self.wu_idx >= len(FINGER_ORDER)
        cur_finger = FINGER_ORDER[self.wu_idx] if not finished else None

        text(self.screen, "Opvarmning: lær at trykke",
             self.f_title, (255, 255, 255), W // 2, _s(35))

        if cur_finger:
            col = FINGER_COLORS[cur_finger]
            remaining = WARMUP_TOUCHES - self.wu_touches[cur_finger]
            text(self.screen,
                 f"Rør ved  {cur_finger}-pladen  {remaining} gang{'e' if remaining != 1 else '} til'}",
                 self.f_large, col, W // 2, _s(115))
            text(self.screen,
                 "(træk fingeren tilbage mellem hvert tryk)",
                 self.f_med, (150, 165, 190), W // 2, _s(165))
        else:
            text(self.screen, "Alle fingre er opvarmede!",
                 self.f_large, (80, 255, 150), W // 2, _s(120))
            text(self.screen, "Øvelsen starter om et øjeblik…",
                 self.f_med, (160, 175, 200), W // 2, _s(168))

        self._draw_plates(highlight=cur_finger, show_warmup_count=True)

        # Samlet fremgangsbjælke
        progress = self.wu_idx / len(FINGER_ORDER)
        self._draw_bar(W // 2 - _s(200), H - _s(70), _s(400), _s(18), progress,
                       label=f"Fingre færdige: {self.wu_idx}/{len(FINGER_ORDER)}")

    # ──────────────────────────────────────────────────────────────────────────

    def _draw_exercise(self):
        self.screen.fill((18, 18, 40))

        text(self.screen, "Øvelse", self.f_title, (255, 255, 255), W // 2, _s(28))

        # Tryk-tæller
        text(self.screen,
             f"Tryk  {self.ex_round} / {EXERCISE_TOTAL}",
             self.f_med, (170, 185, 215), W // 2, _s(105))

        # Vejledende plade-prompt
        if not self.ex_done and self.ex_target:
            col = FINGER_COLORS[self.ex_target]
            pre  = self.f_large.render("Prøv at ramme midten af  ", True, (200, 215, 240))
            name = self.f_large.render(self.ex_target, True, col)
            post = self.f_large.render("-pladen", True, (200, 215, 240))
            total_w = pre.get_width() + name.get_width() + post.get_width()
            px = W // 2 - total_w // 2
            py = _s(148)
            self.screen.blit(pre,  (px, py))
            self.screen.blit(name, (px + pre.get_width(), py))
            self.screen.blit(post, (px + pre.get_width() + name.get_width(), py))

        # Kort "HIT!"-flash efter godkendt tryk
        if self.ex_hit_timer > 0:
            fade = min(1.0, self.ex_hit_timer / 0.45)
            hit_col = (60, min(255, int(220 * fade + 35)), 80)
            hit_surf = self.f_large.render(f"{self.ex_hit_display}", True, hit_col)
            self.screen.blit(hit_surf, (W // 2 - hit_surf.get_width() // 2, _s(200)))

        # Plader — fremhæv vejledende mål og vis nøjagtighedsbjælke under målet
        hl = self.ex_target if not self.ex_done else None
        self._draw_plates(highlight=hl)
        self._draw_accuracy_bars_under_plates(target=hl)

    # ──────────────────────────────────────────────────────────────────────────

    def _draw_results(self):
        self.screen.fill((14, 18, 32))

        title_col  = (80, 255, 150)
        title_text = "Øvelse gennemført!"

        text(self.screen, title_text, self.f_title, title_col, W // 2, _s(70))

        text(self.screen,
             f"{self.ex_round} tryk gennemført!",
             self.f_large, (210, 225, 255), W // 2, _s(185))

        msg = "Godt klaret! Du er klar til terapi-spillene."
        text(self.screen, msg, self.f_med, (100, 220, 140), W // 2, _s(255))

        self._draw_plates()

        text(self.screen, "Luk vinduet eller tryk Esc for at afslutte.",
             self.f_med, (90, 100, 120), W // 2, H - _s(50))

    # ══════════════════════════════════════════════════════════════════════════
    # Hovedløkke
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        running = True

        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.anim_t += dt

            # ── systemhændelser ───────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_TAB:
                        self.ml.toggle_mode()
                    if event.key == pygame.K_RETURN and self.phase == PHASE_INTRO:
                        if self.intro_page < 1:
                            self.intro_page += 1
                            self._play_speech(SPEECH_INTRO_P2)
                        else:
                            self.phase = PHASE_WARMUP
                            self._play_speech(SPEECH_INTRO_P3)

                    if event.key == pygame.K_r and self.phase == PHASE_INTRO:
                        speech = SPEECH_INTRO_P1 if self.intro_page == 0 else SPEECH_INTRO_P2
                        self._play_speech(speech)

            self.ml.pump()

            # ══════════════════════════════════════════════════════════════
            # FASE: INTRO
            # ══════════════════════════════════════════════════════════════
            if self.phase == PHASE_INTRO:
                self._draw_intro()

            # ══════════════════════════════════════════════════════════════
            # FASE: OPVARMNING
            # ══════════════════════════════════════════════════════════════
            elif self.phase == PHASE_WARMUP:
                if self.wu_idx < len(FINGER_ORDER):
                    cur_finger   = FINGER_ORDER[self.wu_idx]
                    expected_key = FINGER_TO_KEY[cur_finger]

                    for key in ALL_KEYS:
                        if self.ml.get_input(key):
                            if key == expected_key:
                                # Korrekt finger – tæl trykket
                                self.wu_touches[cur_finger] += 1
                                self.sounds[cur_finger].play()
                                if self.wu_touches[cur_finger] >= WARMUP_TOUCHES:
                                    self.wu_idx += 1   # gå videre til næste finger
                            # (ingen straf for forkert finger under opvarmning)
                            break

                else:
                    # Alle fingre færdige – vent kort og skift fase
                    if self.wu_done_at is None:
                        self.wu_done_at = self.anim_t
                    if self.anim_t - self.wu_done_at > 1.8:
                        self.phase = PHASE_EXERCISE
                        self._play_speech(SPEECH_INTRO_P4)
                        self._pick_target()

                self._draw_warmup()

            # ══════════════════════════════════════════════════════════════
            # FASE: ØVELSE
            # ══════════════════════════════════════════════════════════════
            elif self.phase == PHASE_EXERCISE:
                # Opdater live accuracy hvert frame (til visning på pladerne)
                self.ex_live_acc = dict(self.ml.last_accuracy)

                # Nedtæl HIT-flash-timer
                if self.ex_hit_timer > 0:
                    self.ex_hit_timer -= dt

                # Ryd latch kun for taster der er LØFTET (ikke længere nede).
                # Behold scored-keys der stadig er trykket — ellers ville latchen
                # blive ryddet hvert frame mens fingeren stadig hviler på pladen.
                self.ex_scored_keys &= self.ml.current_keys

                # Et tryk registreres når fingeren ER på pladen OG accuracy ≥ tærskel.
                # Vi tjekker hvert frame (is_pressed), ikke kun på første berøring,
                # så spilleren kan centrere fingeren og opnå ≥30 % mens den stadig
                # hviler på pladen. ex_scored_keys forhindrer dobbelttælling per tryk.
                if not self.ex_done:
                    target_key = FINGER_TO_KEY[self.ex_target]
                    for key in ALL_KEYS:
                        if key != target_key:
                            continue   # kun den fremhævede plade tæller
                        if self.ml.is_pressed(key) and key not in self.ex_scored_keys:
                            acc = self.ml.last_accuracy.get(key, 0.0)
                            if acc >= PASS_THRESHOLD:
                                finger = KEY_TO_FINGER[key]
                                self.ex_scored_keys.add(key)   # latch: kun ét hit per tryk
                                self.ex_round += 1
                                self.ex_press_count += 1
                                self.sounds[finger].play()
                                # Vis "Godt gået!"/"Sådan!" og afspil tale kun hvert 3. tryk
                                if self.ex_press_count % 3 == 0:
                                    self.ex_hit_display = self.ex_hit_label
                                    speech_file = (SPEECH_HIT_SADAN if self.ex_hit_label == "Sådan!"
                                                   else SPEECH_HIT_GODT)
                                    self._play_speech(speech_file)
                                    self.ex_hit_label = ("Godt gået!" if self.ex_hit_label == "Sådan!"
                                                         else "Sådan!")
                                else:
                                    self.ex_hit_display = "Korrekt!"
                                self.ex_hit_timer = 0.45
                                if self.ex_round >= EXERCISE_TOTAL:
                                    self.ex_done = True
                                else:
                                    self._pick_target()
                                break

                # Gå til resultat straks når alle hits er registreret
                if self.ex_done and self.ex_hit_timer <= 0:
                    self.phase = PHASE_RESULTS

                self._draw_exercise()

            # ══════════════════════════════════════════════════════════════
            # FASE: RESULTAT
            # ══════════════════════════════════════════════════════════════
            elif self.phase == PHASE_RESULTS:
                self._draw_results()

            pygame.display.flip()

        # Cleanup
        self.ml.close()
        pygame.quit()
        sys.exit()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    game = IntroGame()
    game.run()
