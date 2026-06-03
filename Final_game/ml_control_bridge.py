import json
import socket

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 50555


class MLControlBridge:
    """
    Bridge that receives hand/ML key events over UDP and exposes them to games.

    Press modes
    -----------
    "single"     – get_input() fires once per press; the finger must be fully
                   removed before the same key can trigger again.
    "continuous" – get_input() returns True every frame the key is held.

    Call toggle_mode() (or press Tab in any game) to switch between modes.
    The current mode is always shown on-screen inside each game.
    """

    def __init__(self, host=CONTROL_HOST, port=CONTROL_PORT, press_mode="single"):
        self.current_keys = set()
        self.previous_keys = set()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        # "single" or "continuous"
        self.press_mode = press_mode

    # ------------------------------------------------------------------
    # Core update – call once per frame at the top of the game loop
    # ------------------------------------------------------------------
    def pump(self):
        self.previous_keys = set(self.current_keys)
        latest_keys = None

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
            except Exception:
                continue

        if latest_keys is not None:
            self.current_keys = latest_keys

    # ------------------------------------------------------------------
    # Press-mode helpers (used internally and available to game code)
    # ------------------------------------------------------------------
    def is_pressed(self, *keys):
        """True every frame the key is held (continuous)."""
        return any(str(k).lower() in self.current_keys for k in keys)

    def was_pressed(self, *keys):
        """True only on the first frame a key goes from unpressed → pressed (single)."""
        for key in keys:
            key_name = str(key).lower()
            if key_name in self.current_keys and key_name not in self.previous_keys:
                return True
        return False

    # ------------------------------------------------------------------
    # Unified input query – respects the current press_mode
    # ------------------------------------------------------------------
    def get_input(self, *keys):
        """
        The single function games should call instead of is_pressed/was_pressed.
        Behaviour is determined by self.press_mode:
          'single'     → equivalent to was_pressed()
          'continuous' → equivalent to is_pressed()
        """
        if self.press_mode == "continuous":
            return self.is_pressed(*keys)
        return self.was_pressed(*keys)

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------
    def toggle_mode(self):
        """Flip between 'single' and 'continuous' press modes."""
        self.press_mode = "continuous" if self.press_mode == "single" else "single"

    def set_mode(self, mode):
        """Explicitly set press mode: 'single' or 'continuous'."""
        if mode in ("single", "continuous"):
            self.press_mode = mode

    # ------------------------------------------------------------------
    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass