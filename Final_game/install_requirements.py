import subprocess
import sys
from pathlib import Path

requirements = Path(__file__).resolve().parent / "requirements.txt"
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)
input("\nDone! Press Enter to close...")
