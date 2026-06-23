"""
backend/ssh_config.py
---------------------
Centralised SSH connection settings for the remote GVHMR server.
Import SSH_KEY_PATH and other constants from here instead of hardcoding
them in remote_ssh_pipeline.py.

SSH_KEY_PATH resolves in this order:
  1. R2P_SSH_KEY environment variable (set this in production)
  2. ~/.ssh/id_ed25519  (standard Ed25519 key — works on any OS/user)
  3. ~/.ssh/id_rsa      (RSA fallback)
"""

import os
from pathlib import Path

# ── Remote server ─────────────────────────────────────────────────────────────
REMOTE_IP   = "101.6.162.37"
REMOTE_PORT = 62222
USERNAME    = "ai"

# ── SSH key — resolved dynamically so it works for any Windows/Mac/Linux user ─
def _find_ssh_key() -> str:
    # 1. Explicit environment override
    env_key = os.environ.get("R2P_SSH_KEY")
    if env_key and Path(env_key).exists():
        return env_key

    home = Path.home()

    # 2. Ed25519 (preferred)
    ed = home / ".ssh" / "id_ed25519"
    if ed.exists():
        return str(ed)

    # 3. RSA fallback
    rsa = home / ".ssh" / "id_rsa"
    if rsa.exists():
        return str(rsa)

    raise FileNotFoundError(
        f"No SSH key found. Expected {ed} or {rsa}.\n"
        "Set the R2P_SSH_KEY environment variable to your key path:\n"
        "  Windows PowerShell:  $env:R2P_SSH_KEY = 'C:\\Users\\YOU\\.ssh\\id_ed25519'\n"
        "  Linux/Mac:           export R2P_SSH_KEY=~/.ssh/id_ed25519"
    )

SSH_KEY_PATH = _find_ssh_key()