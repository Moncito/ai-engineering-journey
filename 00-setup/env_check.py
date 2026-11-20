"""Phase 00 — environment check.

Prints a report of your AI dev environment: Python version, GPU, API keys.

YOUR TASK: replace every `raise NotImplementedError` with a real implementation
until `uv run pytest 00-setup` is all green. Then run:

    uv run python 00-setup/env_check.py
"""

import os
import platform
import shutil
import subprocess
import sys

from dotenv import load_dotenv


def python_version() -> str:
    """Return the running Python version as 'MAJOR.MINOR.MICRO', e.g. '3.12.2'.

    Hint: look at `sys.version_info`.
    """
    raise NotImplementedError


def mask_secret(secret: str | None) -> str:
    """Return a safe-to-print version of a secret.

    Rules:
      - None or empty string        -> "missing"
      - 8 characters or fewer       -> "****"   (too short to reveal anything)
      - longer                      -> first 4 chars + "..." + last 4 chars
                                       e.g. "sk-ant-api03-abcdefghijklmnop" -> "sk-a...mnop"
    """
    raise NotImplementedError


def parse_nvidia_smi(output: str) -> list[dict]:
    """Parse the output of:

        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    Each non-blank line looks like:  "NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB"

    Return one dict per GPU:  {"name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU", "memory_mib": 4096}
    Blank lines are ignored. Empty output -> [].
    """
    raise NotImplementedError


def gpu_info() -> list[dict]:
    """Run nvidia-smi and return parsed GPUs, or [] if nvidia-smi is not installed.

    Hint: `shutil.which("nvidia-smi")` tells you if it exists.
          `subprocess.run([...], capture_output=True, text=True)` runs it.
          Reuse parse_nvidia_smi — keep the parsing logic pure and testable.
    """
    raise NotImplementedError


def main() -> None:
    load_dotenv()  # reads .env in the current directory into os.environ

    print("=== AI Engineering environment ===")
    print(f"Python      : {python_version()}  ({sys.executable})")
    print(f"OS          : {platform.system()} {platform.release()}")

    gpus = gpu_info()
    if gpus:
        for gpu in gpus:
            print(f"GPU         : {gpu['name']} ({gpu['memory_mib']} MiB)")
    else:
        print("GPU         : none detected (CPU only)")

    print(f"ANTHROPIC_API_KEY : {mask_secret(os.getenv('ANTHROPIC_API_KEY'))}")


if __name__ == "__main__":
    main()
