"""Phase 00 — AI Engineering environment check."""

import os
import platform
import shutil
import subprocess
import sys

from dotenv import load_dotenv


def python_version() -> str:
    """Return Python version as MAJOR.MINOR.MICRO."""
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def mask_secret(secret: str | None) -> str:
    """Mask sensitive API keys before displaying them."""
    if not secret:
        return "missing"

    if len(secret) <= 8:
        return "****"

    return f"{secret[:4]}...{secret[-4:]}"


def parse_nvidia_smi(output: str) -> list[dict]:
    """Parse NVIDIA GPU information."""
    gpus = []

    for line in output.splitlines():
        if not line.strip():
            continue

        name, memory = line.rsplit(",", 1)

        memory_mib = int(memory.strip().split()[0])

        gpus.append(
            {
                "name": name.strip(),
                "memory_mib": memory_mib,
            }
        )

    return gpus


def gpu_info() -> list[dict]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return parse_nvidia_smi(result.stdout)

    except (subprocess.CalledProcessError, OSError, ValueError):
        return []


def main() -> None:
    load_dotenv()

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
