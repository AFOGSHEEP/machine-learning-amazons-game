"""Exit 0 if PyTorch sees CUDA, else print diagnosis and exit 1."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed in this interpreter:", sys.executable, file=sys.stderr)
        return 1

    ver = torch.__version__
    if "+cpu" in ver or torch.version.cuda is None:
        print("ERROR: CPU-only PyTorch:", ver, file=sys.stderr)
        print("  Interpreter:", sys.executable, file=sys.stderr)
        print("  Fix: use conda env with CUDA build, e.g. conda activate myPytorch", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: PyTorch is CUDA build but cuda.is_available() is False", file=sys.stderr)
        print("  Interpreter:", sys.executable, file=sys.stderr)
        print("  Check: nvidia-smi, driver, reboot", file=sys.stderr)
        return 1

    print("OK:", sys.executable)
    print("torch", ver, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
