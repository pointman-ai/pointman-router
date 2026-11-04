"""Where Linux finds the GPU driver and CUDA libraries, set before torch loads.

The dynamic loader reads LD_LIBRARY_PATH once, when a process starts, and the node starts extensions with a
clean environment. So on Linux the extension starts itself again with the path set (os.execve keeps its pid,
stdin and stdout, so the node sees nothing): the CUDA libraries pip put in the environment, and on NixOS the
driver's folder, which holds libcuda outside the usual places. The same approach as the voice extension's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DONE = "POINTMAN_ROUTER_LIBS"


def _nvidia_libs(prefix: Path) -> list[str]:
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    return sorted(str(p) for p in (site / "nvidia").glob("*/lib") if p.is_dir())


def wanted(nixos: bool, env: dict[str, str], prefix: Path) -> dict[str, str]:
    """The environment to add, or {} when nothing's missing."""
    paths = [p for p in ["/run/opengl-driver/lib"] if nixos and Path(p).is_dir()]
    paths += _nvidia_libs(prefix)
    have = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    add = [p for p in paths if p not in have]
    return {"LD_LIBRARY_PATH": ":".join(add + have)} if add else {}


def ensure() -> None:
    """On Linux, start again once with the libraries' folders on the loader's path."""
    if not sys.platform.startswith("linux") or os.environ.get(DONE):
        return
    add = wanted(Path("/etc/NIXOS").exists(), dict(os.environ), Path(sys.prefix))
    if not add:
        return
    os.execve(sys.executable, [sys.executable, "-m", "router", *sys.argv[1:]], {**os.environ, **add, DONE: "1"})
