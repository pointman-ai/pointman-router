"""The model: checked, loaded once and kept, then asked one or two questions per post.

The gate asks "reply needed" or "no reply needed" of the post's state; p_reply is the first's share of the two.
The router asks which candidate, with each agent's description as its label's description, only when the
caller says the rule can't place the post and p_reply reaches the cutoff. Both calls are the ones the model
was scored and timed with, so its answers match those runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

REPLY, NO_REPLY = "reply needed", "no reply needed"
GATE = {"respond": {"labels": [REPLY, NO_REPLY], "multi_label": True, "cls_threshold": 0.0}}
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
DEFAULT_CUTOFF = 0.05


class Failure(Exception):
    """A failure a caller can branch on: `code` goes to the node as the failed command's code, and starts the
    message too (`no_weights: …`), as the node passes on only a failed job's message for now."""

    def __init__(self, message: str, code: str):
        prefix = f"{code}: "
        super().__init__(message if message.startswith(prefix) else prefix + message)
        self.code = code


def route_task(candidates: dict[str, str]) -> dict:
    return {"route": {"labels": dict(candidates), "multi_label": True, "cls_threshold": 0.0}}


def scores_of(out: dict, task: str) -> dict[str, float]:
    x = out[task]
    return {r["label"]: float(r["confidence"]) for r in ([x] if isinstance(x, dict) else x)}


def p_reply(out: dict) -> float:
    s = scores_of(out, "respond")
    r = s.get(REPLY, 0.0)
    return r / max(r + s.get(NO_REPLY, 0.0), 1e-9)


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def params_of(p: dict) -> tuple[str, str, dict[str, str], bool, float]:
    """The command's params, checked. Messages name the field, never its text."""
    post, state, cands, need, cutoff = (p.get("post", ""), p.get("state"), p.get("candidates", {}),
                                        p.get("need_router"), p.get("cutoff", DEFAULT_CUTOFF))
    if not isinstance(post, str):
        raise Failure("post must be a post id", "bad_params")
    if not isinstance(state, str) or not state.strip():
        raise Failure("state must be the post's text", "bad_params")
    if not isinstance(cands, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cands.items()):
        raise Failure("candidates must map each handle to its description", "bad_params")
    if not isinstance(need, bool):
        raise Failure("need_router must be true or false", "bad_params")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not 0 <= cutoff <= 1:
        raise Failure("cutoff must be a number from 0 to 1", "bad_params")
    return post, state, cands, need, float(cutoff)


def pick_device(setting: str, torch: Any, nvidia: bool | None = None, apple: bool | None = None) -> str:
    """The device to run on. A GPU it picks (or is told to use) but torch can't reach is an error: a quiet
    fall back to the CPU would make every decision 20 times slower without anyone knowing."""
    nvidia = Path("/proc/driver/nvidia/version").exists() if nvidia is None else nvidia
    apple = (sys.platform == "darwin" and platform.machine() == "arm64") if apple is None else apple
    want = setting if setting != "auto" else "cuda" if nvidia else "mps" if apple else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise Failure("torch can't reach the NVIDIA GPU here (CUDA isn't available)", "no_device")
    if want == "mps" and not torch.backends.mps.is_available():
        raise Failure("torch can't reach the Apple GPU here (MPS isn't available)", "no_device")
    return want


def load_gliner(model_dir: Path, settings: dict) -> tuple[Any, str, int]:
    """The checkpoint in fp16 on a GPU, fp32 on the CPU: (model, device, threads)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # everything it needs is in the folder
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from gliner2 import AutoExtractor

    threads = int(settings.get("threads") or 0)
    if threads:
        torch.set_num_threads(threads)
    device = pick_device(settings.get("device") or "auto", torch)
    m = AutoExtractor.from_pretrained(str(model_dir)).eval()
    if device != "cpu":
        m = m.to(device).half()
    return m, device, torch.get_num_threads()


class Worker:
    """The model in its own process (router.worker), asked over its stdin and stdout. Closing it frees all of
    its memory."""

    def __init__(self, argv: list[str] | None = None):
        self.proc = subprocess.Popen(argv or [sys.executable, "-m", "router.worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                     cwd=Path(__file__).resolve().parents[1])

    def ask(self, req: dict) -> Any:
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            self.close()
            raise Failure(f"the model's process stopped (exit {self.proc.returncode}); the next post starts it again",
                          "model_stopped")
        answer = json.loads(line)
        if "error" in answer:
            raise Failure(answer["error"], answer.get("code") or "model_error")
        return answer["ok"]

    def classify_text(self, text: str, tasks: dict, include_confidence: bool = True) -> dict:
        return self.ask({"text": text, "tasks": tasks})

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def load_in_worker(model_dir: Path, settings: dict) -> tuple[Worker, str, int]:
    w = Worker()
    try:
        info = w.ask({"load": {"model_dir": str(model_dir), "settings": settings}})
    except Failure:
        w.close()
        raise
    return w, info["device"], info["threads"]


class Router:
    """One model, loaded on first need (or at start) in its own process, kept until it's idle for
    idle_unload_minutes. A model process that stopped is started again on the next post."""

    def __init__(self, settings: dict, loader: Callable[[Path, dict], tuple[Any, str, int]] = load_in_worker,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.lock = threading.RLock()
        self.model: Any = None
        self.device: str | None = None
        self.threads = 0
        self.sha = ""
        self._sha_seen: dict[tuple, str] = {}
        self._loader = loader
        self._clock = clock
        self.last_used = clock()

    @property
    def model_dir(self) -> Path:
        return Path(os.path.expanduser(self.settings.get("model_dir") or "")).resolve()

    def check_weights(self) -> str:
        """model.safetensors' sha256, once its folder has every file and the hash is the one expected."""
        d = self.model_dir
        missing = [f for f in FILES if not (d / f).is_file()] + ([] if (d / "encoder_config").is_dir()
                                                                   else ["encoder_config/"])
        if missing:
            raise Failure(f"no weights in {d}: missing {', '.join(missing)}", "no_weights")
        f = d / "model.safetensors"
        st = f.stat()
        key = (str(f), st.st_size, st.st_mtime_ns, st.st_ino)
        sha = self._sha_seen.get(key) or sha256_of(f)
        self._sha_seen = {key: sha}
        want = str(self.settings.get("weights_sha256") or "").strip().lower()
        if sha != want:
            raise Failure(f"model.safetensors' sha256 is {sha[:12]}…, not the expected {want[:12]}…", "bad_weights")
        return sha

    def ensure_loaded(self) -> float:
        """Load it if it isn't: the milliseconds that took, 0 when it was loaded already."""
        with self.lock:
            if self.model is not None:
                if getattr(self.model, "alive", True):
                    return 0.0
                self.unload()                      # its process stopped between posts
            t = time.perf_counter()
            sha = self.check_weights()
            self.model, self.device, self.threads = self._loader(self.model_dir, self.settings)
            self.sha = sha
            self.last_used = self._clock()
            return (time.perf_counter() - t) * 1000

    def unload(self) -> None:
        with self.lock:
            model, self.model = self.model, None
            if hasattr(model, "close"):
                model.close()

    def unload_if_idle(self) -> bool:
        minutes = int(self.settings.get("idle_unload_minutes") or 0)
        with self.lock:
            if not minutes or self.model is None or self._clock() - self.last_used < minutes * 60:
                return False
            self.unload()
            return True

    def decide(self, params: dict) -> dict:
        post, state, candidates, need_router, cutoff = params_of(params)
        with self.lock:
            load_ms = self.ensure_loaded()
            t = time.perf_counter()
            pr = p_reply(self.model.classify_text(state, GATE, include_confidence=True))
            scores: dict[str, float] = {}
            if need_router and pr >= cutoff and candidates:
                scores = scores_of(self.model.classify_text(state, route_task(candidates), include_confidence=True),
                                   "route")
            ms = (time.perf_counter() - t) * 1000
            self.last_used = self._clock()
            return {"p_reply": pr, "scores": scores, "model": self.settings.get("model_name") or "",
                    "weights_sha256": self.sha[:12], "device": self.device, "threads": self.threads,
                    "ms": round(ms, 2), "load_ms": round(load_ms, 1)}
