"""Launch bare llama-server with pinned flags, one model per process.

The flags come from `llama_server` in models.yaml and are the same for every local model. The full
command line and `llama-server --version` are written to results/<tag>/meta.json for each run.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def server_cfg(cfg: dict) -> dict:
    return cfg["llama_server"]


def binary(cfg: dict) -> str:
    exe = os.environ.get("LLAMA_SERVER_BIN") or server_cfg(cfg)["binary"]
    found = shutil.which(exe) or (exe if Path(exe).exists() else None)
    if not found:
        raise SystemExit(f"llama-server not found ('{exe}'). Install it with `brew install llama.cpp`, "
                         "or set LLAMA_SERVER_BIN / llama_server.binary in models.yaml.")
    return found


def models_dir(cfg: dict) -> Path:
    return Path(os.environ.get("TINY_SAGE_MODELS_DIR") or server_cfg(cfg)["models_dir"]).expanduser()


def gguf_path(cfg: dict, model: dict) -> Path:
    return models_dir(cfg) / model["hf_repo"] / model["gguf"]


def base_url(cfg: dict) -> str:
    s = server_cfg(cfg)
    return f"http://{s['host']}:{s['port']}"


def version(cfg: dict) -> str | None:
    try:
        out = subprocess.run([binary(cfg), "--version"], capture_output=True, text=True, timeout=30)
    except (SystemExit, OSError, subprocess.TimeoutExpired):
        return None
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith("version")]
    return lines[0] if lines else (out.stdout + out.stderr).strip()[-200:] or None


def command(cfg: dict, model: dict, log_file: Path | None = None) -> list[str]:
    s = server_cfg(cfg)
    path = gguf_path(cfg, model)
    if not path.exists():
        raise SystemExit(f"Model file missing: {path}\nDownload it with: python -m harness download --models {model['name']}")
    cmd = [binary(cfg), "--model", str(path), "--alias", model["name"],
           "--host", s["host"], "--port", str(s["port"]), "--ctx-size", str(cfg["defaults"]["context_length"])]
    for flag, value in s["flags"].items():
        if value is True:
            cmd.append(flag)
        elif value is not False and value is not None:
            cmd += [flag, str(value)]
    if log_file:
        cmd += ["--log-file", str(log_file)]
    return cmd


def _get(url: str, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"null")


def served_models(cfg: dict) -> list[str] | None:
    """Aliases served by whatever is listening on the configured port, or None if nothing is."""
    try:
        _, body = _get(base_url(cfg) + "/v1/models")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [m["id"] for m in body.get("data", [])]


class LlamaServer:
    """Context manager: start llama-server for one model, wait until it's healthy, stop it on exit."""

    def __init__(self, cfg: dict, model: dict, log_dir: Path, startup_timeout_s: int = 600):
        self.cfg, self.model = cfg, model
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"llama-server__{model['name']}.log"
        self.cmd = command(cfg, model, self.log_path)
        self.startup_timeout_s = startup_timeout_s
        self.proc = None

    def __enter__(self):
        if served_models(self.cfg) is not None:
            raise SystemExit(f"Something is already listening on {base_url(self.cfg)}. Stop it, or drop "
                             "--manage-server and start the right model yourself with scripts/serve.sh.")
        print(f"  llama-server: {' '.join(self.cmd)}")
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                status, _ = _get(base_url(self.cfg) + "/health", timeout=2)
                if status == 200:
                    print(f"  llama-server ready (pid {self.proc.pid}), log: {self.log_path}")
                    return self
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1)
        self.stop()
        raise SystemExit(f"llama-server did not become healthy within {self.startup_timeout_s}s; see {self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def __exit__(self, *exc):
        self.stop()
