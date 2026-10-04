import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
TIERS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Current date and time: Monday, March 16, 2026, 13:00 UTC.
The user is Sam Rivera (sam.rivera@example.com).

Use a tool when you need information or need to take an action. Answer directly when you can without one.
If a request is missing information you need, ask the user a clarifying question instead of guessing.
If a tool returns an error, decide whether to retry, fix your arguments, or explain the problem to the user.
When you have the answer, reply concisely."""


def load_config(path: Path = ROOT / "models.yaml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["defaults"]["max_tokens"] = _on_off_keys(cfg["defaults"]["max_tokens"])
    srv = cfg["llama_server"]
    cfg["backends"]["llamacpp"]["base_url"] = f"http://{srv['host']}:{srv['port']}/v1"
    for name, m in cfg["models"].items():
        m["name"] = name
        m["thinking"] = _on_off_keys(m.get("thinking") or {})
        m.setdefault("model_id", name)  # llama-server serves each model under its name (--alias)
    return cfg


def _on_off_keys(d):
    """YAML 1.1 reads bare on/off keys as booleans; map them back to 'on'/'off'."""
    if not isinstance(d, dict):
        return d
    return {({True: "on", False: "off"}.get(k, k) if isinstance(k, bool) else k): (v if v is not None else {})
            for k, v in d.items()}


def select_models(cfg: dict, spec: str) -> list[dict]:
    """spec: 'all', 'ladder', 'reference', 'parity', 'local', 'hosted', or comma-separated names."""
    models = list(cfg["models"].values())
    local = lambda m: cfg["backends"][m["backend"]].get("local", False)
    groups = {
        "all": models,
        "ladder": [m for m in models if m["role"] == "ladder"],
        "reference": [m for m in models if m["role"] == "reference"],
        "parity": [m for m in models if m["role"] == "parity"],
        "local": [m for m in models if local(m)],
        "hosted": [m for m in models if not local(m)],
    }
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part in groups:
            out += groups[part]
        elif part in cfg["models"]:
            out.append(cfg["models"][part])
        else:
            raise SystemExit(f"Unknown model or group '{part}'. Known: {', '.join([*groups, *cfg['models']])}")
    seen = set()
    return [m for m in out if not (m["name"] in seen or seen.add(m["name"]))]


def api_key(cfg: dict, backend: str) -> str:
    b = cfg["backends"][backend]
    if "api_key_env" in b:
        key = os.environ.get(b["api_key_env"])
        if not key:
            raise SystemExit(f"Set {b['api_key_env']} in .env to use the {backend} backend.")
        return key
    return b.get("api_key", "none")


def load_tasks(tiers: list[str] | None = None) -> list[dict]:
    tasks = []
    for tier in tiers or TIERS:
        path = TASKS_DIR / f"{tier}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing. Build the tasks first: python -m tasks.build")
        tasks += [json.loads(line) for line in path.open()]
    return tasks
