"""Run the experiment matrix and write one JSONL transcript file per (model, thinking mode, sample)."""

import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from harness import llamaserver
from harness.backends import make_backend
from harness.config import RESULTS_DIR, api_key, load_tasks
from harness.runner import run_task


def result_path(tag: str, model: str, mode: str, temperature: float, sample: int) -> Path:
    return RESULTS_DIR / tag / model / f"think-{mode}__t{temperature:g}__s{sample}.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["task_id"] for line in path.open() if line.strip()}


def _status(rec: dict) -> str:
    g = rec["grade"]
    return "ok" if g["success"] else f"FAIL {g['failure_mode']}"


def select_tasks(tiers, task_ids=None, limit=None) -> list[dict]:
    tasks = load_tasks(tiers)
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if limit:
        per_tier = {}
        tasks = [t for t in tasks if per_tier.setdefault(t["tier"], []).append(t) or len(per_tier[t["tier"]]) <= limit]
    return tasks


def write_meta(tag: str, cfg: dict, args: dict):
    path = RESULTS_DIR / tag / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(path.read_text()) if path.exists() else {"runs": []}
    meta["runs"].append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": args,
        "machine": platform.platform(), "python": platform.python_version(),
        "llama_server_version": llamaserver.version(cfg), "llama_server": cfg["llama_server"],
        "models": cfg["models"], "defaults": cfg["defaults"], "server_commands": {},
    })
    path.write_text(json.dumps(meta, indent=2, default=str))


def _note_server_command(tag: str, model: str, cmd: list[str]):
    path = RESULTS_DIR / tag / "meta.json"
    meta = json.loads(path.read_text())
    meta["runs"][-1]["server_commands"][model] = cmd
    path.write_text(json.dumps(meta, indent=2, default=str))


def run_matrix(cfg, models, tasks, modes, *, samples=1, temperature=None, lenient=False, workers=1,
               tag="main", manage_server=False, overwrite=False):
    settings = dict(cfg["defaults"])
    if temperature is not None:
        settings["temperature"] = temperature
    write_meta(tag, cfg, {"models": [m["name"] for m in models], "tasks": len(tasks), "modes": modes,
                          "samples": samples, "temperature": settings["temperature"], "lenient": lenient,
                          "manage_server": manage_server})

    for mi, model in enumerate(models):
        local = cfg["backends"][model["backend"]].get("local", False)
        print(f"\n=== {model['name']} ({model['backend']}: {model.get('gguf') or model['model_id']}) ===")
        if not local and not model.get("provider"):
            print("  ! no provider pinned; OpenRouter may route to different providers (and quantizations)")
        if local and manage_server:
            with llamaserver.LlamaServer(cfg, model, RESULTS_DIR / tag / "logs") as srv:
                _note_server_command(tag, model["name"], srv.cmd)
                _run_model(cfg, model, tasks, modes, settings, samples, lenient, workers, tag, overwrite, local)
            if mi < len(models) - 1 and settings.get("cooldown_s"):
                print(f"  cooling down {settings['cooldown_s']}s")
                time.sleep(settings["cooldown_s"])
        else:
            if local:
                served = llamaserver.served_models(cfg)
                if served is None:
                    raise SystemExit(f"No llama-server at {llamaserver.base_url(cfg)}. Start it with "
                                     f"`scripts/serve.sh {model['name']}`, or pass --manage-server.")
                if model["name"] not in served:
                    raise SystemExit(f"llama-server is serving {served}, not {model['name']}. "
                                     f"Restart it with `scripts/serve.sh {model['name']}`.")
            _run_model(cfg, model, tasks, modes, settings, samples, lenient, workers, tag, overwrite, local)


def _run_model(cfg, model, tasks, modes, settings, samples, lenient, workers, tag, overwrite, local):
    backend = make_backend(cfg, model, api_key(cfg, model["backend"]))
    n_workers = 1 if local else max(1, workers)
    for mode in modes:
        if mode not in model.get("thinking", {}):
            print(f"  - thinking {mode}: not supported by this model, skipping")
            continue
        for s in range(samples):
            path = result_path(tag, model["name"], mode, settings["temperature"], s)
            path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite and path.exists():
                path.unlink()
            done = _done_ids(path)
            todo = [t for t in tasks if t["id"] not in done]
            print(f"  - thinking {mode}, sample {s}: {len(todo)} to run ({len(done)} already done) -> {path.relative_to(RESULTS_DIR.parent)}")
            lock = threading.Lock()
            counter = {"n": 0, "ok": 0}

            def one(task):
                rec = run_task(task, backend, model=model, mode=mode, settings=settings, sample=s,
                               lenient=lenient, sample_memory=local)
                with lock:
                    with path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    counter["n"] += 1
                    counter["ok"] += rec["grade"]["success"]
                    print(f"    [{counter['n']:>3}/{len(todo)}] {task['id']:<22} {_status(rec):<22} "
                          f"{rec['timing']['wall_s']:>6.1f}s  turns={len(rec['turns'])}")

            if n_workers == 1:
                for t in todo:
                    one(t)
            else:
                with ThreadPoolExecutor(n_workers) as pool:
                    list(pool.map(one, todo))
            if todo:
                print(f"    -> {counter['ok']}/{counter['n']} succeeded")
