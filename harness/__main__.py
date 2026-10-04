"""tiny_sage command line. Run `python -m harness --help`."""

import argparse
import contextlib
import json
import os
import sys
import urllib.error
import urllib.request

from harness import llamaserver
from harness import run as runmod
from harness.backends import ModelTurn, OracleBackend, OpenAIBackend, make_backend
from harness.config import RESULTS_DIR, TIERS, api_key, load_config, load_tasks, select_models
from harness.grader import grade
from harness.runner import run_task


def _tiers(spec):
    return TIERS if spec == "all" else [t.strip().upper() for t in spec.split(",")]


def _modes(spec):
    return ["off", "on"] if spec == "both" else [spec]


def cmd_list(args):
    cfg = load_config()
    print("Models:")
    for m in cfg["models"].values():
        if m["backend"] == "llamacpp":
            path = llamaserver.gguf_path(cfg, m)
            where = f"{m['hf_repo']}/{m['gguf']}  {'(downloaded)' if path.exists() else '(not downloaded)'}"
        else:
            where = f"{m['model_id']}  provider={m.get('provider')}"
        print(f"  {m['name']:<16} {m['role']:<10} {m['family']:<7} {m['params_b']:>5}B  {m['backend']:<11} {where}")
    print(f"\nGGUF directory: {llamaserver.models_dir(cfg)}")
    print("\nTasks:")
    tasks = load_tasks()
    for tier in TIERS:
        print(f"  {tier}: {sum(t['tier'] == tier for t in tasks)}")
    print(f"  total: {len(tasks)}")


def cmd_models(args):
    cfg = load_config()
    b = cfg["backends"][args.backend]
    backend = OpenAIBackend({"model_id": None}, b, api_key(cfg, args.backend))
    ids = sorted(m["id"] for m in backend.list_models())
    ids = [i for i in ids if not args.search or args.search.lower() in i.lower()]
    print("\n".join(ids) or "(none)")


def cmd_providers(args):
    """OpenRouter: list the providers serving a model, with quantization and price, to pick a pin."""
    url = f"https://openrouter.ai/api/v1/models/{args.model_id}/endpoints"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            eps = json.load(r)["data"]["endpoints"]
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenRouter has no model '{args.model_id}' ({e.code}). Try: python -m harness models openrouter --search <name>")
    print(f"{'pin (provider tag)':<26} {'quant':<8} {'ctx':>7} {'$/M in':>7} {'$/M out':>7} {'uptime':>7}  tools")
    for e in eps:
        p = e.get("pricing", {})
        price = lambda k: f"{float(p.get(k, 0)) * 1e6:.3f}"
        tools = "yes" if "tools" in (e.get("supported_parameters") or []) else "no"
        up = e.get("uptime_last_1d")
        print(f"{e.get('tag') or e.get('provider_name', '?'):<26} {str(e.get('quantization')):<8} "
              f"{e.get('context_length', 0):>7} {price('prompt'):>7} {price('completion'):>7} "
              f"{f'{up:.1f}%' if up is not None else '?':>7}  {tools}")
    print("\nPut the tag in models.yaml as `provider:` for that model.")


def cmd_serve(args):
    """Run llama-server in the foreground for one local model, with the pinned flags."""
    cfg = load_config()
    model = cfg["models"].get(args.model)
    if not model or model["backend"] != "llamacpp":
        local = [n for n, m in cfg["models"].items() if m["backend"] == "llamacpp"]
        sys.exit(f"'{args.model}' is not a local model. Local models: {', '.join(local)}")
    cmd = llamaserver.command(cfg, model)
    print(" ".join(cmd), flush=True)
    os.execv(cmd[0], cmd)


def cmd_download(args):
    """Download the local models' GGUF files into the shared models directory (skips files already there)."""
    from huggingface_hub import hf_hub_download
    cfg = load_config()
    for m in select_models(cfg, args.models):
        if m["backend"] != "llamacpp":
            continue
        path = llamaserver.gguf_path(cfg, m)
        if path.exists():
            print(f"{m['name']}: already at {path}")
            continue
        print(f"{m['name']}: downloading {m['hf_repo']}/{m['gguf']}")
        hf_hub_download(m["hf_repo"], m["gguf"], local_dir=path.parent)
        print(f"  -> {path}")


def _maybe_server(cfg, model, manage, tag="smoke"):
    if manage and model["backend"] == "llamacpp":
        return llamaserver.LlamaServer(cfg, model, RESULTS_DIR / tag / "logs")
    return contextlib.nullcontext()


def cmd_smoke(args):
    """One easy task per model and thinking mode: is the endpoint up, are tool calls structured, does thinking toggle?"""
    cfg = load_config()
    task = next(t for t in load_tasks(["T0"]) if t["id"] == args.task)
    settings = dict(cfg["defaults"])
    rows = []
    for m in select_models(cfg, args.models):
        if m["backend"] == "llamacpp" and not args.manage_server:
            served = llamaserver.served_models(cfg)
            if not served or m["name"] not in served:
                print(f"{m['name']:<16} skipped: llama-server isn't serving it (use --manage-server, or scripts/serve.sh {m['name']})")
                continue
        with _maybe_server(cfg, m, args.manage_server):
            _smoke_model(cfg, m, task, settings, args, rows)
    bad = [r for r in rows if r["thinking"] == "off" and r["thinking_seen"]] + \
          [r for r in rows if r["thinking"] == "on" and not r["thinking_seen"]]
    for r in bad:
        print(f"! {r['model']}: thinking={r['thinking']} but thinking_seen={r['thinking_seen']}; fix its thinking config in models.yaml")


def _smoke_model(cfg, m, task, settings, args, rows):
    backend = make_backend(cfg, m, api_key(cfg, m["backend"]))
    backend.max_attempts = 1  # fail fast: smoke is for spotting setup problems
    for mode in _modes(args.thinking):
        if mode not in m.get("thinking", {}):
            continue
        rec = run_task(task, backend, model=m, mode=mode, settings=settings)
        calls = rec["calls"]
        rows.append({
            "model": m["name"], "thinking": mode,
            "result": "ok" if rec["grade"]["success"] else (rec["error"] or rec["grade"]["failure_mode"]),
            "structured_call": bool(calls) and not any(c.get("error_kind") == "text_tool_call" for c in calls),
            "thinking_seen": rec["thinking_detected"],
            "secs": rec["timing"]["wall_s"], "provider": ",".join(rec["providers"]) or "-",
            "final": (rec["final"] or "")[:60].replace("\n", " "),
        })
        r = rows[-1]
        print(f"{r['model']:<16} think={mode:<3} {str(r['result']):<14} structured={str(r['structured_call']):<5} "
              f"thinking_seen={str(r['thinking_seen']):<5} {r['secs']:>6.1f}s  {r['provider']:<12} {r['final']}")


def cmd_run(args):
    cfg = load_config()
    models = select_models(cfg, args.models)
    tasks = runmod.select_tasks(_tiers(args.tiers), args.task, args.limit)
    if not tasks:
        sys.exit("No tasks selected.")
    runmod.run_matrix(cfg, models, tasks, _modes(args.thinking), samples=args.samples, temperature=args.temperature,
                      lenient=args.lenient, workers=args.workers, tag=args.tag, manage_server=args.manage_server,
                      overwrite=args.overwrite)


class _NullBackend:
    """Always answers without tools. Should fail nearly everything except the T3 restraint tasks."""

    def chat(self, *a, **k):
        return ModelTurn(content="I'm not sure.", reasoning="", tool_calls=[], finish_reason="stop")


def cmd_selftest(args):
    cfg = load_config()
    settings = dict(cfg["defaults"])
    tasks = load_tasks()
    fake = {"name": "oracle", "family": None, "backend": None, "model_id": None, "thinking": {"off": {}}}
    failed = []
    for t in tasks:
        rec = run_task(t, OracleBackend(t), model=fake, mode="off", settings=settings)
        if not rec["grade"]["success"]:
            failed.append((t["id"], rec["grade"]["checks"], rec["final"]))
    null_ok = [t["id"] for t in tasks
               if run_task(t, _NullBackend(), model=fake, mode="off", settings=settings)["grade"]["success"]]
    print(f"oracle: {len(tasks) - len(failed)}/{len(tasks)} tasks solved")
    for tid, checks, final in failed:
        print(f"  FAIL {tid}: {checks}  final={final!r:.120}")
    print(f"null model (never calls tools): {len(null_ok)}/{len(tasks)} solved: {', '.join(null_ok)}")
    sys.exit(1 if failed else 0)


def cmd_regrade(args):
    tasks = {t["id"]: t for t in load_tasks()}
    n = 0
    for path in (RESULTS_DIR / args.tag).rglob("*.jsonl"):
        recs = [json.loads(line) for line in path.open() if line.strip()]
        for r in recs:
            r["grade"] = grade(tasks[r["task_id"]], r)
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        n += len(recs)
    print(f"regraded {n} records under results/{args.tag}")


def cmd_analyze(args):
    from analysis.analyze import main as analyze
    analyze(args.tag)


def main():
    p = argparse.ArgumentParser(prog="python -m harness", description="tiny_sage tool-calling experiment")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show configured models and task counts").set_defaults(fn=cmd_list)

    s = sub.add_parser("serve", help="run llama-server in the foreground for one local model (pinned flags)")
    s.add_argument("model")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("download", help="download local models' GGUFs into the shared models directory")
    s.add_argument("--models", default="local")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("models", help="list model IDs a backend serves")
    s.add_argument("backend", choices=["llamacpp", "openrouter"])
    s.add_argument("--search", default="")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("providers", help="list OpenRouter providers for a model (to pick a pin)")
    s.add_argument("model_id")
    s.set_defaults(fn=cmd_providers)

    s = sub.add_parser("smoke", help="one quick task per model to check the setup")
    s.add_argument("--models", default="all")
    s.add_argument("--thinking", default="both", choices=["off", "on", "both"])
    s.add_argument("--task", default="T0-weather-00")
    s.add_argument("--manage-server", action="store_true", help="start/stop llama-server for each local model")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("run", help="run the experiment matrix (resumes where it left off)")
    s.add_argument("--models", default="ladder", help="names, or groups: all, ladder, reference, parity, local, hosted")
    s.add_argument("--tiers", default="all", help="e.g. T0,T1 or all")
    s.add_argument("--thinking", default="both", choices=["off", "on", "both"])
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--samples", type=int, default=1)
    s.add_argument("--limit", type=int, default=None, help="max tasks per tier")
    s.add_argument("--task", action="append", help="run only this task id (repeatable)")
    s.add_argument("--lenient", action="store_true", help="execute tool calls the model wrote as plain text")
    s.add_argument("--workers", type=int, default=1, help="parallel tasks for hosted models")
    s.add_argument("--tag", default="main", help="results/<tag>/")
    s.add_argument("--manage-server", action="store_true", help="start/stop llama-server for each local model")
    s.add_argument("--overwrite", action="store_true", help="discard existing results for the selected runs")
    s.set_defaults(fn=cmd_run)

    sub.add_parser("selftest", help="replay every task's oracle solution; all must pass").set_defaults(fn=cmd_selftest)

    s = sub.add_parser("regrade", help="re-grade saved transcripts after changing the grader or tasks")
    s.add_argument("--tag", default="main")
    s.set_defaults(fn=cmd_regrade)

    s = sub.add_parser("analyze", help="tables and charts from results/<tag>")
    s.add_argument("--tag", default="main")
    s.set_defaults(fn=cmd_analyze)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
