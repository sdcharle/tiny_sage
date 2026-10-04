"""The agent loop: prompt -> model -> execute tool calls -> repeat, logging everything for later re-grading.

Deliberately framework-free so that nothing but the model and its chat template decides what happens.
"""

import json
import re
import time
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from harness.backends import BackendError, ModelTurn
from harness.config import SYSTEM_PROMPT
from harness.grader import grade
from harness.metrics import MemorySampler
from tools import SCHEMAS, World, openai_tools

VALIDATORS = {name: Draft202012Validator(s["parameters"]) for name, s in SCHEMAS.items()}

THINK_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.S),
    re.compile(r"<\|channel\>thought(.*?)<channel\|>", re.S),   # Gemma 4 style thought channel
]
TEXT_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
    re.compile(r"```(?:json|tool_code|tool_call)?\s*(\{.*?\})\s*```", re.S),
]


def split_thinking(content: str) -> tuple[str, str]:
    thoughts = []
    for pat in THINK_PATTERNS:
        thoughts += pat.findall(content)
        content = pat.sub("", content)
    if "</think>" in content:  # opening tag consumed by the chat template
        head, content = content.split("</think>", 1)
        thoughts.append(head)
    return content.strip(), "\n".join(t.strip() for t in thoughts)


def detect_text_tool_calls(content: str, available: set[str]) -> list[dict]:
    """Tool calls written as plain text instead of structured tool_calls (a common small-model failure)."""
    candidates = []
    for pat in TEXT_CALL_PATTERNS:
        candidates += pat.findall(content)
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    found = []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name") or obj.get("tool")
        if name in available:
            args = fn.get("arguments", fn.get("parameters", {}))
            found.append({"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)})
    return found


def check_call(name: str, raw_args, available: set[str]) -> tuple[dict | None, str | None, str | None]:
    """Returns (args, error_kind, detail)."""
    if name not in available:
        return None, "unknown_tool", f"Unknown tool '{name}'. Available tools: {', '.join(sorted(available))}."
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(raw_args) if (raw_args or "").strip() else {}
        except json.JSONDecodeError as e:
            return None, "bad_json", f"Arguments were not valid JSON: {e}"
    if not isinstance(args, dict):
        return None, "bad_json", "Arguments must be a JSON object."
    err = next(iter(VALIDATORS[name].iter_errors(args)), None)
    if err is not None:
        where = "/".join(map(str, err.absolute_path)) or "arguments"
        return args, "schema", f"Invalid arguments ({where}): {err.message}"
    return args, None, None


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"... [truncated, {len(s)} chars]"


def run_task(task: dict, backend, *, model: dict, mode: str, settings: dict, sample: int = 0,
             lenient: bool = False, sample_memory: bool = False) -> dict:
    think = model.get("thinking", {}).get(mode, {}) if model else {}
    available = set(task["tools"])
    tools = openai_tools(task["tools"])
    world = World(task)
    log_chars = settings.get("tool_result_log_chars", 2000)
    max_tokens = settings["max_tokens"][mode] if isinstance(settings["max_tokens"], dict) else settings["max_tokens"]

    messages = [{"role": "system", "content": think.get("system_prefix", "") + SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"] + think.get("user_suffix", "")}]
    rec = {
        "task_id": task["id"], "tier": task["tier"], "model": model["name"], "family": model.get("family"),
        "backend": model.get("backend"), "model_id": model.get("model_id"), "thinking": mode,
        "temperature": settings["temperature"], "sample": sample, "lenient": lenient,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turns": [], "calls": [], "final": None, "stop_reason": None, "error": None,
    }
    model_s = tool_s = 0.0
    t_start = time.perf_counter()

    with MemorySampler(enabled=sample_memory) as mem:
        for turn_i in range(settings["max_turns"]):
            try:
                mt: ModelTurn = backend.chat(messages, tools, settings["temperature"], max_tokens, think.get("extra_body"))
            except BackendError as e:
                rec["stop_reason"], rec["error"] = "api_error", str(e)
                break
            model_s += mt.latency_s
            content, inline_thoughts = split_thinking(mt.content)
            reasoning = "\n".join(x for x in (mt.reasoning, inline_thoughts) if x)
            rec["turns"].append({
                "turn": turn_i, "latency_s": round(mt.latency_s, 3), "usage": mt.usage, "finish_reason": mt.finish_reason,
                "provider": mt.provider, "content": content, "reasoning": reasoning,
                "tool_calls": mt.tool_calls,
            })

            calls, text_format = mt.tool_calls, False
            if not calls:
                text_calls = detect_text_tool_calls(content, available)
                if text_calls and lenient:
                    calls, text_format = [{**tc, "id": f"text_{turn_i}_{k}"} for k, tc in enumerate(text_calls)], True
                else:
                    for tc in text_calls:
                        rec["calls"].append({"turn": turn_i, "name": tc["name"], "raw_args": tc["arguments"], "args": None,
                                             "valid": False, "error_kind": "text_tool_call",
                                             "detail": "Tool call written as text instead of a structured tool call."})

            if not calls:
                rec["final"] = content
                rec["stop_reason"] = "length" if mt.finish_reason == "length" else "final"
                break

            if text_format:
                messages.append({"role": "assistant", "content": mt.content})
            else:
                messages.append({"role": "assistant", "content": content or None, "tool_calls": [
                    {"id": tc["id"] or f"call_{turn_i}_{k}", "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}
                    for k, tc in enumerate(calls)]})

            for k, tc in enumerate(calls):
                args, err_kind, detail = check_call(tc["name"], tc["arguments"], available)
                t0 = time.perf_counter()
                result = {"error": detail} if err_kind else world.call(tc["name"], args)
                tool_s += time.perf_counter() - t0
                result_s = result if isinstance(result, str) else json.dumps(result)
                rec["calls"].append({"turn": turn_i, "name": tc["name"], "raw_args": tc["arguments"], "args": args,
                                     "valid": err_kind is None, "error_kind": err_kind, "detail": detail,
                                     "text_format": text_format, "result": _truncate(result_s, log_chars)})
                if text_format:
                    messages.append({"role": "user", "content": f"[tool result for {tc['name']}]\n{result_s}"})
                else:
                    messages.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{turn_i}_{k}", "content": result_s})
        else:
            rec["stop_reason"] = "max_turns"

    usage = [t["usage"] for t in rec["turns"]]
    rec["state"] = world.state()
    rec["timing"] = {"wall_s": round(time.perf_counter() - t_start, 3), "model_s": round(model_s, 3),
                     "tool_s": round(tool_s, 4)}
    rec["tokens"] = {k: sum(u.get(k, 0) or 0 for u in usage)
                     for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")}
    rec["cost_usd"] = round(sum(u.get("cost_usd", 0) or 0 for u in usage), 6)
    rec["tokens_per_s"] = round(rec["tokens"]["completion_tokens"] / model_s, 1) if model_s and rec["tokens"]["completion_tokens"] else None
    rec["memory"] = mem.result()
    rec["thinking_detected"] = any(t["reasoning"] for t in rec["turns"]) or bool(rec["tokens"]["reasoning_tokens"])
    rec["providers"] = sorted({t["provider"] for t in rec["turns"] if t["provider"]})
    rec["grade"] = grade(task, rec)
    return rec
