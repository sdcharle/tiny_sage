"""Grade a transcript against its task's `expect` spec. Pure function of (task, record), so runs can be re-graded.

Expect spec keys (all optional):
  first_call:        tool name the model should pick first (T1 selection metric)
  required_calls:    [{"name": ..., "args": {key: value_spec}}] each matched by some valid call
  no_calls:          true -> no tool calls except `allowed_calls`
  asks_clarification:true -> no disallowed calls and the final answer asks a question
  allowed_calls:     tool names that don't count as over-calling for no_calls / asks_clarification
  max_calls:         upper bound on call attempts
  final_all:         [regex | [regex, ...]] every entry must match the normalized final answer (a list = any-of)
  final_none:        [regex] none may match
  state:             {"emails"|"events": [{field: value_spec}]} each expected item matched by a side effect
  state_counts:      {"emails"|"events": n} exact number of side effects

Value specs: a literal (case-insensitive for strings, 1e-6 tolerance for numbers), or one of
  {"$contains": s} {"$regex": r} {"$any": [specs]} {"$approx": x, "$tol": t} {"$set": [items]}
  {"$default": v} (absent or equal to v) {"$absent": true}
"""

import re

MISSING = object()

FAILURE_ORDER = ["api_error", "under_call", "over_call", "format_error", "wrong_tool", "bad_args",
                 "no_final", "wrong_answer"]


def normalize_final(text: str | None) -> str:
    t = (text or "").lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)   # 1,234 -> 1234
    t = t.replace("**", "").replace("`", "").replace("−", "-")
    return t


def match_value(spec, actual) -> bool:
    if isinstance(spec, dict) and spec and all(k.startswith("$") for k in spec):
        if "$default" in spec:
            return actual is MISSING or actual is None or match_value(spec["$default"], actual)
        if "$absent" in spec:
            return actual is MISSING or actual is None
        if actual is MISSING or actual is None:
            return False
        if "$contains" in spec:
            return str(spec["$contains"]).lower() in str(actual).lower()
        if "$regex" in spec:
            return re.search(spec["$regex"], str(actual), re.I) is not None
        if "$any" in spec:
            return any(match_value(s, actual) for s in spec["$any"])
        if "$approx" in spec:
            return _is_num(actual) and abs(float(actual) - spec["$approx"]) <= spec.get("$tol", 0.01)
        if "$set" in spec:
            if not isinstance(actual, list):
                return False
            norm = lambda xs: sorted(str(x).strip().lower() for x in xs)
            return norm(spec["$set"]) == norm(actual)
        raise ValueError(f"unknown spec {spec}")
    if actual is MISSING:
        return False
    if isinstance(spec, bool) or isinstance(actual, bool):
        return spec == actual
    if _is_num(spec):
        return _is_num(actual) and abs(float(spec) - float(actual)) < 1e-6
    if isinstance(spec, str):
        return isinstance(actual, str) and spec.strip().lower() == actual.strip().lower()
    if isinstance(spec, list):
        return isinstance(actual, list) and len(spec) == len(actual) and all(map(match_value, spec, actual))
    if isinstance(spec, dict):
        return isinstance(actual, dict) and match_fields(spec, actual)
    return spec == actual


def match_fields(spec: dict, obj: dict) -> bool:
    return all(match_value(v, obj.get(k, MISSING)) for k, v in spec.items())


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _final_ok(patterns, final: str) -> bool:
    for p in patterns:
        alts = p if isinstance(p, list) else [p]
        if not any(re.search(a, final, re.I) for a in alts):
            return False
    return True


def grade(task: dict, rec: dict) -> dict:
    exp = task["expect"]
    calls = rec.get("calls", [])
    valid = [c for c in calls if c["valid"]]
    final = normalize_final(rec.get("final"))
    stop = rec.get("stop_reason")
    allowed = set(exp.get("allowed_calls", []))
    checks = {}

    if stop == "api_error":
        checks["api"] = False

    # Selection: the first call attempt whose name is a real tool (a schema slip still counts as choosing it).
    named = [c for c in calls if c.get("name") and c.get("error_kind") != "unknown_tool"]
    if "first_call" in exp:
        checks["selection"] = bool(named) and named[0]["name"] == exp["first_call"]

    req = exp.get("required_calls", [])
    req_name_hits = req_arg_hits = 0
    for r in req:
        same_name = [c for c in valid if c["name"] == r["name"]]
        if same_name:
            req_name_hits += 1
        if any(match_fields(r.get("args", {}), c["args"]) for c in same_name):
            req_arg_hits += 1
    if req:
        checks["required_tools"] = req_name_hits == len(req)
        checks["required_args"] = req_arg_hits == len(req)

    disallowed = [c for c in calls if c.get("name") not in allowed]
    over_call = (exp.get("no_calls") or exp.get("asks_clarification")) and bool(disallowed)
    if exp.get("no_calls") or exp.get("asks_clarification"):
        checks["restraint"] = not over_call
    if exp.get("asks_clarification"):
        checks["clarifies"] = "?" in final
    if "max_calls" in exp:
        checks["max_calls"] = len(calls) <= exp["max_calls"]

    if stop in ("max_turns", "length") or rec.get("final") is None:
        checks["finished"] = False
    if exp.get("final_all"):
        checks["final"] = _final_ok(exp["final_all"], final)
    if exp.get("final_none"):
        checks["final_none"] = not any(re.search(p, final, re.I) for p in exp["final_none"])

    state = rec.get("state", {})
    for kind, items in exp.get("state", {}).items():
        checks[f"state_{kind}"] = all(any(match_fields(it, a) for a in state.get(kind, [])) for it in items)
    for kind, n in exp.get("state_counts", {}).items():
        checks[f"count_{kind}"] = len(state.get(kind, [])) == n

    success = all(checks.values())
    under_call = bool(req) and not calls
    n_invalid = sum(not c["valid"] for c in calls)

    failure = None
    if not success:
        reasons = {
            "api_error": stop == "api_error",
            "under_call": under_call,
            "over_call": bool(over_call),
            "format_error": n_invalid > 0,
            "wrong_tool": checks.get("selection") is False or checks.get("required_tools") is False,
            "bad_args": checks.get("required_args") is False,
            "no_final": checks.get("finished") is False,
            "wrong_answer": True,
        }
        failure = next(k for k in FAILURE_ORDER if reasons[k])

    turns = len(rec.get("turns", []))
    return {
        "success": success,
        "failure_mode": failure,
        "checks": checks,
        "n_calls": len(calls),
        "n_valid_calls": len(valid),
        "valid_call_rate": (len(valid) / len(calls)) if calls else None,
        "selection_ok": checks.get("selection"),
        "args_match_rate": (req_arg_hits / len(req)) if req else None,
        "over_call": bool(over_call),
        "under_call": under_call,
        "steps": turns,
        "optimal_steps": task.get("optimal_steps"),
    }
