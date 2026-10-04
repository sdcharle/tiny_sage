import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness.backends import ModelTurn, OpenAIBackend, OracleBackend
from harness.config import load_config, load_tasks
from harness.grader import grade, match_value, normalize_final
from harness.runner import detect_text_tool_calls, run_task, split_thinking
from tasks.build import money_re, num_re, time_re
from tools import World

CFG = load_config()
SETTINGS = dict(CFG["defaults"])
MODEL = {"name": "test", "family": None, "backend": None, "model_id": "m", "thinking": {"off": {}, "on": {}}}
TASKS = {t["id"]: t for t in load_tasks()}


class Scripted:
    """Backend that returns pre-baked turns in order."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, *a, **k):
        self.seen.append(json.loads(json.dumps(messages)))
        return self.turns.pop(0)


def turn(content="", calls=()):
    return ModelTurn(content=content, reasoning="", finish_reason="tool_calls" if calls else "stop",
                     tool_calls=[{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)])


def run(task_id, *turns, **kw):
    return run_task(TASKS[task_id], Scripted(*turns), model=MODEL, mode="off", settings=SETTINGS, **kw)


# ---- config / tasks --------------------------------------------------------------

def test_on_off_keys_survive_yaml():
    assert set(CFG["defaults"]["max_tokens"]) == {"off", "on"}
    assert all(set(m["thinking"]) <= {"on", "off"} for m in CFG["models"].values())


def test_every_task_solved_by_its_oracle():
    for t in TASKS.values():
        rec = run_task(t, OracleBackend(t), model=MODEL, mode="off", settings=SETTINGS)
        assert rec["grade"]["success"], (t["id"], rec["grade"]["checks"])


# ---- patterns and matching -------------------------------------------------------

@pytest.mark.parametrize("x,text,ok", [
    (230.0, "that's 230.00 euros", True), (230.0, "about 2300", False), (213.06, "€213.06", True),
    (213.06, "213.1", False), (-2, "it's -2°c", True), (19, "19°c", True), (19, "119", False),
])
def test_num_re(x, text, ok):
    import re
    assert bool(re.search(num_re(x), normalize_final(text))) is ok


def test_money_re_accepts_rounded():
    import re
    assert any(re.search(p, "roughly 1266 dollars") for p in money_re(1265.82))


@pytest.mark.parametrize("hhmm,text", [("22:00", "it's 10 pm"), ("22:00", "22:00 JST"), ("06:00", "6:00 a.m."),
                                       ("18:30", "6:30pm")])
def test_time_re(hhmm, text):
    import re
    assert any(re.search(p, normalize_final(text)) for p in time_re(hhmm))


def test_match_value_specs():
    assert match_value({"$contains": "paris"}, "Paris, France")
    assert match_value({"$set": ["a@x.com", "b@x.com"]}, ["B@x.com", "a@x.com"])
    from harness.grader import MISSING
    assert match_value({"$default": 60}, MISSING) and match_value({"$default": 60}, 60)
    assert not match_value({"$default": 60}, 30)
    assert match_value({"$approx": 102.45}, 102.45) and not match_value({"$approx": 102.45}, "102.45")
    assert match_value(2500, 2500.0) and not match_value("USD", "EUR")


# ---- runner behaviour ------------------------------------------------------------

def test_structured_call_success():
    rec = run("T0-weather-00", turn(calls=[("get_weather", '{"city": "Paris"}')]), turn("14°C and partly cloudy."))
    assert rec["grade"]["success"] and rec["grade"]["valid_call_rate"] == 1.0
    assert rec["stop_reason"] == "final" and len(rec["turns"]) == 2


def test_text_tool_call_strict_vs_lenient():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    strict = run("T0-weather-00", turn(text))
    assert not strict["grade"]["success"] and strict["grade"]["failure_mode"] == "format_error"
    assert strict["calls"][0]["error_kind"] == "text_tool_call"
    lenient = run("T0-weather-00", turn(text), turn("It's 14°C."), lenient=True)
    assert lenient["grade"]["success"] and lenient["calls"][0]["text_format"]


def test_schema_and_json_errors_are_reported_back():
    b = Scripted(turn(calls=[("get_weather", '{"town": "Paris"}')]),
                 turn(calls=[("get_weather", '{"city": "Paris"')]),
                 turn(calls=[("get_weather", '{"city": "Paris"}')]), turn("14°C"))
    rec = run_task(TASKS["T0-weather-00"], b, model=MODEL, mode="off", settings=SETTINGS)
    kinds = [c["error_kind"] for c in rec["calls"]]
    assert kinds == ["schema", "bad_json", None]
    assert "Invalid arguments" in b.seen[1][-1]["content"]
    assert rec["grade"]["success"] and rec["grade"]["valid_call_rate"] == pytest.approx(1 / 3)


def test_unknown_tool_and_over_call():
    rec = run("T3-direct-01", turn(calls=[("search_web", '{"query": "capital of australia"}')]), turn("Canberra"))
    assert rec["grade"]["failure_mode"] == "over_call"
    rec = run("T0-weather-00", turn(calls=[("get_forecast", '{"city": "Paris"}')]), turn("no idea"))
    assert rec["calls"][0]["error_kind"] == "unknown_tool"


def test_clarification_with_allowed_lookup():
    rec = run("T3-clarify-07", turn(calls=[("find_user", '{"query": "Bob"}')]),
              turn("There are two Bobs: Bob Ng and Bob Okafor. Which one?"))
    assert rec["grade"]["success"]


def test_max_turns():
    looping = [turn(calls=[("get_weather", '{"city": "Paris"}')]) for _ in range(SETTINGS["max_turns"])]
    rec = run("T0-weather-00", *looping)
    assert rec["stop_reason"] == "max_turns" and rec["grade"]["failure_mode"] == "no_final"


def test_faults_then_recovery():
    rec = run("T5-retry-00", turn(calls=[("get_weather", '{"city": "Paris"}')]),
              turn(calls=[("get_weather", '{"city": "Paris"}')]), turn("14°C in Paris"))
    assert "503" in rec["calls"][0]["result"] and rec["grade"]["success"]


def test_hallucinating_through_outage_fails():
    calls = [turn(calls=[("get_weather", '{"city": "Tokyo"}')])] * 2
    rec = run("T5-down-00", *calls, turn("It's 16°C in Tokyo."))
    assert not rec["grade"]["success"]


def test_split_thinking_and_text_call_detection():
    assert split_thinking("<think>hmm</think>Answer") == ("Answer", "hmm")
    assert split_thinking("hmm</think>Answer") == ("Answer", "hmm")
    found = detect_text_tool_calls('```json\n{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}\n```', {"get_time"})
    assert found and found[0]["name"] == "get_time"
    assert detect_text_tool_calls('{"name": "nope"}', {"get_time"}) == []


def test_world_tools():
    w = World()
    assert w.call("calculator", {"expression": "2^3"})["error"].startswith("Unsupported")
    assert w.call("get_order", {"order_id": "ORD5003"})["error"].startswith("Invalid order id")
    assert w.call("get_time", {"timezone": "Asia/Tokyo"})["local_time"] == "2026-03-16 22:00"
    assert w.call("find_user", {"query": "bob"})["results"].__len__() == 2
    assert w.call("convert_currency", {"amount": 250, "from_currency": "USD", "to_currency": "EUR"})["result"] == 230.0


# ---- real OpenAI client against a fake OpenAI-compatible server -------------------

class FakeServer(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.requests.append(body)
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if has_tool_result:
            msg = {"role": "assistant", "content": "<think>done</think>It's 14°C and partly cloudy in Paris."}
            finish = "stop"
        else:
            msg = {"role": "assistant", "content": None, "reasoning_content": "need weather",
                   "tool_calls": [{"id": "call_1", "type": "function",
                                   "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}
            finish = "tool_calls"
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_end_to_end_through_openai_client():
    srv = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        backend = OpenAIBackend({"model_id": "fake-model"}, {"base_url": f"http://127.0.0.1:{srv.server_port}/v1"}, "k")
        model = {**MODEL, "thinking": {"on": {"system_prefix": "<|think|>\n", "extra_body": {"foo": 1}}}}
        rec = run_task(TASKS["T0-weather-00"], backend, model=model, mode="on", settings=SETTINGS)
    finally:
        srv.shutdown()
    assert rec["grade"]["success"], rec["grade"]
    assert rec["thinking_detected"] and rec["tokens"]["completion_tokens"] == 40
    first = FakeServer.requests[0]
    assert first["messages"][0]["content"].startswith("<|think|>") and first["foo"] == 1
    assert first["tools"][0]["function"]["name"] == "get_weather"
    second = FakeServer.requests[1]["messages"]
    assert second[2]["tool_calls"][0]["id"] == "call_1" and second[3]["role"] == "tool"
