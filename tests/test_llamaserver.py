"""Exercise the llama-server launcher with a fake `llama-server` executable (no real model needed)."""

import json
import socket
import stat
import sys
import textwrap

import pytest

from harness import llamaserver
from harness import run as runmod
from harness.config import load_config

FAKE_SERVER = textwrap.dedent('''\
    #!{python}
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    if "--version" in args:
        print("version: 9999 (fake)"); sys.exit(0)
    get = lambda f: args[args.index(f) + 1]
    port, alias = int(get("--port")), get("--alias")
    open(get("--log-file"), "w").write(json.dumps(args))

    class H(BaseHTTPRequestHandler):
        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self._send({{"status": "ok"}} if self.path == "/health" else {{"data": [{{"id": alias}}]}})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if any(m["role"] == "tool" for m in body["messages"]):
                msg, fin = {{"role": "assistant", "content": "It's 14°C in Paris."}}, "stop"
            else:
                msg = {{"role": "assistant", "content": None, "tool_calls": [{{"id": "c1", "type": "function",
                       "function": {{"name": "get_weather", "arguments": json.dumps({{"city": "Paris"}})}}}}]}}
                fin = "tool_calls"
            self._send({{"id": "x", "object": "chat.completion", "created": 0, "model": alias,
                        "choices": [{{"index": 0, "message": msg, "finish_reason": fin}}],
                        "usage": {{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}}})
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE_SERVER.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    models = tmp_path / "gguf"
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(exe))
    monkeypatch.setenv("TINY_SAGE_MODELS_DIR", str(models))
    cfg = load_config()
    port = _free_port()
    cfg["llama_server"]["port"] = port
    cfg["backends"]["llamacpp"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    cfg["defaults"]["cooldown_s"] = 0
    model = cfg["models"]["gemma4-e4b"]
    gguf = llamaserver.gguf_path(cfg, model)
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(runmod, "RESULTS_DIR", tmp_path / "results")
    return cfg, model, tmp_path


def test_command_has_pinned_flags(env):
    cfg, model, tmp = env
    cmd = llamaserver.command(cfg, model)
    assert cmd[cmd.index("--ctx-size") + 1] == "16384"
    assert cmd[cmd.index("--alias") + 1] == "gemma4-e4b"
    assert "--jinja" in cmd and cmd[cmd.index("--flash-attn") + 1] == "on"
    assert cmd[cmd.index("--model") + 1].endswith("unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")
    assert llamaserver.version(cfg) == "version: 9999 (fake)"


def test_missing_model_file_is_a_clear_error(env):
    cfg, _, _ = env
    with pytest.raises(SystemExit, match="download"):
        llamaserver.command(cfg, cfg["models"]["qwen35-4b"])


def test_manage_server_runs_tasks_and_stops(env):
    cfg, model, tmp = env
    tasks = runmod.select_tasks(["T0"], ["T0-weather-00"])
    runmod.run_matrix(cfg, [model], tasks, ["off"], tag="t", manage_server=True)
    assert llamaserver.served_models(cfg) is None  # stopped afterwards
    rec = json.loads((tmp / "results/t/gemma4-e4b/think-off__t0__s0.jsonl").read_text())
    assert rec["grade"]["success"] and rec["model_id"] == "gemma4-e4b"
    meta = json.loads((tmp / "results/t/meta.json").read_text())["runs"][-1]
    assert meta["llama_server_version"] == "version: 9999 (fake)"
    assert "--ctx-size" in meta["server_commands"]["gemma4-e4b"]


def test_run_refuses_wrong_model_on_server(env):
    cfg, model, tmp = env
    other = cfg["models"]["gemma4-e2b"]
    gguf = llamaserver.gguf_path(cfg, other)
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    tasks = runmod.select_tasks(["T0"], ["T0-weather-00"])
    with llamaserver.LlamaServer(cfg, other, tmp / "logs"):
        with pytest.raises(SystemExit, match="not gemma4-e4b"):
            runmod.run_matrix(cfg, [model], tasks, ["off"], tag="t2")
