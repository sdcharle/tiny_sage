"""Model backends. Each returns a normalized ModelTurn so the runner doesn't care where the model lives.

llama-server and OpenRouter share one OpenAI-compatible client; only base_url, key and a few
OpenRouter-specific request fields differ. OracleBackend replays a task's known-good solution.
"""

import json
import time
from dataclasses import dataclass, field

import openai

RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


class BackendError(Exception):
    pass


@dataclass
class ModelTurn:
    content: str
    reasoning: str
    tool_calls: list[dict]          # [{"id", "name", "arguments"}], arguments as the raw string the model produced
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    provider: str | None = None
    latency_s: float = 0.0


class OpenAIBackend:
    def __init__(self, model: dict, backend_cfg: dict, api_key: str, max_attempts: int = 6):
        self.model = model
        self.is_openrouter = "openrouter.ai" in backend_cfg["base_url"]
        self.client = openai.OpenAI(base_url=backend_cfg["base_url"], api_key=api_key, timeout=600, max_retries=0)
        self.max_attempts = max_attempts

    def chat(self, messages, tools, temperature, max_tokens, extra_body=None) -> ModelTurn:
        body = dict(extra_body or {})
        headers = None
        if self.is_openrouter:
            body.setdefault("usage", {"include": True})
            if self.model.get("provider"):
                body["provider"] = {"order": [self.model["provider"]], "allow_fallbacks": False}
            headers = {"X-Title": "tiny_sage"}
        kwargs = dict(model=self.model["model_id"], messages=messages, temperature=temperature,
                      max_tokens=max_tokens, extra_body=body or None, extra_headers=headers)
        if tools:
            kwargs["tools"] = tools

        delay = 5.0
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                if not resp.choices:  # OpenRouter can return 200 with an error body
                    raise BackendError(f"empty choices: {getattr(resp, 'error', None) or resp.model_dump()}")
                return self._normalize(resp, time.perf_counter() - t0)
            except (*RETRYABLE, BackendError) as e:
                if attempt == self.max_attempts:
                    raise BackendError(f"{type(e).__name__}: {e}") from e
                print(f"    ! {type(e).__name__}, retrying in {delay:.0f}s ({attempt}/{self.max_attempts})")
                time.sleep(delay)
                delay = min(delay * 2, 120)
            except openai.APIStatusError as e:
                raise BackendError(f"{e.status_code}: {e.message}") from e

    @staticmethod
    def _normalize(resp, latency) -> ModelTurn:
        choice = resp.choices[0]
        msg = choice.message
        extra = msg.model_extra or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                 for tc in (msg.tool_calls or [])]
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            details = getattr(resp.usage, "completion_tokens_details", None)
            if details and getattr(details, "reasoning_tokens", None):
                usage["reasoning_tokens"] = details.reasoning_tokens
            cost = (resp.usage.model_extra or {}).get("cost")
            if cost is not None:
                usage["cost_usd"] = cost
        return ModelTurn(content=msg.content or "", reasoning=reasoning, tool_calls=calls,
                         finish_reason=choice.finish_reason, usage=usage,
                         provider=(resp.model_extra or {}).get("provider"), latency_s=latency)

    def list_models(self) -> list[dict]:
        return [m.model_dump() for m in self.client.models.list().data]


class OracleBackend:
    """Plays back task["oracle"]: one round of tool calls per turn, then the final answer."""

    def __init__(self, task: dict):
        self.oracle = task["oracle"]

    def chat(self, messages, tools, temperature, max_tokens, extra_body=None) -> ModelTurn:
        i = sum(m["role"] == "assistant" for m in messages)
        rounds = self.oracle["rounds"]
        if i < len(rounds):
            calls = [{"id": f"oracle_{i}_{k}", "name": c["name"], "arguments": json.dumps(c["args"])}
                     for k, c in enumerate(rounds[i])]
            return ModelTurn(content="", reasoning="", tool_calls=calls, finish_reason="tool_calls")
        return ModelTurn(content=self.oracle["final"], reasoning="", tool_calls=[], finish_reason="stop")


def make_backend(cfg: dict, model: dict, api_key: str) -> OpenAIBackend:
    return OpenAIBackend(model, cfg["backends"][model["backend"]], api_key)
