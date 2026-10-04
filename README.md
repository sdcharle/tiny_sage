# tiny_sage

How small can a local model get and still be a useful tool-calling agent? And is that about size, or about the model family?

The plan behind this is in [PLAN.md](PLAN.md). This file covers how to run things.

## Setup

```sh
uv sync                      # install dependencies
cp .env.example .env         # then paste your OpenRouter key into .env
uv run python -m tasks.build # build the 224-task suite into tasks/T*.jsonl
uv run python -m harness selftest   # every task's known-good solution must pass
uv run pytest -q
```

**Local models: bare llama-server (llama.cpp)**
1. Install llama.cpp: `brew install llama.cpp`. This gives you `llama-server`.
2. Download the five Unsloth Q4_K_M GGUFs (Gemma 4 E2B, E4B and 12B; Qwen 3.5 4B and 9B) into the shared models directory:
   ```sh
   uv run python -m harness download --models local
   ```
   The directory is `~/models/gguf` by default (`llama_server.models_dir` in `models.yaml`, or set `TINY_SAGE_MODELS_DIR`). Files go in `<models_dir>/<hf_repo>/<file>.gguf`, so any other llama.cpp-based tool can point at the same files and each model is only downloaded once.
3. One model runs at a time. llama-server always starts with the same pinned flags from `models.yaml`: context 16384, all layers on the GPU, flash attention on, `--jinja`, one slot, a fixed seed, thinking parsed into `reasoning_content`, and a 3072-token thinking cap.
   - **By hand:** `scripts/serve.sh gemma4-e4b` prints the exact command and runs it in the foreground.
   - **Automatically:** pass `--manage-server` to `run` or `smoke`. Each local model is started in turn, checked for health, run, and stopped. The exact command line and `llama-server --version` are recorded in `results/<tag>/meta.json`, and server logs go to `results/<tag>/logs/`.

## Check the setup

```sh
uv run python -m harness list                          # models, whether each GGUF is downloaded, task counts
uv run python -m harness providers qwen/qwen3.5-9b     # OpenRouter providers, quantization, price -> pick a pin
uv run python -m harness smoke --models all --thinking both --manage-server
```

`smoke` runs one easy task per model and thinking mode. It tells you:
- whether tool calls come back **structured** (if not, the model's chat template isn't handling tools in that backend),
- whether **thinking actually toggles**. It warns if thinking shows up when it should be off, or the other way round. Local models switch thinking with `chat_template_kwargs: {enable_thinking: ...}`, which llama-server passes to the model's chat template. That's a best guess for Gemma 4 and Qwen 3.5, so this check matters.

## Run

```sh
# Calibration (Phase 2): 10 tasks per tier on one strong model
uv run python -m harness run --models qwen35-9b --limit 10 --tag calib

# Main matrix: everything, thinking off and on, temperature 0
uv run python -m harness run --models all --manage-server --workers 4

# Consistency: temperature 0.7, 5 samples, on a subset
uv run python -m harness run --models ladder --temperature 0.7 --samples 5 --tiers T1,T4 --tag consistency

uv run python -m harness analyze            # -> analysis/out/main/summary.md + charts
```

- Runs **resume**: rerun the same command after a crash or rate limit and it skips finished tasks. Use `--overwrite` to start over.
- `--workers` only applies to hosted models. Local models always run one task at a time.
- Without `--manage-server`, `run` uses the llama-server that's already running, and refuses to start if it's serving a different model from the one requested.
- `--lenient` also executes tool calls the model wrote as plain text instead of structured calls. By default these count as format errors.
- Model groups for `--models`: `all`, `ladder`, `reference`, `parity`, `local`, `hosted`, or comma-separated names.

## What's where

| Path | What |
|---|---|
| `models.yaml` | The ladder (GGUF files or OpenRouter IDs, provider pins, thinking switches), pinned llama-server flags, defaults |
| `tools/` | 13 deterministic mock tools (weather, calculator, search, files, users, orders, currency, time, stocks, email, calendar), plus fault injection and long-document generators |
| `tasks/build.py` | Generates the task suite. Expected answers are computed from the mock data. Each task has an oracle solution. |
| `harness/runner.py` | The agent loop: no framework, max 10 turns, every message and call logged |
| `harness/grader.py` | Grades a transcript against the task's `expect` spec. `regrade` reruns it on saved transcripts. |
| `harness/backends.py` | One OpenAI-compatible client for llama-server and OpenRouter |
| `harness/llamaserver.py`, `scripts/serve.sh` | Builds the pinned llama-server command; starts, health-checks and stops it |
| `analysis/analyze.py` | Tables and charts |
| `results/<tag>/<model>/think-<mode>__t<temp>__s<n>.jsonl` | Raw transcripts, one line per task |

## Tiers

| Tier | Tests | Tasks |
|---|---|---|
| T0 | Valid, schema-matching call (only the one relevant tool offered) | 37 |
| T1 | Picking the right tool out of 13 | 33 |
| T2 | Arguments: relative dates, enums, number words, arrays, optional fields | 34 |
| T3 | Restraint: answering directly, or asking when information is missing | 29 |
| T4 | Multi-step chains (2–4 dependent calls) | 31 |
| T5 | Recovery: transient errors, typos with hints, garbage responses, empty results, a service that stays down | 30 |
| T6 | Long context: ~10k-token logs, handbooks and CSVs, with the answer at 10%, 50% or 90% of the way in and look-alike distractors | 30 |

The fixed "now" for every task is Monday 2026-03-16 13:00 UTC.
