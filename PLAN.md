# tiny_sage — Master Plan

## Goal

Find out how small a local model can get before it stops being a useful tool-calling agent, and whether the answer depends on the model's size or on which family it comes from.

We use two model families, each in several sizes, and run the same agent tasks on every model on one 16 GB laptop.

## Core questions

1. **Size curve:** How do tool-calling accuracy and task success change from about 2B to about 12B parameters?
2. **Family vs. size:** If two models are about the same size but come from different families (e.g. Gemma E4B vs. Qwen 4B), do they score differently? If so, family behavior is a confound.
3. **Thinking:** Does turning on native "thinking" help small models enough to cover a size gap?
4. **Practical floor:** What is the smallest model that is good enough for the tasks we care about, given the latency and memory a laptop allows?

## Model ladder

| Family | Model | Where it runs | Approx. 4-bit RAM / price | Notes |
|---|---|---|---|---|
| Gemma 4 | E2B | Local (llama-server) | ~1.5–2 GB | Smallest rung. Not on hosted providers yet. |
| Gemma 4 | E4B | Local (llama-server) | ~3 GB | Not on hosted providers yet. |
| Gemma 4 | 12B | Local (llama-server) | ~7–8 GB | Added to the lineup later. No hosted provider found. |
| Qwen 3.5 | 4B | Local (llama-server) | ~3 GB | Native thinking + tool calls, 256k context |
| Qwen 3.5 | 9B | Hosted (OpenRouter) | ~$0.08 / $0.13 per 1M tokens (in / out) | Reportedly strong for its size |
| Gemma 4 | 26B A4B (MoE) | Hosted (OpenRouter, free tier) | $0 (rate-limited) | **Ceiling reference.** Mixture-of-experts with ~4B active params. |
| Gemma 4 | 31B | Hosted (OpenRouter, free tier) | $0 (rate-limited) | **Ceiling reference** |

- Every Gemma 4 model has native tool calling. Gemma 4 is licensed under Apache 2.0, while Gemma 3 used a custom license.
- Qwen is the cross-check family because it has the best reputation for small-model tool calling.
- The two hosted Gemma models sit above the ladder. They show what "good" looks like on our task suite, so the small models have something to be measured against.
- **Before pulling anything:** check the model names, sizes, licenses and context lengths against the official model cards. The figures above come from secondary roundups.

**Matched pairs to compare:** Gemma E4B ↔ Qwen 4B, and Gemma 12B ↔ Qwen 9B.

## Infrastructure: hybrid local + hosted

- **Local: bare llama-server (llama.cpp).** Runs Gemma E2B, E4B, 12B and Qwen 4B (plus Qwen 9B for the parity check), one model per server process (OpenAI-compatible API at `http://127.0.0.1:8080/v1`). A script starts it with flags pinned in `models.yaml`: context 16384, full GPU offload, flash attention, `--jinja`, one slot, fixed seed, and a thinking-token budget. The GGUFs are Unsloth's Q4_K_M builds, kept in one shared models directory (`<models_dir>/<hf_repo>/<file>`) so other llama.cpp-based tools can use the same files and each model is downloaded once. Reportedly Unsloth's own tooling can point at existing GGUF folders like this, but that's unverified. This keeps the small models, which matter most for the "how small can it go" question, under identical conditions: the same quantization and the same machine.
- **Hosted: OpenRouter.** Runs Qwen 9B and the Gemma 26B and 31B reference models (OpenAI-compatible API, `https://openrouter.ai/api/v1`).
  - **Pin a single provider for each model** using OpenRouter's provider routing settings, and record which provider served each run. Current pins: Qwen 9B → `darkbloom/fp4` (4-bit, the closest match to the local models); the free Gemma models → `google-ai-studio` (their only provider). `python -m harness providers <model>` lists the options. Providers may serve different quantizations, so letting OpenRouter switch between them would add noise.
  - Free-tier models may log prompts. That's fine here, because all the data is synthetic.
- **Backend parity check:** run Qwen 9B on both local llama-server and OpenRouter. (Qwen 3.5 4B turned out not to be on OpenRouter; the 9B is the only model available on both backends and still fits in 16 GB at 4-bit.) If the two backends give clearly different scores for the same model, the local-vs-hosted split is skewing the comparison, and we need to account for it.
- **Budget:** about 12M tokens for the whole experiment, which comes to roughly **$1–5** in hosted costs. Set an OpenRouter credit limit of $10 as a guardrail.
- **Latency is only compared within one backend.** Hosted speed reflects the provider's servers, not the model, so it isn't comparable with laptop speed.

## Controlled variables

Hold these constant across all models:

- **Quantization:** Q4_K_M (or the closest equivalent) for every local model. As a spot check, run one Q8 model to measure how much quantization costs. For hosted models, record the pinned provider and the quantization it reports.
- **Runtime:** llama-server for local models and OpenRouter for hosted ones, as described in the infrastructure section above. Record the exact llama-server command line and version (done automatically in `meta.json`) and the OpenRouter provider for every run.
- **Sampling:** temperature 0 for the main runs. A secondary run at T=0.7 with N samples measures how consistent each model is.
- **Context window:** pick one fixed setting, e.g. 16k, and use it everywhere, even where a model supports more.
- **Prompts:** use the same system prompt and tool schemas for every model. Each model's own chat template handles the formatting.
- **Hardware:** the same machine, plugged in, with other heavy apps closed.

## Task suite

Tasks are grouped into tiers of increasing difficulty. Every task must have a programmatic pass/fail check.

| Tier | What it tests | Example |
|---|---|---|
| T0 – Format | Emits a valid tool call that matches the schema | "What's the weather in Paris?" → `get_weather(city="Paris")` |
| T1 – Selection | Chooses the right tool out of 5–10 | Calculator vs. search vs. file read |
| T2 – Arguments | Fills arguments correctly, including types, enums and optional fields | Date parsing, nested objects |
| T3 – Restraint | Makes no tool call when none is needed, and asks for clarification when input is missing | Chit-chat, ambiguous requests |
| T4 – Multi-step | Chains 2–5 calls and passes outputs from one call into the next | Look up an ID, then fetch details, then compute |
| T5 – Recovery | Handles tool errors, empty results and malformed responses | Tool returns 404; the model should retry or adapt |
| T6 – Long-context | Picks out the relevant facts from long tool outputs | A 10k-token document followed by a pointed question |

Target **30–50 tasks per tier**. Each tier starts with a small hand-written seed set and is then expanded with templated variations.

The tools are **deterministic mocks** (fake weather, a fake filesystem, a fake DB, a real calculator), so every result can be reproduced and graded automatically.

## Metrics

Per model, per tier:

- **Valid-call rate:** the share of tool calls that parse and match the schema.
- **Tool selection accuracy**
- **Argument match:** whether the required calls' arguments match the expected values. Each task's spec sets how strict the match is: exact, contains, any-of, approximate number or unordered set.
- **Task success rate:** whether the final outcome is correct. This is the headline metric.
- **Over-calling / under-calling rate:** for T3.
- **Steps to completion:** compared with the optimal number of steps.
- **Latency:** tokens/sec and total wall-clock time per task. Requests are not streamed, so time to first token is not measured.
- **Peak memory**

Each model is run twice, with thinking **on** and **off**, wherever the model supports both.

## Architecture

```
tiny_sage/
├── PLAN.md
├── models.yaml          # ladder: name, backend (llamacpp|openrouter), model id, provider pin, quant, thinking support
├── .env                 # OPENROUTER_API_KEY (gitignored)
├── tools/               # mock tool implementations + JSON schemas
├── tasks/               # task definitions (yaml/jsonl), one file per tier
├── harness/
│   ├── runner.py        # agent loop: prompt → model → tool exec → repeat
│   ├── backends.py      # one OpenAI-compatible client; base_url + key switch between llama-server and OpenRouter
│   ├── grader.py        # per-task checkers
│   └── metrics.py       # timing + memory capture
├── results/             # raw JSONL transcripts, one per (model, config, run)
└── analysis/            # notebooks / scripts → tables + charts
```

- **Agent loop:** a minimal hand-written loop rather than an agent framework, so that framework behavior does not affect the results. Max 10 turns per task.
- **Logging:** every message, tool call, raw model output, token count and timing goes into JSONL, so results can be re-graded later without re-running the models.
- **Parsing:** use the backend's native tool-call parsing first. Every raw output that fails to parse is logged, so we can tell whether a failure came from the model or from the template.

## Phases

### Phase 0 — Setup (½ day)
- [ ] `brew install llama.cpp`, then `python -m harness download --models local` to fetch the five Unsloth Q4_K_M GGUFs. (The repos and files are confirmed to exist on Hugging Face.)
- [ ] Create an OpenRouter account, add a $10 credit limit and put the API key in `.env`.
- [x] Confirm the OpenRouter model IDs for Qwen 3.5 9B and Gemma 4 26B and 31B, and pick a provider pin for each. (Qwen 3.5 4B is not hosted, so the parity check moved to the 9B.)
- [ ] Smoke test: `python -m harness smoke --models all --thinking both`.
- [ ] Confirm that thinking toggles on and off for every model (`smoke --manage-server`). The switches in `models.yaml` are best guesses.
- [ ] Check that tool calls come back as structured `tool_calls` from both llama-server and OpenRouter, and not as raw text.

### Phase 1 — Harness (1–2 days)
- [x] Mock tools + schemas.
- [x] Agent loop runner with JSONL logging.
- [x] Grader framework.
- [x] `models.yaml` + a CLI along the lines of `run --model X --tier T1 --thinking off`.

### Phase 2 — Task suite (2–3 days)
- [x] Write the tasks: 224 across 7 tiers, generated by `tasks/build.py` with answers computed from the mock data. Every task's oracle solution passes (`python -m harness selftest`).
- [ ] Calibrate: run `--limit 10` against a strong model and hand-check failures for broken tasks or graders.
- [x] Expand each tier to 30–50 tasks.

### Phase 3 — Main runs (1–2 days, mostly unattended)
- [ ] Full matrix: 8 models (5 ladder + 2 reference + 1 parity) × 7 tiers × thinking on/off at T=0.
- [ ] Parity check: Qwen 9B on both local llama-server and OpenRouter.
- [ ] Consistency run: T=0.7, N=5 on a subset.
- [ ] Quant spot-check: one model at Q8.

### Phase 4 — Analysis (1–2 days)

`python -m harness analyze` builds everything below. The charts and tables were tested on synthetic results; they still need checking against real data.

- [ ] Success-vs-size curve, one line per family.
- [ ] Matched-pair comparison table.
- [ ] Heatmap of failure modes by model and tier.
- [ ] Thinking on/off delta.
- [ ] Latency vs. accuracy scatter, used to answer the practical floor question.

### Phase 5 — Write-up
- [ ] Summary of findings, including a recommendation for the smallest viable model per use case.
- [ ] Hand-pick a few transcripts that show typical failures of the small models.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Chat template or tool-parsing bugs in the backend look like model failures | Log raw output. Hand-inspect a sample of failures. Run the Qwen 9B parity check on both backends. |
| A hosted provider changes the quantization or model version mid-experiment | Pin the provider, log the provider and model ID for every run, and run each model's full matrix in a single session. |
| Free-tier rate limits stall the runs | Add retry with backoff to the runner. If that's still too slow, switch to the cheap paid version of the model. |
| A task suite that is too easy or too hard makes every model look the same | Calibrate on the seed set so the tiers spread from about 90% down to about 20% for the 12B model. |
| Only two families, so conclusions may not generalize | State this limitation openly. Add a third family later if the result is surprising (Llama, Phi or SmolLM, for example). |
| Thinking mode blows up latency or context | Cap thinking tokens and record the cap. |
| Thermal throttling skews latency on a laptop | Run models in sequence, add a cool-down between runs, and record timings while plugged in. |

## Success criteria

- Every model is run on the full matrix and the results can be reproduced from a single command.
- We can make a clear, data-backed statement about how much model size matters compared with model family for tool calling at this scale.
- We can name a concrete recommendation along the lines of: "for single-step tool use, X is enough; for multi-step with recovery, you need at least Y".

## Open decisions

- [x] Backend: hybrid. Bare llama-server, started from a script with pinned flags, runs the small models locally and OpenRouter runs the larger ones (see the infrastructure section).
- [ ] Should the task suite be fully synthetic, or include a real-world flavor such as tasks shaped like an existing personal assistant?
- [ ] Add a third family now, or only if the results call for it?
