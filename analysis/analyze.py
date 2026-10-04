"""Turn results/<tag>/ transcripts into tables (summary.md) and charts (PNG) under analysis/out/<tag>/.

Run: python -m harness analyze --tag main
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from harness.config import RESULTS_DIR, ROOT, TIERS, load_config

OUT_ROOT = ROOT / "analysis" / "out"

# Reference palette: categorical slot 1 (blue) and 2 (orange); text and grid in neutral ink.
FAMILY_COLORS = {"gemma4": "#2a78d6", "qwen35": "#eb6834"}
FAMILY_LABELS = {"gemma4": "Gemma 4", "qwen35": "Qwen 3.5"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MATCHED_PAIRS = [("gemma4-e4b", "qwen35-4b"), ("gemma4-12b", "qwen35-9b")]
PARITY = ("qwen35-9b-local", "qwen35-9b")  # local vs hosted

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


# ---- loading --------------------------------------------------------------------

def load(tag: str) -> pd.DataFrame:
    cfg = load_config()
    rows = []
    for path in sorted((RESULTS_DIR / tag).rglob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            g = r["grade"]
            m = cfg["models"].get(r["model"], {})
            pos = next((t.split("=")[1] for t in r.get("tags", []) if t.startswith("pos=")), None)
            mem = r.get("memory") or {}
            rows.append({
                "model": r["model"], "family": m.get("family", r.get("family")), "role": m.get("role", "?"),
                "params_b": m.get("params_b"), "backend": r.get("backend"), "thinking": r["thinking"],
                "temperature": r["temperature"], "sample": r["sample"], "tier": r["tier"], "task_id": r["task_id"],
                "success": g["success"], "failure_mode": g["failure_mode"] or "ok",
                "n_calls": g["n_calls"], "valid_call_rate": g["valid_call_rate"], "selection_ok": g["selection_ok"],
                "args_match_rate": g["args_match_rate"], "over_call": g["over_call"], "under_call": g["under_call"],
                "steps": g["steps"], "optimal_steps": g["optimal_steps"], "wall_s": r["timing"]["wall_s"],
                "tokens_per_s": r.get("tokens_per_s"), "completion_tokens": r["tokens"]["completion_tokens"],
                "reasoning_tokens": r["tokens"].get("reasoning_tokens", 0), "cost_usd": r.get("cost_usd", 0),
                "peak_mem_mb": mem.get("peak_server_rss_mb"), "thinking_detected": r.get("thinking_detected"),
                "pos": pos, "providers": ",".join(r.get("providers", [])),
            })
    if not rows:
        raise SystemExit(f"No results under {RESULTS_DIR / tag}")
    df = pd.DataFrame(rows)
    # Task ids carry the T6 position in their tags only through the task file, so look it up there.
    if df["pos"].isna().all():
        df["pos"] = df["task_id"].map(_positions())
    return df


def _positions() -> dict:
    out = {}
    path = ROOT / "tasks" / "T6.jsonl"
    if path.exists():
        for line in path.open():
            t = json.loads(line)
            out[t["id"]] = next((x.split("=")[1] for x in t.get("tags", []) if x.startswith("pos=")), None)
    return out


def main_runs(df: pd.DataFrame) -> pd.DataFrame:
    """The headline condition: lowest temperature, first sample."""
    t = df["temperature"].min()
    return df[(df["temperature"] == t) & (df["sample"] == 0)]


def model_order(df: pd.DataFrame) -> list[str]:
    meta = df.drop_duplicates("model").set_index("model")
    role_rank = {"ladder": 0, "parity": 1, "reference": 2}
    return sorted(meta.index, key=lambda m: (role_rank.get(meta.loc[m, "role"], 3), meta.loc[m, "family"] or "",
                                             meta.loc[m, "params_b"] or 0))


# ---- tables ---------------------------------------------------------------------

def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


def md_table(df: pd.DataFrame, floatfmt=pct) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def success_by_tier(d: pd.DataFrame, order) -> pd.DataFrame:
    t = d.pivot_table(index=["model", "thinking"], columns="tier", values="success", aggfunc="mean")
    t["all"] = d.groupby(["model", "thinking"])["success"].mean()
    t["n"] = d.groupby(["model", "thinking"])["success"].size()
    t = t.reindex([c for c in TIERS if c in t.columns] + ["all", "n"], axis=1)
    t = t.reindex([(m, th) for m in order for th in ("off", "on") if (m, th) in t.index])
    t.index = [f"{m} ({th})" for m, th in t.index]
    t.index.name = "model (thinking)"
    t["n"] = t["n"].astype(int).astype(str)
    return t


def metric_table(d: pd.DataFrame, order) -> pd.DataFrame:
    rows = {}
    for (m, th), g in d.groupby(["model", "thinking"]):
        calls = g[g["n_calls"] > 0]
        rows[f"{m} ({th})"] = {
            "task success": g["success"].mean(),
            "valid-call rate": (calls["valid_call_rate"] * calls["n_calls"]).sum() / max(calls["n_calls"].sum(), 1),
            "selection (T1)": g.loc[g["tier"] == "T1", "selection_ok"].astype(float).mean(),
            "args match (T2)": g.loc[g["tier"] == "T2", "args_match_rate"].mean(),
            "over-call (T3)": g.loc[g["tier"] == "T3", "over_call"].mean(),
            "under-call": g.loc[g["tier"] != "T3", "under_call"].mean(),
            "steps / optimal": f"{(g['steps'] / g['optimal_steps']).loc[g['success']].mean():.2f}",
            "median s/task": f"{g['wall_s'].median():.1f}",
            "tok/s": f"{g['tokens_per_s'].median():.0f}" if g["tokens_per_s"].notna().any() else "",
            "peak mem GB": f"{g['peak_mem_mb'].max() / 1024:.1f}" if g["peak_mem_mb"].notna().any() else "",
            "thinking seen": g["thinking_detected"].mean(),
        }
    t = pd.DataFrame(rows).T
    t = t.reindex([f"{m} ({th})" for m in order for th in ("off", "on") if f"{m} ({th})" in t.index])
    t.index.name = "model (thinking)"
    return t


def pair_table(d: pd.DataFrame, a: str, b: str) -> pd.DataFrame | None:
    sub = d[d["model"].isin([a, b]) & (d["thinking"] == "off")]
    if sub["model"].nunique() < 2:
        return None
    t = sub.pivot_table(index="tier", columns="model", values="success", aggfunc="mean")[[a, b]]
    t.loc["all"] = sub.groupby("model")["success"].mean()[[a, b]]
    t[f"{a} − {b}"] = (t[a] - t[b]).map(lambda x: f"{100 * x:+.0f} pts")
    return t


def thinking_delta(d: pd.DataFrame, order) -> pd.DataFrame | None:
    t = d.pivot_table(index=["model", "tier"], columns="thinking", values="success", aggfunc="mean")
    if not {"on", "off"} <= set(t.columns):
        return None
    delta = (t["on"] - t["off"]).unstack("tier")
    delta["all"] = d[d["thinking"] == "on"].groupby("model")["success"].mean() - \
        d[d["thinking"] == "off"].groupby("model")["success"].mean()
    delta = delta.dropna(how="all").reindex([m for m in order if m in delta.index])
    delta.index.name = "model"
    return delta.apply(lambda col: col.map(lambda x: "" if pd.isna(x) else f"{100 * x:+.0f}"))


def consistency(df: pd.DataFrame) -> pd.DataFrame | None:
    hot = df[df["temperature"] > df["temperature"].min()]
    if hot.empty or hot["sample"].nunique() < 2:
        return None
    per_task = hot.groupby(["model", "thinking", "task_id"])["success"].agg(["mean", "size"])
    per_run = hot.groupby(["model", "thinking", "sample"])["success"].mean()
    out = pd.DataFrame({
        "samples": per_task["size"].groupby(level=[0, 1]).max(),
        "mean success": per_run.groupby(level=[0, 1]).mean(),
        "sd across samples": per_run.groupby(level=[0, 1]).std().map(lambda x: f"{100 * x:.1f} pts"),
        "tasks always same outcome": per_task["mean"].isin([0, 1]).groupby(level=[0, 1]).mean(),
    })
    out.index = [f"{m} ({th})" for m, th in out.index]
    out.index.name = "model (thinking)"
    out["samples"] = out["samples"].astype(int).astype(str)
    return out


# ---- charts ---------------------------------------------------------------------

def _label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", va="center", fontsize=9, color=INK_2)


def _log_size_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1.6, 48)
    ticks = [2, 4, 8, 16, 32]
    ax.set_xticks(ticks, [f"{t}B" for t in ticks])
    ax.minorticks_off()


def chart_size_curve(d: pd.DataFrame, out: Path, tier: str | None = None, ax=None):
    sub = d[d["thinking"] == "off"]
    if tier:
        sub = sub[sub["tier"] == tier]
    agg = sub.groupby("model").agg(success=("success", "mean"), params_b=("params_b", "first"),
                                   family=("family", "first"), role=("role", "first")).reset_index()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fam, g in agg.groupby("family"):
        color = FAMILY_COLORS.get(fam, INK_2)
        lad = g[g["role"] == "ladder"].sort_values("params_b")
        ax.plot(lad["params_b"], lad["success"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=FAMILY_LABELS.get(fam, fam), zorder=3)
        ref = g[g["role"] == "reference"]
        if not ref.empty:
            ax.plot(ref["params_b"], ref["success"], ls="none", marker="o", ms=7, markerfacecolor=SURFACE,
                    markeredgecolor=color, markeredgewidth=2, zorder=3,
                    label=f"{FAMILY_LABELS.get(fam, fam)} reference" if own else None)
        if own:
            # Gemma labels sit below their points and Qwen labels above, so neighbours at ~4B don't collide.
            dy = -11 if fam == "gemma4" else 11
            for k, (_, r) in enumerate(pd.concat([lad, ref.sort_values("params_b")]).iterrows()):
                flip = -1 if (r["role"] == "reference" and k % 2) else 1
                _label_end(ax, r["params_b"], r["success"], f"{r['model']}  {r['success']:.0%}", color, dy * flip)
    _log_size_axis(ax)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if own:
        ax.set_xlabel("parameters (log scale; E-models use effective params, 26B is MoE with ~4B active)")
        ax.set_ylabel("task success")
        ax.set_title("Task success vs. model size (thinking off, all tiers)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "size_curve.png", dpi=160)
        plt.close(fig)
    else:
        ax.set_title(tier, fontsize=10)


def chart_tier_curves(d: pd.DataFrame, out: Path):
    tiers = [t for t in TIERS if t in set(d["tier"])]
    cols = 4
    rows = -(-len(tiers) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharey=True)
    axes = axes.flatten()
    for ax, tier in zip(axes, tiers):
        chart_size_curve(d, out, tier=tier, ax=ax)
    for ax in axes[len(tiers):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels))
    fig.suptitle("Task success vs. size, per tier (thinking off; hollow = reference models)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "tier_curves.png", dpi=160)
    plt.close(fig)


def chart_failure_heatmap(d: pd.DataFrame, out: Path, order):
    fails = d[(~d["success"]) & (d["thinking"] == "off")]
    if fails.empty:
        return
    modes = ["api_error", "under_call", "over_call", "format_error", "wrong_tool", "bad_args", "no_final", "wrong_answer"]
    t = pd.crosstab(fails["model"], fails["failure_mode"]).reindex(columns=modes, fill_value=0)
    t = t.reindex([m for m in order if m in t.index])
    share = t.div(d[d["thinking"] == "off"].groupby("model").size().reindex(t.index), axis=0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.6))
    ax.imshow(share.values, cmap=cmap, vmin=0, vmax=max(share.values.max(), 0.01), aspect="auto")
    ax.set_xticks(range(len(modes)), [m.replace("_", " ") for m in modes], rotation=30, ha="right")
    ax.set_yticks(range(len(t)), t.index)
    ax.grid(False)
    vmax = share.values.max()
    for i in range(len(t)):
        for j in range(len(modes)):
            v = share.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.55 * vmax else INK)
    ax.set_title("How models fail: share of all tasks, by failure mode (thinking off)")
    fig.tight_layout()
    fig.savefig(out / "failure_heatmap.png", dpi=160)
    plt.close(fig)


def chart_thinking(d: pd.DataFrame, out: Path, order):
    t = d.groupby(["model", "thinking"])["success"].mean().unstack()
    if not {"on", "off"} <= set(t.columns):
        return
    t = t.dropna().reindex([m for m in order if m in t.dropna().index])
    fam = d.drop_duplicates("model").set_index("model")["family"]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(t) + 1.4))
    for i, (m, r) in enumerate(t.iterrows()):
        color = FAMILY_COLORS.get(fam[m], INK_2)
        ax.plot([r["off"], r["on"]], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(r["off"], i, "o", ms=7, markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2, zorder=2)
        ax.plot(r["on"], i, "o", ms=7, color=color, markeredgecolor=color, markeredgewidth=2, zorder=3)
        ax.annotate(f"{100 * (r['on'] - r['off']):+.0f} pts", (max(r["on"], r["off"]), i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(t)), t.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.plot([], [], "o", markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=2, label="thinking off")
    ax.plot([], [], "o", color=INK_2, label="thinking on")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.set_title("Does thinking help? Task success, off → on", pad=24)
    fig.tight_layout()
    fig.savefig(out / "thinking_delta.png", dpi=160)
    plt.close(fig)


def chart_latency(d: pd.DataFrame, out: Path):
    local = d[d["backend"] == "llamacpp"]
    if local.empty:
        return
    agg = local.groupby(["model", "thinking"]).agg(success=("success", "mean"), wall=("wall_s", "median"),
                                                   family=("family", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for _, r in agg.iterrows():
        color = FAMILY_COLORS.get(r["family"], INK_2)
        filled = r["thinking"] == "on"
        ax.plot(r["wall"], r["success"], "o", ms=8, color=color if filled else SURFACE,
                markeredgecolor=color, markeredgewidth=2)
        _label_end(ax, r["wall"], r["success"], f"{r['model']} ({r['thinking']})", color)
    for fam, c in FAMILY_COLORS.items():
        if fam in set(agg["family"]):
            ax.plot([], [], "o", color=c, label=FAMILY_LABELS[fam])
    ax.legend(loc="lower right", title="filled = thinking on", title_fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("median seconds per task (local models only, log scale)")
    ax.set_ylabel("task success")
    ax.set_title("Accuracy vs. latency on the laptop")
    fig.tight_layout()
    fig.savefig(out / "latency_vs_accuracy.png", dpi=160)
    plt.close(fig)


# ---- main -----------------------------------------------------------------------

def main(tag: str = "main"):
    df = load(tag)
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_results.csv", index=False)
    d = main_runs(df)
    order = model_order(d)

    md = [f"# tiny_sage results: `{tag}`", "",
          f"{len(df)} task runs, {df['model'].nunique()} models. Headline tables use temperature "
          f"{d['temperature'].min():g}, sample 0.", ""]
    md += ["## Task success by tier", "", md_table(success_by_tier(d, order)), "",
           "![size curve](size_curve.png)", "", "![tier curves](tier_curves.png)", ""]
    md += ["## Metrics", "", md_table(metric_table(d, order)), "",
           "Valid-call rate counts every tool-call attempt, including ones written as text. "
           "Latency is only comparable within one backend.", ""]

    md += ["## Matched pairs (thinking off)", ""]
    for a, b in MATCHED_PAIRS:
        t = pair_table(d, a, b)
        md += [f"### {a} vs {b}", "", md_table(t) if t is not None else "_not enough data yet_", ""]

    t = pair_table(d, *PARITY)
    md += ["## Backend parity: same model, local llama-server vs OpenRouter", "",
           md_table(t) if t is not None else "_not enough data yet_", "",
           "A large gap here means the local-vs-hosted split is affecting the comparison.", ""]

    t = thinking_delta(d, order)
    md += ["## Thinking on − off (percentage points)", "",
           md_table(t, floatfmt=str) if t is not None else "_no paired on/off runs yet_", "",
           "![thinking](thinking_delta.png)", ""]

    md += ["## Failure modes", "", "![failures](failure_heatmap.png)", ""]
    fm = pd.crosstab(d["model"], d["failure_mode"]).reindex([m for m in order if m in set(d["model"])])
    fm.index.name = "model"
    md += [md_table(fm, floatfmt=str), ""]

    t6 = d[(d["tier"] == "T6") & d["pos"].notna()]
    if not t6.empty:
        pos = t6.pivot_table(index="model", columns="pos", values="success", aggfunc="mean")
        pos = pos.reindex([m for m in order if m in pos.index])
        pos.columns = [f"needle at {float(c):.0%}" for c in pos.columns]
        pos.index.name = "model"
        md += ["## Long context: success by needle position (T6)", "", md_table(pos), ""]

    t = consistency(df)
    if t is not None:
        md += ["## Consistency at higher temperature", "", md_table(t), ""]

    md += ["## Accuracy vs latency", "", "![latency](latency_vs_accuracy.png)", ""]

    cost = df.groupby("model")["cost_usd"].sum()
    if cost.sum() > 0:
        md += ["## Hosted cost", "", *[f"- {m}: ${c:.4f}" for m, c in cost.items() if c > 0],
               f"- **total: ${cost.sum():.4f}**", ""]
    provs = df[df["providers"] != ""].groupby("model")["providers"].agg(lambda s: ", ".join(sorted(set(",".join(s).split(",")))))
    if not provs.empty:
        md += ["## OpenRouter providers that served each model", "", *[f"- {m}: {p}" for m, p in provs.items()], ""]

    chart_size_curve(d, out)
    chart_tier_curves(d, out)
    chart_failure_heatmap(d, out, order)
    chart_thinking(d, out, order)
    chart_latency(d, out)
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out.relative_to(ROOT)}/summary.md, charts and all_results.csv")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
