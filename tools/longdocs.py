"""Deterministic long-document generators for T6 (long-context) tasks.

Each generator takes plain params and returns (text, info). tasks/build.py calls them to work out
expected answers; World calls them again with the same params to materialize the file.
"""

import random
from datetime import datetime, timedelta

CHARS_PER_TOKEN = 3  # conservative: logs and CSVs are digit-heavy and tokenize densely

COMPONENTS = ["api", "worker-1", "worker-2", "worker-3", "scheduler", "cache", "auth", "billing"]
INFO_MSGS = [
    "Processed batch {n} in {ms}ms",
    "GET /v1/orders/{n} 200 {ms}ms",
    "POST /v1/checkout 201 {ms}ms",
    "Cache hit ratio {pct}% over last 60s",
    "Heartbeat ok (queue depth {n})",
    "Refreshed token for client {n}",
    "Scheduled job sync-inventory finished in {ms}ms",
    "GET /v1/users/{n} 200 {ms}ms",
]
WARN_MSGS = [
    "Slow query on orders table ({ms}ms)",
    "Retrying request to payments gateway (attempt 2)",
    "Queue depth above soft limit ({n})",
]


def server_log(seed: int, target_tokens: int, needles: list[dict]) -> tuple[str, dict]:
    """needles: [{"pos": 0.0-1.0, "level": "ERROR", "component": "...", "message": "..."}]"""
    rng = random.Random(seed)
    n_lines = target_tokens * CHARS_PER_TOKEN // 72
    t = datetime(2026, 3, 15, 0, 0, 0)
    needle_at = {min(n_lines - 1, int(nd["pos"] * n_lines)): nd for nd in needles}
    lines, info = [], {"needles": []}
    for i in range(n_lines):
        t += timedelta(seconds=rng.randint(20, 180))
        ts = t.strftime("%Y-%m-%d %H:%M:%S")
        if i in needle_at:
            nd = needle_at[i]
            lines.append(f"{ts} {nd['level']:<5} [{nd['component']}] {nd['message']}")
            info["needles"].append({**nd, "timestamp": ts, "line": i + 1})
            continue
        if rng.random() < 0.04:
            level, tmpl = "WARN", rng.choice(WARN_MSGS)
        else:
            level, tmpl = "INFO", rng.choice(INFO_MSGS)
        msg = tmpl.format(n=rng.randint(100, 99999), ms=rng.randint(3, 900), pct=rng.randint(60, 99))
        lines.append(f"{ts} {level:<5} [{rng.choice(COMPONENTS)}] {msg}")
    return "\n".join(lines) + "\n", info


TOPICS = ["onboarding", "security", "travel", "equipment", "time off", "communication", "expenses", "remote work",
          "performance reviews", "benefits", "training", "facilities"]
FILLER_SENTENCES = [
    "Employees should review this section with their manager during their first month.",
    "Questions about this policy can be raised with the People team at any time.",
    "This guidance applies to all full-time and part-time staff unless stated otherwise.",
    "Managers are responsible for making sure their teams are aware of these expectations.",
    "Exceptions require written approval from a department head.",
    "We review this policy every year and publish changes in the internal newsletter.",
    "Local laws take precedence where they set a higher standard than this handbook.",
    "Please keep records of any approvals for at least two years.",
    "The goal of this policy is to keep things simple, fair and transparent.",
    "Contractors should check their agreement, as different terms may apply.",
    "Teams are encouraged to document their own working agreements on top of this baseline.",
    "If anything here is unclear, ask in the #people-help channel.",
]


def handbook(seed: int, target_tokens: int, needle: str, pos: float, distractors: list[str]) -> tuple[str, dict]:
    rng = random.Random(seed)
    n_paras = target_tokens * CHARS_PER_TOKEN // 420
    paras = []
    for i in range(n_paras):
        topic = rng.choice(TOPICS)
        sents = rng.sample(FILLER_SENTENCES, 5)
        paras.append(f"## {i + 1}. {topic.title()}\n\n" + " ".join(sents))
    # Distractors spread evenly, needle at pos. Each replaces the last sentence of a paragraph.
    slots = {}
    for k, d in enumerate(distractors):
        slots[int((k + 1) / (len(distractors) + 1) * n_paras)] = d
    needle_idx = min(n_paras - 1, int(pos * n_paras))
    while needle_idx in slots:
        needle_idx += 1
    slots[needle_idx] = needle
    for idx, sent in slots.items():
        paras[idx] = paras[idx] + " " + sent
    return "# Employee Handbook (2026 edition)\n\n" + "\n\n".join(paras) + "\n", {"needle_section": needle_idx + 1}


PRODUCTS = ["Desk lamp", "USB-C cable", "Monitor arm", "Notebook set", "Webcam", "Ring light", "Wrist rest",
            "Mechanical keyboard", "Ergonomic chair", "Standing desk", "Laptop stand", "Cable tray", "Desk mat",
            "Headset", "Docking station", "Footrest", "Whiteboard", "Plant pot", "Mouse", "Speaker"]
COLORS = ["black", "white", "grey", "oak", "walnut", "navy", "sage", "red"]


def catalog_csv(seed: int, target_tokens: int, needle_sku: str, needle_price: float, pos: float,
                near_skus: list[str]) -> tuple[str, dict]:
    """A product catalog with one target SKU and some lookalike SKUs (e.g. off by one digit)."""
    rng = random.Random(seed)
    n_rows = target_tokens * CHARS_PER_TOKEN // 52
    rows = ["sku,name,color,price_usd,stock"]
    used = {needle_sku, *near_skus}
    needle_idx = min(n_rows - 1, int(pos * n_rows))
    near_idx = {min(n_rows - 1, int(f * n_rows)): s for f, s in zip([0.2, 0.45, 0.7, 0.85], near_skus)}
    for i in range(n_rows):
        if i == needle_idx:
            sku, price = needle_sku, needle_price
        elif i in near_idx and near_idx[i] != needle_sku:
            sku, price = near_idx[i], round(rng.uniform(5, 600), 2)
        else:
            while True:
                sku = f"SKU-{rng.randint(10000, 99999)}"
                if sku not in used:
                    break
            used.add(sku)
            price = round(rng.uniform(5, 600), 2)
        rows.append(f"{sku},{rng.choice(PRODUCTS)},{rng.choice(COLORS)},{price:.2f},{rng.randint(0, 400)}")
    return "\n".join(rows) + "\n", {"needle_row": needle_idx + 2}


GENERATORS = {"server_log": server_log, "handbook": handbook, "catalog_csv": catalog_csv}


def generate(spec: dict) -> str:
    return GENERATORS[spec["generator"]](**spec["params"])[0]
