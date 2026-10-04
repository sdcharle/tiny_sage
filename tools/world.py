"""The mock environment a task runs in: fixture data, per-task files, injected faults and side effects."""

import ast
import copy
import operator
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools import fixtures as fx
from tools.longdocs import generate

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


class World:
    def __init__(self, task: dict | None = None):
        task = task or {}
        self.files = dict(fx.FILES)
        for f in task.get("files", []):
            self.files[f["path"]] = f["content"] if "content" in f else generate(f)
        self.faults = copy.deepcopy(task.get("faults", []))
        self.emails: list[dict] = []
        self.events: list[dict] = []
        self.call_log: list[dict] = []

    # ---- dispatch -------------------------------------------------------------

    def call(self, name: str, args: dict):
        self.call_log.append({"name": name, "args": args})
        for fault in self.faults:
            if fault["tool"] == name and fault.get("times", 1) > 0 and _args_match(fault.get("when_args"), args):
                fault["times"] = fault.get("times", 1) - 1
                return fault["response"]
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}

    def state(self) -> dict:
        return {"emails": self.emails, "events": self.events}

    # ---- tools ----------------------------------------------------------------

    def t_get_weather(self, city, units="celsius"):
        key = _norm_city(city)
        match = next((c for c in fx.WEATHER if c.lower() == key), None)
        if match is None:
            return {"error": f"City not found: '{city}'."}
        temp_c, cond, hum = fx.WEATHER[match]
        temp = temp_c if units == "celsius" else round(temp_c * 9 / 5 + 32, 1)
        return {"city": match, "temperature": temp, "units": units, "condition": cond, "humidity_pct": hum}

    def t_calculator(self, expression):
        if "^" in expression:
            return {"error": "Unsupported operator '^'. Use ** for exponents."}
        try:
            value = _safe_eval(expression)
        except ZeroDivisionError:
            return {"error": "Division by zero."}
        except Exception:
            return {"error": f"Could not parse expression: '{expression}'."}
        if isinstance(value, float):
            value = round(value, 10)
            if value.is_integer():
                value = int(value)
        return {"expression": expression, "result": value}

    def t_search_web(self, query, max_results=5, recency="any"):
        words = set(re.findall(r"[a-z0-9.\-]+", query.lower()))
        now = fx.NOW_UTC.date()
        window = {"any": None, "day": 1, "week": 7, "month": 31}[recency] if recency in ("any", "day", "week", "month") else None
        scored = []
        for doc in fx.SEARCH_CORPUS:
            age = (now - datetime.strptime(doc["date"], "%Y-%m-%d").date()).days
            if window is not None and age > window:
                continue
            score = len(words & set(doc["keywords"]))
            if score:
                scored.append((score, doc["date"], doc))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)  # best match first, then newest
        results = [{k: d[k] for k in ("title", "url", "date", "snippet")} for _, _, d in scored[:max_results]]
        return {"query": query, "results": results}

    def t_read_file(self, path):
        if path in self.files:
            return {"path": path, "content": self.files[path]}
        return {"error": f"File not found: '{path}'."}

    def t_list_files(self, directory):
        d = directory.rstrip("/") or "/"
        prefix = d if d.endswith("/") else d + "/"
        entries = set()
        for p in self.files:
            if p.startswith(prefix):
                rest = p[len(prefix):]
                entries.add(rest.split("/")[0] + ("/" if "/" in rest else ""))
        if not entries:
            return {"error": f"Directory not found or empty: '{directory}'."}
        return {"directory": d, "entries": sorted(entries)}

    def t_find_user(self, query):
        q = query.strip().lower()
        tokens = q.split()
        results = [u for u in fx.USERS
                   if q in u["email"] or q in u["name"].lower() or all(t in u["name"].lower() for t in tokens)]
        if not results:
            return {"results": [], "message": f"No users found matching '{query}'."}
        return {"results": results}

    def t_get_orders(self, user_id, status="all"):
        if not any(u["user_id"] == user_id for u in fx.USERS):
            return {"error": f"Unknown user_id '{user_id}'."}
        orders = [o for o in fx.ORDERS if o["user_id"] == user_id and status in ("all", o["status"])]
        orders.sort(key=lambda o: o["date"], reverse=True)
        return {"user_id": user_id, "status_filter": status, "orders": [
            {"order_id": o["order_id"], "date": o["date"], "status": o["status"],
             "total_usd": o["total_usd"], "item_count": sum(i["qty"] for i in o["items"])} for o in orders]}

    def t_get_order(self, order_id):
        if not re.fullmatch(r"ORD-\d{4}", order_id):
            return {"error": f"Invalid order id format '{order_id}'. Expected format ORD-NNNN, e.g. ORD-5001."}
        o = next((o for o in fx.ORDERS if o["order_id"] == order_id), None)
        if o is None:
            return {"error": f"Order not found: '{order_id}'."}
        return copy.deepcopy(o)

    def t_convert_currency(self, amount, from_currency, to_currency):
        f, t = from_currency.upper(), to_currency.upper()
        if f not in fx.RATES or t not in fx.RATES:
            return {"error": f"Unsupported currency. Supported: {', '.join(sorted(fx.RATES))}."}
        rate = fx.RATES[t] / fx.RATES[f]
        return {"amount": amount, "from_currency": f, "to_currency": t,
                "rate": round(rate, 6), "result": round(amount * rate, 2)}

    def t_get_time(self, timezone):
        try:
            local = fx.NOW_UTC.astimezone(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            return {"error": f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/London'."}
        off = local.utcoffset() or timedelta(0)
        mins = int(off.total_seconds() // 60)
        sign = "+" if mins >= 0 else "-"
        return {"timezone": timezone, "local_time": local.strftime("%Y-%m-%d %H:%M"),
                "weekday": local.strftime("%A"), "utc_offset": f"{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"}

    def t_get_stock_price(self, ticker):
        t = ticker.strip().upper().lstrip("$")
        if t not in fx.STOCKS:
            return {"error": f"Unknown ticker '{ticker}'."}
        name, price = fx.STOCKS[t]
        return {"ticker": t, "company": name, "price_usd": price}

    def t_send_email(self, to, subject, body, cc=None):
        for addr in [to, *(cc or [])]:
            if not EMAIL_RE.match(addr.strip()):
                return {"error": f"Invalid email address '{addr}'."}
        msg = {"to": to.strip(), "subject": subject, "body": body, "cc": [c.strip() for c in (cc or [])]}
        self.emails.append(msg)
        return {"status": "sent", "message_id": f"MSG-{len(self.emails):03d}"}

    def t_create_event(self, title, date, start_time, duration_minutes=60, attendees=None, location=None):
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": f"Invalid date '{date}'. Use YYYY-MM-DD."}
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", start_time):
            return {"error": f"Invalid start_time '{start_time}'. Use 24-hour HH:MM."}
        ev = {"title": title, "date": date, "start_time": start_time, "duration_minutes": duration_minutes,
              "attendees": list(attendees or []), "location": location}
        self.events.append(ev)
        return {"status": "created", "event_id": f"EVT-{len(self.events):03d}", **ev}


# ---- helpers ------------------------------------------------------------------

def _norm_city(city: str) -> str:
    c = city.split(",")[0].strip().lower()
    return {"nyc": "new york", "new york city": "new york", "la": "los angeles"}.get(c, c)


def _args_match(spec, args) -> bool:
    if not spec:
        return True
    return all(str(args.get(k, "")).lower() == str(v).lower() for k, v in spec.items())


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr.replace(",", ""), mode="eval"))
