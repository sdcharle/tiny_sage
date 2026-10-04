"""Build the task suite: tasks/T0.jsonl ... tasks/T6.jsonl.

Every expected answer is computed from the mock World, so fixtures and tasks can't drift apart.
Every task also carries an `oracle` (a known-good sequence of tool calls plus a final answer);
`python -m harness selftest` replays those to prove each task is solvable and its grader is sane.

Run: python -m tasks.build
"""

import json
import re
from pathlib import Path

from tools import ALL_TOOLS, World
from tools import fixtures as fx
from tools.longdocs import catalog_csv, handbook, server_log

OUT = Path(__file__).parent
W = World()
LONG_TOKENS = 10_000


def run(name, **args):
    out = W.call(name, args)
    assert "error" not in out, (name, args, out)
    return out


def call(name, **args):
    return {"name": name, "args": args}


def task(tid, tier, prompt, expect, rounds, final, tools=None, faults=None, files=None, tags=()):
    t = {"id": tid, "tier": tier, "prompt": prompt, "tools": tools or ALL_TOOLS, "expect": expect,
         "oracle": {"rounds": rounds, "final": final}, "optimal_steps": len(rounds) + 1, "tags": list(tags)}
    if faults:
        t["faults"] = faults
    if files:
        t["files"] = files
    return t


# ---- answer patterns ------------------------------------------------------------

def num_re(x, dp=2):
    s = f"{x:.{dp}f}"
    ip, frac = s.split(".") if "." in s else (s, "")
    frac = frac.rstrip("0")
    ip = re.escape(ip)
    if frac:
        return rf"(?<![\d.]){ip}\.{frac}0*(?!\d)"
    return rf"(?<![\d.]){ip}(?:\.0+)?(?!\d)"


def money_re(x):
    """Exact to the cent, or rounded to a whole unit."""
    return [num_re(x), num_re(round(x), 0)]


def time_re(hhmm):
    h, m = map(int, hhmm.split(":"))
    h12, ampm = (h % 12 or 12), ("a" if h < 12 else "p")
    ampm_re = rf"\s*{ampm}\.?\s?m\b"
    pats = [rf"(?<!\d){h:02d}:{m:02d}(?!\d)", rf"(?<!\d){h}:{m:02d}(?!\d)" if h < 10 else None,
            rf"(?<!\d){h12}:{m:02d}{ampm_re}"]
    if m == 0:
        pats.append(rf"(?<!\d){h12}{ampm_re}")
    return [p for p in pats if p]


def cdeg(c):
    return f"{c}°C"


# ---- T0: format ------------------------------------------------------------------

def t0():
    tasks = []
    phr = ["What's the weather like in {c} right now?", "How warm is it in {c} at the moment?",
           "Can you check the current weather in {c}?"]
    for i, c in enumerate(["Paris", "Toronto", "Lagos", "Madrid", "Dublin", "Tokyo", "Berlin", "Sydney", "Cairo", "Reykjavik"]):
        w = run("get_weather", city=c)
        tasks.append(task(f"T0-weather-{i:02d}", "T0", phr[i % 3].format(c=c), {
            "first_call": "get_weather",
            "required_calls": [{"name": "get_weather", "args": {"city": {"$contains": c}, "units": {"$default": "celsius"}}}],
            "final_all": [num_re(w["temperature"], 0)],
        }, [[call("get_weather", city=c)]], f"It's {cdeg(w['temperature'])} and {w['condition']} in {c}.",
            tools=["get_weather"]))

    for i, t in enumerate(["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]):
        name, price = fx.STOCKS[t]
        tasks.append(task(f"T0-stock-{i:02d}", "T0", f"What's {name}'s stock price right now?", {
            "first_call": "get_stock_price",
            "required_calls": [{"name": "get_stock_price", "args": {"ticker": {"$regex": rf"^\$?{t}$"}}}],
            "final_all": [money_re(price)],
        }, [[call("get_stock_price", ticker=t)]], f"{name} ({t}) is at ${price:.2f}.", tools=["get_stock_price"]))

    exprs = ["1234 * 5678", "98765 / 25", "(4500 - 1275) * 1.08", "2 ** 20", "17.5 * 342.8",
             "(123 + 456) * (789 - 321)", "100000 / 7", "365 * 24 * 60"]
    for i, e in enumerate(exprs):
        r = run("calculator", expression=e)["result"]
        pat = r"14285\.71" if e == "100000 / 7" else num_re(r)
        tasks.append(task(f"T0-calc-{i:02d}", "T0", f"Use the calculator to work out {e}.", {
            "first_call": "calculator", "required_calls": [{"name": "calculator"}], "final_all": [pat],
        }, [[call("calculator", expression=e)]], f"{e} = {r}", tools=["calculator"]))

    zones = [("Tokyo", ["Asia/Tokyo"]), ("London", ["Europe/London"]), ("Los Angeles", ["America/Los_Angeles"]),
             ("Mumbai", ["Asia/Kolkata", "Asia/Calcutta"]), ("Paris", ["Europe/Paris"])]
    for i, (city, tzs) in enumerate(zones):
        lt = run("get_time", timezone=tzs[0])["local_time"][-5:]
        tasks.append(task(f"T0-time-{i:02d}", "T0", f"What time is it in {city} right now?", {
            "first_call": "get_time",
            "required_calls": [{"name": "get_time", "args": {"timezone": {"$any": tzs}}}],
            "final_all": [time_re(lt)],
        }, [[call("get_time", timezone=tzs[0])]], f"It's {lt} in {city}.", tools=["get_time"]))

    files = [("/home/notes/todo.txt", "renew passport"), ("/home/notes/groceries.txt", "spinach"),
             ("/home/notes/ideas.md", "ramen"), ("/home/reports/q1_summary.txt", r"1\.84")]
    for i, (p, ans) in enumerate(files):
        tasks.append(task(f"T0-file-{i:02d}", "T0", f"Show me what's in {p}.", {
            "first_call": "read_file", "required_calls": [{"name": "read_file", "args": {"path": p}}],
            "final_all": [ans],
        }, [[call("read_file", path=p)]], W.files[p], tools=["read_file"]))

    convs = [(250, "USD", "EUR"), (1000, "GBP", "USD"), (50, "EUR", "JPY"), (75, "CAD", "GBP"), (1200, "INR", "USD")]
    for i, (a, f, t) in enumerate(convs):
        r = run("convert_currency", amount=a, from_currency=f, to_currency=t)["result"]
        tasks.append(task(f"T0-fx-{i:02d}", "T0", f"Convert {a} {f} to {t}.", {
            "first_call": "convert_currency",
            "required_calls": [{"name": "convert_currency", "args": {"amount": a, "from_currency": f, "to_currency": t}}],
            "final_all": [money_re(r)],
        }, [[call("convert_currency", amount=a, from_currency=f, to_currency=t)]], f"{a} {f} = {r:.2f} {t}",
            tools=["convert_currency"]))
    return tasks


# ---- T1: selection ---------------------------------------------------------------

def t1():
    tasks = []

    def add(tid, prompt, c, final_pat, final_text, state=None):
        exp = {"first_call": c["name"], "required_calls": [{"name": c["name"]}]}
        if final_pat:
            exp["final_all"] = [final_pat]
        if state:
            exp["state"] = state
        tasks.append(task(f"T1-{tid}", "T1", prompt, exp, [[c]], final_text))

    add("weather-0", "Should I bring an umbrella in London today?", call("get_weather", city="London"), "rain",
        "Yes, there's light rain in London.")
    add("weather-1", "Is it t-shirt weather in Madrid right now?", call("get_weather", city="Madrid"), num_re(19, 0),
        "It's 19°C and sunny in Madrid.")
    add("weather-2", "Is it snowing in Toronto?", call("get_weather", city="Toronto"), "snow",
        "Yes, there are snow showers in Toronto.")

    for tid, prompt, e in [("calc-0", "What's 17.5% of 48,213.60?", "48213.60 * 0.175"),
                           ("calc-1", "If I split a $1,847.35 bill eight ways, how much does each person pay?", "1847.35 / 8"),
                           ("calc-2", "How many seconds are there in 3.5 weeks?", "3.5 * 7 * 24 * 3600")]:
        r = run("calculator", expression=e)["result"]
        add(tid, prompt, call("calculator", expression=e), money_re(r) if isinstance(r, float) else num_re(r), f"{r}")
    # 1847.35 / 8 = 230.91875: accept 230.92 or 230.91875
    tasks[-2]["expect"]["final_all"] = [[r"230\.92", r"230\.91875"]]
    tasks[-2]["oracle"]["final"] = "$230.92 each."

    add("search-0", "Any news on the Rust 2.0 release?", call("search_web", query="Rust 2.0 release"), r"q4",
        "The Rust 2.0 roadmap targets a stable release in Q4 2026.")
    add("search-1", "What's been happening with coffee prices recently?", call("search_web", query="coffee prices"),
        r"4\.12", "Arabica futures rose to $4.12 per pound.")
    add("search-2", "Has anyone broken the marathon world record lately?", call("search_web", query="marathon world record"),
        r"2:00:35", "Yes, Daniel Kiprop ran 2:00:35 in Tokyo.")

    add("file-0", "What's on my todo list? It's at /home/notes/todo.txt.", call("read_file", path="/home/notes/todo.txt"),
        "passport", "Renew passport, book dentist, send Q1 invoice, water the plants.")
    add("file-1", "What do I need from the shops? The list is in /home/notes/groceries.txt.",
        call("read_file", path="/home/notes/groceries.txt"), "spinach", "Oat milk, eggs, spinach, sourdough, coffee beans.")
    add("file-2", "What database host does /home/config/app.yaml point to?", call("read_file", path="/home/config/app.yaml"),
        r"db-prod-3", "db-prod-3.internal")

    add("ls-0", "What files are in my notes folder (/home/notes)?", call("list_files", directory="/home/notes"),
        r"todo\.txt", "groceries.txt, ideas.md, todo.txt")
    add("ls-1", "Which config files do I have under /home/config?", call("list_files", directory="/home/config"),
        r"staging\.yaml", "app.yaml, dev.yaml, staging.yaml")

    add("user-0", "What's Priya Raman's email address?", call("find_user", query="Priya Raman"),
        r"priya\.raman@example\.com", "priya.raman@example.com")
    add("user-1", "Which city is Kenji Sato based in?", call("find_user", query="Kenji Sato"), "tokyo", "Tokyo.")
    add("user-2", "Look up Lena Fischer in the company directory.", call("find_user", query="Lena Fischer"),
        r"lena\.fischer@example\.com", "Lena Fischer (U1010), lena.fischer@example.com, Berlin.")

    add("orders-0", "What orders does user U1004 have?", call("get_orders", user_id="U1004"), r"ord-5008",
        "ORD-5008 (pending) and ORD-5007 (delivered).")
    add("orders-1", "Show me the shipped orders for user U1001.", call("get_orders", user_id="U1001", status="shipped"),
        r"ord-5002", "ORD-5002, shipped on 2026-02-20.")

    add("order-0", "What items are in order ORD-5009?", call("get_order", order_id="ORD-5009"), "webcam",
        "A webcam and a ring light.")
    add("order-1", "What's the total of order ORD-5012?", call("get_order", order_id="ORD-5012"), money_re(499),
        "$499.00")

    for tid, prompt, (a, f, t) in [("fx-0", "How much is 80 GBP in yen?", (80, "GBP", "JPY")),
                                   ("fx-1", "I have 500 Swiss francs. What's that in euros?", (500, "CHF", "EUR"))]:
        r = run("convert_currency", amount=a, from_currency=f, to_currency=t)["result"]
        add(tid, prompt, call("convert_currency", amount=a, from_currency=f, to_currency=t), money_re(r), f"{r:.2f} {t}")

    add("time-0", "What time is it for my colleague in Berlin?", call("get_time", timezone="Europe/Berlin"),
        time_re("14:00"), "It's 14:00 in Berlin.")
    add("time-1", "Is it still morning in Los Angeles?", call("get_time", timezone="America/Los_Angeles"),
        time_re("06:00"), "Yes, it's 06:00 in Los Angeles.")
    add("time-2", "What's the date in Sydney right now?", call("get_time", timezone="Australia/Sydney"),
        r"17|tuesday", "It's Tuesday, 2026-03-17 in Sydney.")

    for tid, prompt, t in [("stock-0", "How's NVIDIA trading today?", "NVDA"),
                           ("stock-1", "What's one share of Meta worth right now?", "META"),
                           ("stock-2", "Check the Microsoft share price for me.", "MSFT")]:
        price = fx.STOCKS[t][1]
        add(tid, prompt, call("get_stock_price", ticker=t), money_re(price), f"${price:.2f}")

    add("email-0", "Send bob.ng@example.com a quick email saying I'm running 10 minutes late.",
        call("send_email", to="bob.ng@example.com", subject="Running late", body="Running about 10 minutes late, sorry!"),
        None, "Sent.", state={"emails": [{"to": "bob.ng@example.com"}]})
    add("email-1", "Email erin.walsh@example.com with the subject 'Lunch' and ask whether Thursday works.",
        call("send_email", to="erin.walsh@example.com", subject="Lunch", body="Does Thursday work for lunch?"),
        None, "Sent.", state={"emails": [{"to": "erin.walsh@example.com", "subject": {"$contains": "lunch"}}]})

    add("event-0", "Put 'Dentist' on my calendar for 2026-03-18 at 08:30.",
        call("create_event", title="Dentist", date="2026-03-18", start_time="08:30"), None, "Added.",
        state={"events": [{"title": {"$contains": "dentist"}, "date": "2026-03-18", "start_time": "08:30"}]})
    add("event-1", "Block 2026-03-20 from 13:00 for an hour and call it 'Focus time'.",
        call("create_event", title="Focus time", date="2026-03-20", start_time="13:00", duration_minutes=60), None,
        "Blocked.", state={"events": [{"title": {"$contains": "focus"}, "date": "2026-03-20", "start_time": "13:00",
                                       "duration_minutes": 60}]})
    return tasks


# ---- T2: arguments ---------------------------------------------------------------

def t2():
    tasks = []
    # Today is Monday 2026-03-16.
    events = [
        ("Schedule 'Design review' this Thursday at 3:30pm for 45 minutes with alice.chen@example.com and bob.ng@example.com.",
         dict(title="Design review", date="2026-03-19", start_time="15:30", duration_minutes=45,
              attendees=["alice.chen@example.com", "bob.ng@example.com"])),
        ("Create an event called 'Standup' tomorrow at 9am, 15 minutes long.",
         dict(title="Standup", date="2026-03-17", start_time="09:00", duration_minutes=15)),
        ("Add 'Lunch with Carol' the day after tomorrow at noon, location 'Cafe Rosa'.",
         dict(title="Lunch with Carol", date="2026-03-18", start_time="12:00", location="Cafe Rosa")),
        ("Set up a 90 minute 'Quarterly planning' session on March 30th at 10am with priya.raman@example.com.",
         dict(title="Quarterly planning", date="2026-03-30", start_time="10:00", duration_minutes=90,
              attendees=["priya.raman@example.com"])),
        ("Book 'Team offsite' a week from today at half past ten in the morning, for 3 hours.",
         dict(title="Team offsite", date="2026-03-23", start_time="10:30", duration_minutes=180)),
        ("Put 'Call with Kenji' on April 1st at 7:15pm for 30 minutes and invite kenji.sato@example.com.",
         dict(title="Call with Kenji", date="2026-04-01", start_time="19:15", duration_minutes=30,
              attendees=["kenji.sato@example.com"])),
        ("Schedule 'Budget sync' this Friday at 4pm with carol.diaz@example.com, dev.patel@example.com and erin.walsh@example.com.",
         dict(title="Budget sync", date="2026-03-20", start_time="16:00",
              attendees=["carol.diaz@example.com", "dev.patel@example.com", "erin.walsh@example.com"])),
        ("Create 'Gym' for tomorrow at 6:45 in the morning, 50 minutes, at 'Downtown gym'.",
         dict(title="Gym", date="2026-03-17", start_time="06:45", duration_minutes=50, location="Downtown gym")),
        ("Add 'Flight to Tokyo' on 2026-04-12 at 23:05. It's 13 hours long.",
         dict(title="Flight to Tokyo", date="2026-04-12", start_time="23:05", duration_minutes=780)),
        ("Create an event 'Retro' next Monday at 2pm for an hour and a half.",
         dict(title="Retro", date="2026-03-23", start_time="14:00", duration_minutes=90)),
        ("Schedule 'Interview: backend candidate' on the 25th of this month from 11:00 to 11:45.",
         dict(title="Interview: backend candidate", date="2026-03-25", start_time="11:00", duration_minutes=45)),
        ("Make a calendar entry 'Pay rent' on the last day of this month at 9am.",
         dict(title="Pay rent", date="2026-03-31", start_time="09:00")),
    ]
    for i, (prompt, a) in enumerate(events):
        key_word = a["title"].split()[0].rstrip(":")
        spec = {"title": {"$contains": key_word}, "date": a["date"], "start_time": a["start_time"],
                "duration_minutes": a.get("duration_minutes", {"$default": 60})}
        if "attendees" in a:
            spec["attendees"] = {"$set": a["attendees"]}
        if "location" in a:
            spec["location"] = {"$contains": a["location"].split()[-1]}
        tasks.append(task(f"T2-event-{i:02d}", "T2", prompt, {
            "first_call": "create_event", "required_calls": [{"name": "create_event", "args": spec}],
            "state": {"events": [spec]},
        }, [[call("create_event", **a)]], "Done, it's on your calendar."))

    for i, c in enumerate(["Chicago", "New York", "Seoul", "Sydney"]):
        w = run("get_weather", city=c, units="fahrenheit")
        tasks.append(task(f"T2-units-{i:02d}", "T2", f"What's the temperature in {c} in Fahrenheit?", {
            "first_call": "get_weather",
            "required_calls": [{"name": "get_weather", "args": {"city": {"$contains": c}, "units": "fahrenheit"}}],
            "final_all": [num_re(w["temperature"], 1)],
        }, [[call("get_weather", city=c, units="fahrenheit")]], f"{w['temperature']}°F in {c}."))

    for i, (prompt, uid, status, ans) in enumerate([
        ("Show me Alice Chen's pending orders. Her user ID is U1001.", "U1001", "pending", "ORD-5003"),
        ("List only the delivered orders for user U1004.", "U1004", "delivered", "ORD-5007"),
        ("John Smith is user U1007. Which of his orders have shipped?", "U1007", "shipped", "ORD-5011"),
        ("Are there any pending orders for user U1008?", "U1008", "pending", "ORD-5013"),
    ]):
        tasks.append(task(f"T2-orders-{i:02d}", "T2", prompt, {
            "first_call": "get_orders",
            "required_calls": [{"name": "get_orders", "args": {"user_id": uid, "status": status}}],
            "final_all": [ans.lower()],
        }, [[call("get_orders", user_id=uid, status=status)]], ans))

    for i, (prompt, a, f, t) in enumerate([
        ("How much is two and a half thousand euros in US dollars?", 2500, "EUR", "USD"),
        ("Convert 1.2 million yen to Canadian dollars.", 1_200_000, "JPY", "CAD"),
        ("What's 99.99 pounds sterling in Australian dollars?", 99.99, "GBP", "AUD"),
        ("Convert fifteen hundred Mexican pesos to US dollars.", 1500, "MXN", "USD"),
        ("How many Indian rupees is 45 Swiss francs?", 45, "CHF", "INR"),
    ]):
        r = run("convert_currency", amount=a, from_currency=f, to_currency=t)["result"]
        tasks.append(task(f"T2-fx-{i:02d}", "T2", prompt, {
            "first_call": "convert_currency",
            "required_calls": [{"name": "convert_currency", "args": {"amount": a, "from_currency": f, "to_currency": t}}],
            "final_all": [money_re(r)],
        }, [[call("convert_currency", amount=a, from_currency=f, to_currency=t)]], f"{r:.2f} {t}"))

    for i, (prompt, a) in enumerate([
        ("Email carol.diaz@example.com with the subject 'Q2 budget', cc dev.patel@example.com and erin.walsh@example.com, "
         "and tell her the numbers are final.",
         dict(to="carol.diaz@example.com", subject="Q2 budget", cc=["dev.patel@example.com", "erin.walsh@example.com"],
              body="Hi Carol, the Q2 budget numbers are final.")),
        ("Send an email to john.smith@example.com, subject 'Standing desk delivery', copying priya.raman@example.com, "
         "saying the desk arrives Friday.",
         dict(to="john.smith@example.com", subject="Standing desk delivery", cc=["priya.raman@example.com"],
              body="Hi John, the standing desk arrives Friday.")),
        ("Write to lena.fischer@example.com with subject 'Berlin trip' and cc kenji.sato@example.com: "
         "say the hotel is booked for April 14-17.",
         dict(to="lena.fischer@example.com", subject="Berlin trip", cc=["kenji.sato@example.com"],
              body="Hi Lena, the hotel is booked for April 14-17.")),
    ]):
        body_kw = {0: "final", 1: "friday", 2: "14"}[i]
        spec = {"to": a["to"], "subject": {"$contains": a["subject"]}, "cc": {"$set": a["cc"]},
                "body": {"$contains": body_kw}}
        tasks.append(task(f"T2-email-{i:02d}", "T2", prompt, {
            "first_call": "send_email", "required_calls": [{"name": "send_email", "args": spec}],
            "state": {"emails": [spec]},
        }, [[call("send_email", **a)]], "Sent."))

    for i, (prompt, a, ans) in enumerate([
        ("Search for news from the past week about solid-state batteries, just the top 3 results.",
         dict(query="solid-state batteries", recency="week", max_results=3), "900"),
        ("Look up articles from the last month about Python, max 2 results.",
         dict(query="Python", recency="month", max_results=2), r"3\.15"),
        ("Give me the single most relevant result from the past week about the Artemis mission.",
         dict(query="Artemis mission", recency="week", max_results=1), "september"),
    ]):
        tasks.append(task(f"T2-search-{i:02d}", "T2", prompt, {
            "first_call": "search_web",
            "required_calls": [{"name": "search_web", "args": {"recency": a["recency"], "max_results": a["max_results"]}}],
            "final_all": [ans],
        }, [[call("search_web", **a)]], run("search_web", **a)["results"][0]["snippet"]))

    for i, (prompt, e) in enumerate([
        ("What's (17 squared minus 40) divided by 3?", "(17 ** 2 - 40) / 3"),
        ("What's $5,000 worth after 10 years at 4% annual interest, compounded yearly?", "5000 * 1.04 ** 10"),
        ("What is 12 percent of 3,450, plus 275?", "3450 * 0.12 + 275"),
    ]):
        r = run("calculator", expression=e)["result"]
        tasks.append(task(f"T2-calc-{i:02d}", "T2", prompt, {
            "first_call": "calculator", "required_calls": [{"name": "calculator"}], "final_all": [money_re(r)],
        }, [[call("calculator", expression=e)]], f"{r:.2f}"))
    return tasks


# ---- T3: restraint ---------------------------------------------------------------

def t3():
    tasks = []
    no_tool = [
        ("Thanks, that's all for now!", None, "You're welcome!"),
        ("What's the capital of Australia?", "canberra", "Canberra."),
        ("Explain what a for loop is in one sentence.", None, "A for loop repeats a block of code for each item in a sequence."),
        ("Write a haiku about autumn.", None, "Red leaves drift and fall / cool wind hums through empty trees / the year exhales slow"),
        ("How many days are in a leap year?", "366", "366."),
        ("Translate 'good morning' into Spanish.", r"buenos d", "Buenos días."),
        ("What does HTTP stand for?", "hypertext transfer protocol", "HyperText Transfer Protocol."),
        ("hi!", None, "Hi! How can I help?"),
        ("What's 2 + 2?", r"(?<!\d)4(?!\d)|four", "4"),
        ("Give me a synonym for 'happy'.", None, "Joyful."),
        ("Who wrote Romeo and Juliet?", "shakespeare", "William Shakespeare."),
        ("Summarize in five words: 'The meeting was moved to Thursday because the room was double-booked.'", None,
         "Meeting moved: room double-booked."),
        ("Is a tomato a fruit or a vegetable?", "fruit", "Botanically, a fruit."),
        ("What's today's date?", r"march 16|16 march|2026-03-16|3/16", "Monday, March 16, 2026."),
        ("How many meters are in 5 kilometers?", "5000", "5000 meters."),
        ("Tell me a fun fact about octopuses.", None, "Octopuses have three hearts."),
    ]
    for i, (prompt, ans, final) in enumerate(no_tool):
        exp = {"no_calls": True}
        if ans:
            exp["final_all"] = [ans]
        tasks.append(task(f"T3-direct-{i:02d}", "T3", prompt, exp, [], final, tags=["direct"]))

    clarify = [
        ("What's the weather like?", []),
        ("Send an email to my manager saying I'm out sick today.", ["find_user"]),
        ("Schedule a meeting with Alice.", ["find_user"]),
        ("Look up my order.", []),
        ("What's the stock price?", []),
        ("Read the file.", ["list_files"]),
        ("Convert my budget to euros.", []),
        ("Email Bob the report.", ["find_user"]),
        ("What time is it there?", []),
        ("Book it for next week.", []),
        ("How much is it in dollars?", []),
        ("What's the total of the order?", []),
        ("Add the usual stuff to my calendar.", []),
    ]
    for i, (prompt, allowed) in enumerate(clarify):
        tasks.append(task(f"T3-clarify-{i:02d}", "T3", prompt,
                          {"asks_clarification": True, "allowed_calls": allowed}, [],
                          "Could you give me a bit more detail about what you mean?", tags=["clarify"]))
    return tasks


# ---- T4: multi-step --------------------------------------------------------------

def user_by_name(name):
    return next(u for u in fx.USERS if u["name"] == name)


CUR_NAMES = {"EUR": "euros", "GBP": "British pounds", "JPY": "Japanese yen", "CAD": "Canadian dollars",
             "AUD": "Australian dollars", "CHF": "Swiss francs", "INR": "Indian rupees", "MXN": "Mexican pesos"}


def t4():
    tasks = []
    people = ["Alice Chen", "Bob Ng", "Carol Diaz", "Dev Patel", "Erin Walsh", "John Smith", "Priya Raman", "Lena Fischer"]
    for i, (name, cur) in enumerate(zip(people, CUR_NAMES)):
        u = user_by_name(name)
        latest = run("get_orders", user_id=u["user_id"])["orders"][0]
        r = run("convert_currency", amount=latest["total_usd"], from_currency="USD", to_currency=cur)["result"]
        tasks.append(task(f"T4-ordertotal-{i:02d}", "T4", f"What's the total of {name}'s most recent order, in {CUR_NAMES[cur]}?", {
            "required_calls": [
                {"name": "find_user"},
                {"name": "get_orders", "args": {"user_id": u["user_id"]}},
                {"name": "convert_currency", "args": {"amount": {"$approx": latest["total_usd"]}, "from_currency": "USD",
                                                      "to_currency": cur}},
            ],
            "final_all": [money_re(r)],
        }, [[call("find_user", query=name)], [call("get_orders", user_id=u["user_id"])],
            [call("convert_currency", amount=latest["total_usd"], from_currency="USD", to_currency=cur)]],
            f"{latest['order_id']} totals {r:.2f} {cur}."))

    pairs = [("Paris", "Tokyo"), ("Madrid", "Cairo"), ("London", "Dublin"), ("Berlin", "Chicago"),
             ("Sydney", "Los Angeles"), ("Seoul", "New York"), ("Mumbai", "Singapore"), ("Reykjavik", "Toronto")]
    for i, (a, b) in enumerate(pairs):
        ta, tb = fx.WEATHER[a][0], fx.WEATHER[b][0]
        warmer, diff = (a, ta - tb) if ta > tb else (b, tb - ta)
        tasks.append(task(f"T4-compare-{i:02d}", "T4", f"Is it warmer in {a} or {b} right now, and by how many degrees Celsius?", {
            "required_calls": [{"name": "get_weather", "args": {"city": {"$contains": a}}},
                               {"name": "get_weather", "args": {"city": {"$contains": b}}}],
            "final_all": [warmer.lower(), num_re(diff, 0)],
        }, [[call("get_weather", city=a), call("get_weather", city=b)]],
            f"{warmer} is warmer by {diff} degrees ({a} {ta}°C, {b} {tb}°C)."))

    for i, name in enumerate(["Alice Chen", "Kenji Sato", "Lena Fischer", "Erin Walsh", "Priya Raman"]):
        u = user_by_name(name)
        temp, cond, _ = fx.WEATHER[u["city"]]
        tasks.append(task(f"T4-mailweather-{i:02d}", "T4", f"Email {name} a short note with the current weather in their city.", {
            "required_calls": [{"name": "find_user"}, {"name": "get_weather", "args": {"city": {"$contains": u["city"]}}}],
            "state": {"emails": [{"to": u["email"], "body": {"$regex": num_re(temp, 0)}}]},
        }, [[call("find_user", query=name)], [call("get_weather", city=u["city"])],
            [call("send_email", to=u["email"], subject=f"Weather in {u['city']}",
                  body=f"It's currently {temp}°C and {cond} in {u['city']}.")]], "Sent."))

    for i, (path, tz) in enumerate([("/home/config/app.yaml", "Asia/Tokyo"), ("/home/config/staging.yaml", "Europe/Berlin"),
                                    ("/home/config/dev.yaml", "America/Los_Angeles")]):
        lt = run("get_time", timezone=tz)["local_time"][-5:]
        tasks.append(task(f"T4-config-{i:02d}", "T4",
                          f"Check {path} and tell me the current local time in the timezone it's configured for.", {
            "required_calls": [{"name": "read_file", "args": {"path": path}}, {"name": "get_time", "args": {"timezone": tz}}],
            "final_all": [time_re(lt)],
        }, [[call("read_file", path=path)], [call("get_time", timezone=tz)]], f"It's {lt} in {tz}."))

    for i, (t, n, cur) in enumerate([("AAPL", 10, "EUR"), ("MSFT", 3, "GBP"), ("NVDA", 5, "JPY"), ("TSLA", 20, "CAD"),
                                     ("META", 2, "CHF")]):
        name, price = fx.STOCKS[t]
        total = round(price * n, 2)
        r = run("convert_currency", amount=total, from_currency="USD", to_currency=cur)["result"]
        tasks.append(task(f"T4-shares-{i:02d}", "T4", f"How much would {n} shares of {name} cost in {CUR_NAMES[cur]}?", {
            "required_calls": [{"name": "get_stock_price"},
                               {"name": "convert_currency", "args": {"amount": {"$approx": total, "$tol": 0.05},
                                                                     "to_currency": cur}}],
            "final_all": [money_re(r)],
        }, [[call("get_stock_price", ticker=t)],
            [call("convert_currency", amount=total, from_currency="USD", to_currency=cur)]], f"{r:.2f} {cur}"))

    for i, (name, status, item) in enumerate([("Bob Okafor", "pending", "desk lamp"), ("John Smith", "pending", "standing desk")]):
        u = user_by_name(name)
        oid = run("get_orders", user_id=u["user_id"], status=status)["orders"][0]["order_id"]
        tasks.append(task(f"T4-items-{i:02d}", "T4", f"Find {name}'s {status} order and tell me what's in it.", {
            "required_calls": [{"name": "find_user"}, {"name": "get_orders", "args": {"user_id": u["user_id"]}},
                               {"name": "get_order", "args": {"order_id": oid}}],
            "final_all": [item],
        }, [[call("find_user", query=name)], [call("get_orders", user_id=u["user_id"], status=status)],
            [call("get_order", order_id=oid)]], f"{oid} contains: {item}."))
    return tasks


# ---- T5: recovery ----------------------------------------------------------------

UNAVAILABLE = {"error": "503 Service Unavailable: upstream timeout, please retry."}


def t5():
    tasks = []
    for i, c in enumerate(["Paris", "Berlin", "Seoul", "Cairo", "Lagos", "Sydney"]):
        temp = fx.WEATHER[c][0]
        tasks.append(task(f"T5-retry-{i:02d}", "T5", f"What's the weather in {c}?", {
            "required_calls": [{"name": "get_weather", "args": {"city": {"$contains": c}}}],
            "final_all": [num_re(temp, 0)],
        }, [[call("get_weather", city=c)], [call("get_weather", city=c)]], f"{temp}°C in {c}.",
            faults=[{"tool": "get_weather", "times": 1, "response": UNAVAILABLE}]))

    for i, (typo, real, field) in enumerate([("Jon Smyth", "John Smith", "email"), ("Pria Raman", "Priya Raman", "city"),
                                             ("Lena Fisher", "Lena Fischer", "email"), ("Caroline Diaz", "Carol Diaz", "city")]):
        u = user_by_name(real)
        ans = re.escape(u[field].lower())
        tasks.append(task(f"T5-suggest-{i:02d}", "T5", f"What's {typo}'s {field}?", {
            "required_calls": [{"name": "find_user"}], "final_all": [ans],
        }, [[call("find_user", query=typo)], [call("find_user", query=real)]], f"{real}'s {field} is {u[field]}.",
            faults=[{"tool": "find_user", "times": 1, "when_args": {"query": typo},
                     "response": {"results": [], "message": f"No exact match for '{typo}'. Did you mean: {real}?"}}]))

    for i, (raw, oid) in enumerate([("ORD5003", "ORD-5003"), ("ORD_5006", "ORD-5006"), ("5012", "ORD-5012"),
                                    ("ord 5009", "ORD-5009")]):
        o = run("get_order", order_id=oid)
        item = o["items"][0]["name"].lower()
        tasks.append(task(f"T5-orderid-{i:02d}", "T5", f"What's in order {raw}?", {
            "required_calls": [{"name": "get_order", "args": {"order_id": oid}}], "final_all": [item],
        }, [[call("get_order", order_id=raw)], [call("get_order", order_id=oid)]], f"{oid}: {item}."))

    for i, (prompt, bad, good) in enumerate([
        ("Use the calculator to compute 7^5 + 3^4.", "7^5 + 3^4", "7**5 + 3**4"),
        ("Use the calculator to work out 2^10 - 10^3.", "2^10 - 10^3", "2**10 - 10**3"),
        ("Use the calculator: 1.5^3 * 4.", "1.5^3 * 4", "1.5**3 * 4"),
    ]):
        r = run("calculator", expression=good)["result"]
        tasks.append(task(f"T5-calc-{i:02d}", "T5", prompt, {
            "required_calls": [{"name": "calculator"}], "final_all": [num_re(r)],
        }, [[call("calculator", expression=bad)], [call("calculator", expression=good)]], f"{r}"))

    for i, (prompt, q, ans) in enumerate([
        ("Search for news about the city council's bike lane vote.", "bike lane vote", r"7-2|42 km"),
        ("Find recent articles on the four-day work week.", "four-day work week", r"89%"),
        ("Look up the latest on the Python 3.15 beta.", "Python 3.15 beta", r"jit|12%"),
    ]):
        tasks.append(task(f"T5-emptysearch-{i:02d}", "T5", prompt, {
            "required_calls": [{"name": "search_web"}], "final_all": [ans],
        }, [[call("search_web", query=q)], [call("search_web", query=q)]], run("search_web", query=q)["results"][0]["snippet"],
            faults=[{"tool": "search_web", "times": 1, "response": {"query": q, "results": [],
                                                                    "message": "Search index is warming up; no results yet."}}]))

    for i, t in enumerate(["GOOGL", "AMZN", "TSLA", "AAPL"]):
        name, price = fx.STOCKS[t]
        tasks.append(task(f"T5-garbage-{i:02d}", "T5", f"What's the current {name} stock price?", {
            "required_calls": [{"name": "get_stock_price"}], "final_all": [money_re(price)],
        }, [[call("get_stock_price", ticker=t)], [call("get_stock_price", ticker=t)]], f"${price:.2f}",
            faults=[{"tool": "get_stock_price", "times": 1,
                     "response": "<html><body><h1>502 Bad Gateway</h1><p>nginx</p></body></html>"}]))

    for i, (prompt, a) in enumerate([
        ("Add 'Car service' to my calendar on 2026-03-21 at 10:00.", dict(title="Car service", date="2026-03-21", start_time="10:00")),
        ("Create an event 'Book club' on 2026-03-26 at 19:30.", dict(title="Book club", date="2026-03-26", start_time="19:30")),
        ("Schedule 'Vet appointment' for 2026-03-19 at 16:15.", dict(title="Vet appointment", date="2026-03-19", start_time="16:15")),
    ]):
        spec = {"title": {"$contains": a["title"].split()[0]}, "date": a["date"], "start_time": a["start_time"]}
        tasks.append(task(f"T5-busy-{i:02d}", "T5", prompt, {
            "required_calls": [{"name": "create_event"}], "state": {"events": [spec]}, "state_counts": {"events": 1},
        }, [[call("create_event", **a)], [call("create_event", **a)]], "Added.",
            faults=[{"tool": "create_event", "times": 1,
                     "response": {"error": "Calendar service busy (409 conflict on write). Please retry."}}]))

    for i, c in enumerate(["Tokyo", "Madrid", "Chicago"]):
        tasks.append(task(f"T5-down-{i:02d}", "T5", f"What's the weather in {c}?", {
            "required_calls": [{"name": "get_weather"}], "max_calls": 6,
            "final_all": [r"unable|couldn'?t|could not|can'?t|cannot|not able|unavailable|error|issue|problem|trouble|down"],
            "final_none": [r"-?\d+(\.\d+)?\s*°"],
        }, [[call("get_weather", city=c)], [call("get_weather", city=c)]],
            "Sorry, the weather service is unavailable right now, so I couldn't get the weather.",
            faults=[{"tool": "get_weather", "times": 99, "response": UNAVAILABLE}]))
    return tasks


# ---- T6: long context ------------------------------------------------------------

LOG_PATH, HANDBOOK_PATH, CATALOG_PATH = "/var/log/app.log", "/docs/handbook.md", "/data/catalog.csv"


def t6():
    tasks = []
    errors = [("db", "Database connection pool exhausted (max=50)", r"pool|exhausted"),
              ("billing", "Payment gateway returned 502 for 37 consecutive requests", r"payment|gateway|502"),
              ("worker-2", "Disk usage on /var/data reached 98%", r"disk|98%"),
              ("auth", "TLS certificate for api.example.com expired", r"certificate|tls|expired")]
    k = 0
    for comp, msg, kw in errors:
        for pos in (0.1, 0.5, 0.9):
            params = {"seed": 100 + k, "target_tokens": LONG_TOKENS,
                      "needles": [{"pos": pos, "level": "ERROR", "component": comp, "message": msg}]}
            ts = server_log(**params)[1]["needles"][0]["timestamp"]
            hhmm = ts[11:16]
            tasks.append(task(f"T6-log-{k:02d}", "T6",
                              f"Read {LOG_PATH}. When did the ERROR occur, and what was the error message?", {
                "required_calls": [{"name": "read_file", "args": {"path": LOG_PATH}}],
                "final_all": [re.escape(hhmm), kw],
            }, [[call("read_file", path=LOG_PATH)]], f"At {ts}: {msg}.",
                files=[{"path": LOG_PATH, "generator": "server_log", "params": params}], tags=[f"pos={pos}"]))
            k += 1

    for i in range(3):
        comps = [("cache", "Eviction storm: 12000 keys evicted in 1s"), ("scheduler", "Job sync-inventory crashed (exit 137)"),
                 ("api", "Unhandled exception in /v1/checkout")]
        needles = [{"pos": p, "level": "ERROR", "component": c, "message": m}
                   for p, (c, m) in zip((0.15 + 0.1 * i, 0.5, 0.85 - 0.1 * i), comps)]
        params = {"seed": 200 + i, "target_tokens": LONG_TOKENS, "needles": needles}
        tasks.append(task(f"T6-count-{i:02d}", "T6",
                          f"How many ERROR lines are in {LOG_PATH}, and which components logged them?", {
            "required_calls": [{"name": "read_file", "args": {"path": LOG_PATH}}],
            "final_all": [r"(?<!\d)3(?!\d)|three", "cache", "scheduler", r"\bapi\b"],
        }, [[call("read_file", path=LOG_PATH)]], "There are 3 ERROR lines, from cache, scheduler and api.",
            files=[{"path": LOG_PATH, "generator": "server_log", "params": params}], tags=["count"]))

    facts = [
        ("The annual reimbursement limit for home office equipment is $740.",
         ["The annual reimbursement limit for travel upgrades is $1,200.",
          "The reimbursement limit for professional books is $300 per year."],
         "what's the annual reimbursement limit for home office equipment?", num_re(740, 0), [r"1200"]),
        ("New employees accrue 1.75 vacation days per month during their first year.",
         ["After five years, employees accrue 2.25 vacation days per month."],
         "how many vacation days per month do first-year employees accrue?", r"1\.75", [r"2\.25"]),
        ("The emergency on-call phone number is 555-0147.",
         ["The facilities help desk can be reached at 555-0199."],
         "what's the emergency on-call phone number?", r"555-0147", [r"555-0199"]),
    ]
    k = 0
    for needle, distractors, q, ans, nope in facts:
        for pos in (0.1, 0.5, 0.9):
            params = {"seed": 300 + k, "target_tokens": LONG_TOKENS, "needle": needle, "pos": pos, "distractors": distractors}
            handbook(**params)
            tasks.append(task(f"T6-handbook-{k:02d}", "T6", f"According to {HANDBOOK_PATH}, {q}", {
                "required_calls": [{"name": "read_file", "args": {"path": HANDBOOK_PATH}}],
                "final_all": [ans], "final_none": nope,
            }, [[call("read_file", path=HANDBOOK_PATH)]], needle,
                files=[{"path": HANDBOOK_PATH, "generator": "handbook", "params": params}], tags=[f"pos={pos}"]))
            k += 1

    k = 0
    for sku, price, near in [("SKU-48213", 127.45, ["SKU-48231", "SKU-43213", "SKU-48218"]),
                             ("SKU-70552", 18.90, ["SKU-70525", "SKU-75052", "SKU-70553"])]:
        for pos in (0.1, 0.5, 0.9):
            params = {"seed": 400 + k, "target_tokens": LONG_TOKENS, "needle_sku": sku, "needle_price": price,
                      "pos": pos, "near_skus": near}
            catalog_csv(**params)
            tasks.append(task(f"T6-csv-{k:02d}", "T6", f"What's the price of {sku} in {CATALOG_PATH}?", {
                "required_calls": [{"name": "read_file", "args": {"path": CATALOG_PATH}}],
                "final_all": [num_re(price)],
            }, [[call("read_file", path=CATALOG_PATH)]], f"{sku} costs ${price:.2f}.",
                files=[{"path": CATALOG_PATH, "generator": "catalog_csv", "params": params}], tags=[f"pos={pos}"]))
            k += 1
    return tasks


TIERS = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6}


def main():
    total = 0
    for tier, fn in TIERS.items():
        tasks = fn()
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"duplicate ids in {tier}"
        with open(OUT / f"{tier}.jsonl", "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        print(f"{tier}: {len(tasks)} tasks")
        total += len(tasks)
    print(f"total: {total}")


if __name__ == "__main__":
    main()
