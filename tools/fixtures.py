"""Static data behind the mock tools. Edit with care: tasks/build.py derives expected answers from it."""

from datetime import datetime, timezone

# The fixed "now" for every task. Monday, March 16 2026, 13:00 UTC.
NOW_UTC = datetime(2026, 3, 16, 13, 0, 0, tzinfo=timezone.utc)

USERS = [
    {"user_id": "U1001", "name": "Alice Chen", "email": "alice.chen@example.com", "city": "Paris"},
    {"user_id": "U1002", "name": "Bob Ng", "email": "bob.ng@example.com", "city": "Toronto"},
    {"user_id": "U1003", "name": "Bob Okafor", "email": "bob.okafor@example.com", "city": "Lagos"},
    {"user_id": "U1004", "name": "Carol Diaz", "email": "carol.diaz@example.com", "city": "Madrid"},
    {"user_id": "U1005", "name": "Dev Patel", "email": "dev.patel@example.com", "city": "Mumbai"},
    {"user_id": "U1006", "name": "Erin Walsh", "email": "erin.walsh@example.com", "city": "Dublin"},
    {"user_id": "U1007", "name": "John Smith", "email": "john.smith@example.com", "city": "Chicago"},
    {"user_id": "U1008", "name": "Priya Raman", "email": "priya.raman@example.com", "city": "Singapore"},
    {"user_id": "U1009", "name": "Kenji Sato", "email": "kenji.sato@example.com", "city": "Tokyo"},
    {"user_id": "U1010", "name": "Lena Fischer", "email": "lena.fischer@example.com", "city": "Berlin"},
]

# city -> (temp_celsius, condition, humidity_pct)
WEATHER = {
    "Paris": (14, "partly cloudy", 62),
    "Toronto": (3, "snow showers", 80),
    "Lagos": (31, "sunny", 70),
    "Madrid": (19, "sunny", 40),
    "Mumbai": (33, "hazy", 74),
    "Dublin": (9, "light rain", 88),
    "Chicago": (6, "windy", 55),
    "Singapore": (30, "thunderstorms", 85),
    "Tokyo": (16, "clear", 50),
    "Berlin": (8, "overcast", 71),
    "London": (11, "light rain", 83),
    "New York": (10, "cloudy", 60),
    "Sydney": (24, "sunny", 58),
    "Cairo": (27, "clear", 30),
    "Reykjavik": (-2, "snow", 75),
    "Los Angeles": (21, "sunny", 45),
    "Seoul": (12, "clear", 42),
    "Mexico City": (23, "partly cloudy", 35),
}

# Units of currency per 1 USD.
RATES = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.5, "CAD": 1.36,
    "INR": 83.2, "AUD": 1.52, "CHF": 0.88, "MXN": 17.1,
}

STOCKS = {
    "AAPL": ("Apple", 212.34),
    "MSFT": ("Microsoft", 431.10),
    "GOOGL": ("Alphabet", 168.52),
    "NVDA": ("NVIDIA", 924.80),
    "TSLA": ("Tesla", 176.45),
    "AMZN": ("Amazon", 183.90),
    "META": ("Meta", 498.25),
}


def _order(order_id, user_id, date, status, items):
    total = round(sum(q * p for _, _, q, p in items), 2)
    return {
        "order_id": order_id, "user_id": user_id, "date": date, "status": status,
        "items": [{"sku": s, "name": n, "qty": q, "unit_price_usd": p} for s, n, q, p in items],
        "total_usd": total,
    }


ORDERS = [
    _order("ORD-5001", "U1001", "2026-01-12", "delivered", [("SKU-101", "Desk lamp", 1, 39.99), ("SKU-205", "USB-C cable", 2, 12.50)]),
    _order("ORD-5002", "U1001", "2026-02-20", "shipped", [("SKU-330", "Mechanical keyboard", 1, 129.00)]),
    _order("ORD-5003", "U1001", "2026-03-10", "pending", [("SKU-412", "Monitor arm", 1, 89.95), ("SKU-205", "USB-C cable", 1, 12.50)]),
    _order("ORD-5004", "U1002", "2026-02-02", "delivered", [("SKU-118", "Notebook set", 3, 8.75)]),
    _order("ORD-5005", "U1002", "2026-03-08", "shipped", [("SKU-520", "Noise-cancelling headphones", 1, 249.00)]),
    _order("ORD-5006", "U1003", "2026-03-01", "pending", [("SKU-101", "Desk lamp", 2, 39.99)]),
    _order("ORD-5007", "U1004", "2026-01-25", "delivered", [("SKU-640", "Ergonomic chair", 1, 349.00)]),
    _order("ORD-5008", "U1004", "2026-03-12", "pending", [("SKU-205", "USB-C cable", 4, 12.50), ("SKU-118", "Notebook set", 1, 8.75)]),
    _order("ORD-5009", "U1005", "2026-02-14", "shipped", [("SKU-701", "Webcam", 1, 79.00), ("SKU-702", "Ring light", 1, 34.50)]),
    _order("ORD-5010", "U1006", "2026-03-05", "delivered", [("SKU-330", "Mechanical keyboard", 1, 129.00), ("SKU-331", "Wrist rest", 1, 19.99)]),
    _order("ORD-5011", "U1007", "2026-02-28", "shipped", [("SKU-412", "Monitor arm", 2, 89.95)]),
    _order("ORD-5012", "U1007", "2026-03-14", "pending", [("SKU-801", "Standing desk", 1, 499.00)]),
    _order("ORD-5013", "U1008", "2026-03-11", "pending", [("SKU-520", "Noise-cancelling headphones", 1, 249.00), ("SKU-205", "USB-C cable", 2, 12.50)]),
    _order("ORD-5014", "U1009", "2026-01-30", "delivered", [("SKU-701", "Webcam", 2, 79.00)]),
    _order("ORD-5015", "U1010", "2026-03-09", "shipped", [("SKU-640", "Ergonomic chair", 1, 349.00), ("SKU-331", "Wrist rest", 2, 19.99)]),
]

SEARCH_CORPUS = [
    {"title": "Rust 2.0 roadmap announced at RustConf", "url": "https://news.example.com/rust-2-roadmap", "date": "2026-03-12",
     "snippet": "The Rust project published its 2.0 roadmap, targeting a stable release in Q4 2026 with a new async runtime trait.",
     "keywords": ["rust", "2.0", "roadmap", "release", "programming", "language"]},
    {"title": "Solid-state battery startup hits 900 Wh/L in lab tests", "url": "https://tech.example.com/solid-state-900", "date": "2026-03-13",
     "snippet": "Volta Cells reported a solid-state battery cell reaching 900 Wh/L energy density, with pilot production planned for 2027.",
     "keywords": ["solid-state", "solid", "state", "battery", "batteries", "energy", "ev"]},
    {"title": "Review: the history of solid-state batteries", "url": "https://archive.example.com/ssb-history", "date": "2025-06-02",
     "snippet": "A long look at solid-state battery research from the 1990s to today.",
     "keywords": ["solid-state", "solid", "state", "battery", "batteries", "history"]},
    {"title": "Artemis III crew begins final training phase", "url": "https://space.example.com/artemis-iii", "date": "2026-03-10",
     "snippet": "NASA said the Artemis III crew started final mission training ahead of a lunar landing attempt scheduled for September 2026.",
     "keywords": ["artemis", "nasa", "moon", "lunar", "space", "mission"]},
    {"title": "City council approves new bike lane network", "url": "https://local.example.com/bike-lanes", "date": "2026-03-14",
     "snippet": "The council voted 7-2 to fund 42 km of protected bike lanes, with construction starting in May.",
     "keywords": ["bike", "lanes", "council", "city", "cycling", "transport"]},
    {"title": "Python 3.15 beta released", "url": "https://dev.example.com/python-315-beta", "date": "2026-03-09",
     "snippet": "Python 3.15.0b1 ships with a JIT enabled by default on Linux and macOS and a 12% average speedup.",
     "keywords": ["python", "3.15", "beta", "release", "jit", "programming"]},
    {"title": "Coffee prices climb to 3-year high", "url": "https://markets.example.com/coffee-high", "date": "2026-03-11",
     "snippet": "Arabica futures rose to $4.12 per pound, the highest level in three years, after a dry season in Brazil.",
     "keywords": ["coffee", "prices", "arabica", "commodities", "brazil"]},
    {"title": "Study: four-day work week pilots show sustained productivity", "url": "https://work.example.com/four-day-week", "date": "2026-02-27",
     "snippet": "A follow-up study of 61 companies found 89% kept the four-day week two years later, with no drop in output.",
     "keywords": ["four-day", "work", "week", "productivity", "study", "companies"]},
    {"title": "Sourdough starter basics", "url": "https://food.example.com/sourdough-starter", "date": "2024-09-18",
     "snippet": "Feed your starter equal weights of flour and water once a day; it is ready when it doubles within 6 hours.",
     "keywords": ["sourdough", "starter", "bread", "baking", "flour"]},
    {"title": "Marathon world record broken in Tokyo", "url": "https://sports.example.com/tokyo-marathon", "date": "2026-03-01",
     "snippet": "Kenyan runner Daniel Kiprop finished the Tokyo Marathon in 2:00:35, a new world record.",
     "keywords": ["marathon", "world", "record", "tokyo", "running", "sports"]},
]

FILES = {
    "/home/notes/todo.txt": "TODO\n- Renew passport (expires April 2)\n- Book dentist appointment\n- Send Q1 invoice to Acme Corp\n- Water the plants\n",
    "/home/notes/groceries.txt": "oat milk\neggs (12)\nspinach\nsourdough loaf\ncoffee beans\n",
    "/home/notes/ideas.md": "# Ideas\n\n1. Weekend hackathon on local LLM agents\n2. Turn the garage into a workshop\n3. Learn to make ramen from scratch\n",
    "/home/config/app.yaml": "service: orders-api\nenvironment: production\ntimezone: Asia/Tokyo\ndatabase:\n  host: db-prod-3.internal\n  port: 5432\nreplicas: 4\n",
    "/home/config/staging.yaml": "service: orders-api\nenvironment: staging\ntimezone: Europe/Berlin\ndatabase:\n  host: db-stg-1.internal\n  port: 5433\nreplicas: 2\n",
    "/home/config/dev.yaml": "service: orders-api\nenvironment: dev\ntimezone: America/Los_Angeles\ndatabase:\n  host: localhost\n  port: 5432\nreplicas: 1\n",
    "/home/reports/q1_summary.txt": "Q1 2026 summary (preliminary)\nRevenue: $1.84M (+12% QoQ)\nNew customers: 312\nChurn: 2.1%\n",
}
