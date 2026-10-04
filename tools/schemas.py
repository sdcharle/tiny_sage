"""JSON schemas for the mock tools, in OpenAI function-calling format.

Schemas are strict (additionalProperties: false) on purpose: T0 measures whether a model
emits calls that match the schema. Value-level checks such as date formats are done by the
tools themselves and come back as tool errors, so the model gets a chance to recover.
"""

from tools.fixtures import RATES

CURRENCIES = sorted(RATES)


def _fn(name, description, properties, required):
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


SCHEMAS = {s["name"]: s for s in [
    _fn("get_weather", "Get the current weather for a city.", {
        "city": {"type": "string", "description": "City name, e.g. 'Paris'."},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units. Defaults to celsius."},
    }, ["city"]),
    _fn("calculator", "Evaluate an arithmetic expression. Supports + - * / ** % and parentheses.", {
        "expression": {"type": "string", "description": "Python-style arithmetic expression, e.g. '(3 + 4) * 2'."},
    }, ["expression"]),
    _fn("search_web", "Search the web for recent news and articles.", {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum number of results. Defaults to 5."},
        "recency": {"type": "string", "enum": ["any", "day", "week", "month"], "description": "Only return results from this period. Defaults to 'any'."},
    }, ["query"]),
    _fn("read_file", "Read a text file from the user's filesystem.", {
        "path": {"type": "string", "description": "Absolute path, e.g. '/home/notes/todo.txt'."},
    }, ["path"]),
    _fn("list_files", "List the files in a directory of the user's filesystem.", {
        "directory": {"type": "string", "description": "Absolute directory path, e.g. '/home/notes'."},
    }, ["directory"]),
    _fn("find_user", "Find people in the company directory by name or email. Returns user_id, name, email and city.", {
        "query": {"type": "string", "description": "Full or partial name, or an email address."},
    }, ["query"]),
    _fn("get_orders", "List a user's orders, newest first, with status and total in USD.", {
        "user_id": {"type": "string", "description": "User ID, e.g. 'U1001'."},
        "status": {"type": "string", "enum": ["all", "pending", "shipped", "delivered"], "description": "Filter by status. Defaults to 'all'."},
    }, ["user_id"]),
    _fn("get_order", "Get full details of one order, including line items.", {
        "order_id": {"type": "string", "description": "Order ID in the format 'ORD-NNNN'."},
    }, ["order_id"]),
    _fn("convert_currency", "Convert an amount between currencies at today's exchange rate.", {
        "amount": {"type": "number", "description": "Amount to convert."},
        "from_currency": {"type": "string", "enum": CURRENCIES, "description": "ISO currency code to convert from."},
        "to_currency": {"type": "string", "enum": CURRENCIES, "description": "ISO currency code to convert to."},
    }, ["amount", "from_currency", "to_currency"]),
    _fn("get_time", "Get the current local date and time in a timezone.", {
        "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'Europe/London'."},
    }, ["timezone"]),
    _fn("get_stock_price", "Get the latest stock price in USD for a ticker symbol.", {
        "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'."},
    }, ["ticker"]),
    _fn("send_email", "Send an email on the user's behalf.", {
        "to": {"type": "string", "description": "Recipient email address."},
        "subject": {"type": "string"},
        "body": {"type": "string"},
        "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC email addresses."},
    }, ["to", "subject", "body"]),
    _fn("create_event", "Create a calendar event.", {
        "title": {"type": "string"},
        "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
        "start_time": {"type": "string", "description": "24-hour start time in HH:MM format."},
        "duration_minutes": {"type": "integer", "minimum": 5, "description": "Duration in minutes. Defaults to 60."},
        "attendees": {"type": "array", "items": {"type": "string"}, "description": "Optional attendee email addresses."},
        "location": {"type": "string", "description": "Optional location."},
    }, ["title", "date", "start_time"]),
]}

ALL_TOOLS = list(SCHEMAS)


def openai_tools(names):
    return [{"type": "function", "function": SCHEMAS[n]} for n in names]
