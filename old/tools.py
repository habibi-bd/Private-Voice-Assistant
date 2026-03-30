import json

def add_event(title: str, date: str) -> dict:
    # Replace with actual calendar integration
    return {"status": "ok", "title": title, "date": date}

TOOLS = {
    "add_event": add_event,
}

def run_tool(call_json: str) -> str:
    """call_json: '{"tool":"add_event","args":{"title":"Dentist","date":"2025-11-10 10:00"}}'"""
    data = json.loads(call_json)
    name = data["tool"]
    args = data.get("args", {})
    if name in TOOLS:
        result = TOOLS[name](**args)
        return json.dumps({"tool_result": result})
    return json.dumps({"error": "unknown tool"})