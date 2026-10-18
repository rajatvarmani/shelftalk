"""ShelfTalk agent: Gemini + function calling over the shop's ledger.

One turn = the shopkeeper's photo(s) + voice note + text  ->  Gemini plans,
calls ledger tools (sales, restocks, shelf counts, reorders), then replies in
the shopkeeper's own language. Every tool call is recorded in a trace that the
UI shows, and every write is tagged with the turn's request_id so it can be
undone in one tap.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import urllib.parse
import uuid

from .store import get_store

log = logging.getLogger("shelftalk.agent")

MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3-flash-preview,gemini-2.5-flash").split(",") if m.strip()]
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "10"))

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_trace: contextvars.ContextVar[list] = contextvars.ContextVar("trace")


def _shop():
    return get_store().state()["shop"]


def _not_found(name: str, suggestions: list[str]) -> dict:
    return {"ok": False, "error": f"No item matches '{name}'.",
            "did_you_mean": suggestions,
            "hint": "Use a name from list_inventory, or add_new_item if it is genuinely new."}


# ---------------------------------------------------------------------------
# Tools exposed to Gemini. Docstrings + type hints become the function schema.
# ---------------------------------------------------------------------------

def list_inventory() -> dict:
    """List every product in the shop with current stock, unit, reorder level,
    status (ok / low / out), selling price and supplier. Call this first whenever
    you need exact product names or current numbers."""
    inv = get_store().inventory()
    return {"currency": _shop().get("currency"), "items": [
        {"name": i["name"], "stock": i["stock"], "unit": i["unit"],
         "reorder_level": i.get("reorder_level"), "status": i["status"],
         "sell_price": i.get("sell_price"), "cost_price": i.get("cost_price"),
         "supplier": i.get("supplier_name")} for i in inv]}


def record_sale(item_name: str, quantity: float, unit_price: float = 0) -> dict:
    """Record that the shop SOLD some units of a product (stock goes down).

    Args:
        item_name: Product name as it appears in the inventory (fuzzy matching is applied).
        quantity: Number of units sold. Must be positive.
        unit_price: Price per unit actually charged. Use 0 to use the normal selling price.
    """
    store = get_store()
    item, sugg = store.resolve_item(store.state(), item_name)
    if not item:
        return _not_found(item_name, sugg)
    if quantity <= 0:
        return {"ok": False, "error": "quantity must be positive"}
    price = unit_price or item.get("sell_price", 0)
    res = store.apply_change(item["id"], -quantity, "sale", _request_id.get(), price)
    out = {"ok": True, "item": item["name"], "sold": quantity, "stock_before": res["before"],
           "stock_now": res["after"], "amount": round(quantity * price, 2),
           "status": store.item_status(res["item"])}
    if res["before"] < quantity:
        out["warning"] = (f"Only {res['before']} were on record, so stock was floored at 0. "
                          "The record was probably out of date.")
    return out


def record_restock(item_name: str, quantity: float, unit_cost: float = 0) -> dict:
    """Record that stock ARRIVED for a product, e.g. from a supplier delivery or bill (stock goes up).

    Args:
        item_name: Product name as it appears in the inventory.
        quantity: Number of units received. Must be positive.
        unit_cost: Purchase price per unit on the bill. Use 0 if unknown.
    """
    store = get_store()
    item, sugg = store.resolve_item(store.state(), item_name)
    if not item:
        return _not_found(item_name, sugg)
    if quantity <= 0:
        return {"ok": False, "error": "quantity must be positive"}
    out = {}
    if unit_cost and abs(unit_cost - item.get("cost_price", 0)) > 0.009:
        old = item.get("cost_price", 0)
        store.update_item(item["id"], {"cost_price": unit_cost})
        margin = item.get("sell_price", 0) - unit_cost
        out["price_change"] = {"old_cost": old, "new_cost": unit_cost,
                               "margin_per_unit_now": round(margin, 2),
                               "note": "Cost changed versus the last bill. Mention this to the shopkeeper."}
    res = store.apply_change(item["id"], quantity, "restock", _request_id.get(), unit_cost or None)
    out.update({"ok": True, "item": item["name"], "received": quantity,
                "stock_before": res["before"], "stock_now": res["after"],
                "status": store.item_status(res["item"])})
    return out


def set_counted_stock(item_name: str, counted_quantity: float, note: str = "") -> dict:
    """Overwrite the recorded stock with a PHYSICAL COUNT, e.g. units you can clearly
    count in a shelf photo, or a number the shopkeeper says they counted.
    Only use for products you can count with confidence.

    Args:
        item_name: Product name as it appears in the inventory.
        counted_quantity: The number of units actually present.
        note: Short note on where the count came from, e.g. "shelf photo, top row".
    """
    store = get_store()
    item, sugg = store.resolve_item(store.state(), item_name)
    if not item:
        return _not_found(item_name, sugg)
    delta = counted_quantity - item["stock"]
    res = store.apply_change(item["id"], delta, "count", _request_id.get(), None, note)
    return {"ok": True, "item": item["name"], "recorded_before": res["before"],
            "counted": res["after"], "difference": round(delta, 3),
            "status": store.item_status(res["item"]),
            "note": "A large negative difference may mean unrecorded sales or shrinkage."}


def add_new_item(name: str, unit: str, sell_price: float, cost_price: float = 0,
                 opening_stock: float = 0, reorder_level: float = 5,
                 supplier_name: str = "") -> dict:
    """Add a product the shop has never stocked before. Check list_inventory first
    so you do not create duplicates.

    Args:
        name: Clear product name with pack size, e.g. "Bru Coffee 50g".
        unit: Counting unit, e.g. pack, kg, bottle, tray.
        sell_price: Selling price per unit.
        cost_price: Purchase price per unit (0 if unknown).
        opening_stock: Units in the shop right now.
        reorder_level: Reorder when stock falls to this level.
        supplier_name: Supplier who delivers it, if known.
    """
    store = get_store()
    state = store.state()
    existing, _ = store.resolve_item(state, name)
    if existing and existing["name"].lower() == name.lower():
        return {"ok": False, "error": f"'{existing['name']}' already exists."}
    sup = store.resolve_supplier(state, supplier_name) if supplier_name else None
    item = store.add_item({"name": name, "unit": unit, "sell_price": sell_price,
                           "cost_price": cost_price, "stock": opening_stock,
                           "reorder_level": reorder_level,
                           "supplier_id": sup["id"] if sup else None,
                           "aliases": []}, _request_id.get())
    return {"ok": True, "added": item["name"], "stock": item["stock"],
            "supplier": sup["name"] if sup else None}


def get_low_stock() -> dict:
    """List products that are out of stock or at/below their reorder level,
    grouped by supplier, with a suggested order quantity (about one week of sales)."""
    import time as _t

    store = get_store()
    state = store.state()
    since = _t.time() - 7 * 86400
    weekly = {}
    for e in state["ledger"]:
        if e["kind"] == "sale" and e["ts"] >= since:
            weekly[e["item_id"]] = weekly.get(e["item_id"], 0) - e["qty"]
    groups: dict[str, list] = {}
    for i in store.inventory():
        if i["status"] == "ok":
            continue
        need = max(i.get("reorder_level", 0) * 2, weekly.get(i["id"], 0)) - i["stock"]
        groups.setdefault(i.get("supplier_name") or "No supplier", []).append({
            "item": i["name"], "stock": i["stock"], "unit": i["unit"], "status": i["status"],
            "sold_last_7_days": weekly.get(i["id"], 0),
            "suggested_order": max(1, round(need))})
    return {"by_supplier": groups, "count": sum(len(v) for v in groups.values())}


def sales_report(days: int = 7) -> dict:
    """Revenue, profit, best-selling products and daily revenue for the last N days.

    Args:
        days: How many days back to look (1 = today only).
    """
    rep = get_store().report(max(1, min(int(days), 90)))
    rep["currency"] = _shop().get("currency")
    return rep


def list_suppliers() -> dict:
    """List suppliers with the language they prefer and what they supply."""
    return {"suppliers": [{"name": s["name"], "language": s.get("language"),
                           "supplies": s.get("categories")}
                          for s in get_store().state()["suppliers"].values()]}


def draft_supplier_order(supplier_name: str, message: str) -> dict:
    """Save a ready-to-send WhatsApp reorder message for a supplier. Write the
    message yourself in the SUPPLIER'S preferred language: polite, short, with
    one line per product and quantity, signed with the shop name.

    Args:
        supplier_name: Supplier to order from (see list_suppliers).
        message: The complete message text to send.
    """
    store = get_store()
    sup = store.resolve_supplier(store.state(), supplier_name)
    if not sup:
        return {"ok": False, "error": f"Unknown supplier '{supplier_name}'."}
    link = f"https://wa.me/{sup['phone']}?text={urllib.parse.quote(message)}"
    store.save_draft({"supplier": sup["name"], "language": sup.get("language"),
                      "message": message, "whatsapp_link": link})
    return {"ok": True, "supplier": sup["name"], "saved": True,
            "note": "Draft saved. The shopkeeper reviews and taps Send on WhatsApp."}


TOOLS = [list_inventory, record_sale, record_restock, set_counted_stock, add_new_item,
         get_low_stock, sales_report, list_suppliers, draft_supplier_order]
TOOL_MAP = {f.__name__: f for f in TOOLS}
WRITE_TOOLS = {"record_sale", "record_restock", "set_counted_stock", "add_new_item",
               "draft_supplier_order"}


def _declaration(fn) -> dict:
    """Build an explicit Gemini function declaration from a tool's signature + docstring.

    Explicit schemas avoid SDK auto-introspection quirks (e.g. default values)."""
    import inspect
    import re

    doc = inspect.getdoc(fn) or ""
    desc, _, args_block = doc.partition("Args:")
    arg_desc = {m.group(1): " ".join(m.group(2).split())
                for m in re.finditer(r"^\s*(\w+):\s*(.+(?:\n\s{8,}.+)*)", args_block, re.M)}
    tmap = {str: "STRING", float: "NUMBER", int: "INTEGER", bool: "BOOLEAN"}
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        ann = p.annotation if not isinstance(p.annotation, str) else {"str": str, "float": float,
                                                                      "int": int, "bool": bool}[p.annotation]
        props[name] = {"type": tmap[ann], "description": arg_desc.get(name, name)}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    decl = {"name": fn.__name__, "description": " ".join(desc.split())}
    if props:
        decl["parameters"] = {"type": "OBJECT", "properties": props, "required": required}
    return decl


def system_prompt() -> str:
    shop = _shop()
    return f"""You are ShelfTalk, the back-office assistant for {shop.get('name')}, a small
neighbourhood shop in {shop.get('city')} run by {shop.get('owner')}. Currency: {shop.get('currency')}.

The shopkeeper is busy at the counter. They talk to you with photos, voice notes and
short texts, in any language or a mix (Hindi, Marathi, Tamil, Bahasa, English, Hinglish...).

What you do:
- Keep the stock ledger correct using your tools. Never pretend to update anything without
  calling a tool, and never invent numbers.
- Voice notes and texts: extract every sale, delivery and count mentioned and record each
  one. "10 Maggi gaye" = 10 Maggi sold. "Gupta ji ka maal aaya" = a delivery arrived.
- Photo of a supplier bill or invoice: read each line (product, quantity, unit cost) and
  call record_restock for each one. Flag cost increases.
- Photo of a shelf: identify products, count only what you can clearly count, and call
  set_counted_stock for those. Say which ones you could not count and why.
- Photo of a handwritten notebook (khata) page: read the entries and record them.
- Questions about the business: use sales_report / get_low_stock and answer with numbers.
- When things are low, offer to draft supplier orders; when asked, write the message in
  the supplier's preferred language and save it with draft_supplier_order.
- Call list_inventory first if you are unsure of exact product names.
- If something is genuinely ambiguous (which product, or sold vs received), ask one short
  question instead of guessing.

How you reply:
- Reply in the SAME language and script the shopkeeper used (Hinglish in, Hinglish out).
- Be brief: 1 to 5 short lines, like a trusted munim. Lead with what you changed.
- Use the currency symbol for money. No markdown tables, no headings.
"""


def _summarise(name: str, args: dict, result: dict) -> str:
    if not isinstance(result, dict) or not result.get("ok", True):
        return f"{name}: {result.get('error', 'failed') if isinstance(result, dict) else 'failed'}"
    if name == "record_sale":
        return f"Sold {result['sold']:g} × {result['item']} → {result['stock_now']:g} left"
    if name == "record_restock":
        return f"Received {result['received']:g} × {result['item']} → {result['stock_now']:g} in stock"
    if name == "set_counted_stock":
        return f"Counted {result['item']}: {result['counted']:g} (was {result['recorded_before']:g})"
    if name == "add_new_item":
        return f"Added new item {result['added']}"
    if name == "draft_supplier_order":
        return f"Drafted order for {result['supplier']}"
    if name == "get_low_stock":
        return f"Checked low stock: {result['count']} items need attention"
    if name == "sales_report":
        return f"Pulled {result['days']}-day sales report"
    if name == "list_inventory":
        return f"Read inventory ({len(result['items'])} items)"
    if name == "list_suppliers":
        return "Read supplier list"
    return name


def dispatch(name: str, args: dict) -> dict:
    fn = TOOL_MAP.get(name)
    if not fn:
        result = {"ok": False, "error": f"unknown tool {name}"}
    else:
        try:
            result = fn(**args)
        except TypeError as e:
            result = {"ok": False, "error": f"bad arguments: {e}"}
        except Exception as e:  # keep the agent loop alive
            log.exception("tool %s failed", name)
            result = {"ok": False, "error": str(e)}
    _trace.get().append({"tool": name, "args": args, "write": name in WRITE_TOOLS,
                         "ok": bool(result.get("ok", True)),
                         "item": result.get("item") or result.get("added"),
                         "summary": _summarise(name, args, result)})
    return result


# ---------------------------------------------------------------------------
# Gemini loop
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()  # Vertex AI if GOOGLE_GENAI_USE_VERTEXAI=true, else GEMINI_API_KEY
    return _client


def set_client(client) -> None:
    """Used by tests to inject a fake client."""
    global _client
    _client = client


def run_turn(text: str = "", images: list[tuple[bytes, str]] | None = None,
             audio: tuple[bytes, str] | None = None, history: list[dict] | None = None) -> dict:
    from google.genai import types

    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    trace: list = []
    _trace.set(trace)

    contents = []
    for h in (history or [])[-8:]:
        if h.get("text"):
            contents.append(types.Content(role="user" if h.get("role") == "user" else "model",
                                          parts=[types.Part(text=h["text"])]))
    parts = []
    for data, mime in images or []:
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    if audio:
        parts.append(types.Part.from_bytes(data=audio[0], mime_type=audio[1]))
    kinds = []
    if images:
        kinds.append(f"{len(images)} photo(s)")
    if audio:
        kinds.append("a voice note")
    prompt = text.strip()
    if kinds and not prompt:
        prompt = (f"The shopkeeper sent {' and '.join(kinds)} with no text. Work out what it is "
                  "(sale, delivery bill, shelf, khata page, question) and act on it. If it is a "
                  "voice note, reply in the language spoken.")
    parts.append(types.Part(text=prompt or "Hello"))
    contents.append(types.Content(role="user", parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt(),
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**_declaration(f))
                                                 for f in TOOLS])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    client = get_client()
    models = list(MODELS)
    reply = ""
    used_model = models[0]
    for _ in range(MAX_STEPS):
        resp = None
        while models:
            try:
                resp = client.models.generate_content(model=models[0], contents=contents, config=config)
                used_model = models[0]
                break
            except Exception as e:  # model not available in this project/region -> next one
                msg = str(e).lower()
                if len(models) > 1 and ("not found" in msg or "404" in msg or "not supported" in msg):
                    log.warning("model %s unavailable (%s), falling back", models[0], e)
                    models.pop(0)
                    continue
                raise
        if not resp or not resp.candidates:
            reply = "Sorry, I could not process that. Please try again."
            break
        contents.append(resp.candidates[0].content)  # keeps Gemini 3 thought signatures intact
        calls = resp.function_calls or []
        if not calls:
            reply = (resp.text or "").strip()
            break
        fr_parts = []
        for fc in calls:
            result = dispatch(fc.name, dict(fc.args or {}))
            fr_parts.append(types.Part.from_function_response(
                name=fc.name, response={"result": json.loads(json.dumps(result, default=str))}))
        contents.append(types.Content(role="user", parts=fr_parts))
    else:
        reply = "I did a lot of steps there. Check the activity list below."

    return {"request_id": request_id, "reply": reply, "trace": trace, "model": used_model,
            "changed": any(t["write"] and t["ok"] for t in trace)}


BRIEF_PROMPT = ("Give me today's morning brief: yesterday's and this week's sales with profit, "
                "the top sellers, what is out or running low grouped by supplier, and one practical "
                "suggestion. Keep it to 6 short lines. Reply in simple Hinglish.")
