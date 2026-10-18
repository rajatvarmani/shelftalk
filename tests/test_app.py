"""End-to-end tests with a scripted fake Gemini client (no network or API key needed).

Run:  python -m pytest -q   (or: python tests/test_app.py)
"""
import io
import json
import os
import sys
import tempfile
import types as pytypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["STORE"] = "local"
os.environ["LOCAL_DB"] = str(Path(tempfile.mkdtemp()) / "db.json")


# ---------------------------------------------------------------- fake SDK
def install_fake_genai():
    """Register a minimal stand-in for google.genai only if the real SDK is missing."""
    try:
        import google.genai  # noqa: F401
        return
    except Exception:
        pass
    g = sys.modules.get("google") or pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    t = pytypes.ModuleType("google.genai.types")

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Part(_Obj):
        @staticmethod
        def from_bytes(data, mime_type):
            return Part(inline=(len(data), mime_type))

        @staticmethod
        def from_function_response(name, response):
            json.dumps(response)  # must be JSON-serialisable, as with the real API
            return Part(function_response={"name": name, "response": response})

    t.Part = Part
    t.Content = type("Content", (_Obj,), {})
    t.GenerateContentConfig = type("GenerateContentConfig", (_Obj,), {})
    t.AutomaticFunctionCallingConfig = type("AutomaticFunctionCallingConfig", (_Obj,), {})
    t.Tool = type("Tool", (_Obj,), {})
    t.FunctionDeclaration = type("FunctionDeclaration", (_Obj,), {})
    genai.types = t
    genai.Client = lambda: None
    g.genai = genai
    sys.modules["google"] = g
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = t


install_fake_genai()

from app import agent  # noqa: E402
from app.main import app  # noqa: E402
from app.store import get_store  # noqa: E402


class FC:
    def __init__(self, name, args):
        self.name, self.args = name, args


class Resp:
    def __init__(self, calls=None, text=""):
        self.function_calls = calls or None
        self.text = text
        self.candidates = [pytypes.SimpleNamespace(content={"role": "model", "calls": calls, "text": text})]


class FakeModels:
    def __init__(self, script, fail_first_model=False):
        self.script = list(script)
        self.calls = []
        self.fail_first_model = fail_first_model

    def generate_content(self, model, contents, config):
        self.calls.append((model, len(contents)))
        if self.fail_first_model and model == agent.MODELS[0]:
            raise RuntimeError("404 NOT_FOUND: model not found")
        assert config.tools and config.system_instruction
        return self.script.pop(0)


def fake_client(script, **kw):
    return pytypes.SimpleNamespace(models=FakeModels(script, **kw))


def setup():
    get_store().reset()


def stock(name):
    return next(i for i in get_store().inventory() if i["name"] == name)["stock"]


# ---------------------------------------------------------------- tests
def test_voice_note_turn_records_sales_and_delivery():
    setup()
    m0, a0, d0 = stock("Maggi Noodles 70g"), stock("Aashirvaad Atta 5kg"), stock("Toor Dal 1kg")
    agent.set_client(fake_client([
        Resp([FC("record_sale", {"item_name": "maggi", "quantity": 10}),
              FC("record_sale", {"item_name": "आटा", "quantity": 2}),
              FC("record_restock", {"item_name": "arhar dal", "quantity": 10, "unit_cost": 150})]),
        Resp(text="✓ 10 Maggi, 2 Atta bik gaye. 10 Toor Dal aayi — cost ₹140 se ₹150 ho gaya."),
    ]))
    r = agent.run_turn(text="", audio=(b"x" * 2000, "audio/wav"))
    assert r["changed"] and len(r["trace"]) == 3
    assert stock("Maggi Noodles 70g") == m0 - 10
    assert stock("Aashirvaad Atta 5kg") == a0 - 2
    assert stock("Toor Dal 1kg") == d0 + 10
    item = next(i for i in get_store().inventory() if i["name"] == "Toor Dal 1kg")
    assert item["cost_price"] == 150
    assert r["trace"][2]["item"] == "Toor Dal 1kg"
    # undo reverts the whole turn
    n = get_store().undo(r["request_id"])
    assert n == 3
    assert stock("Maggi Noodles 70g") == m0 and stock("Toor Dal 1kg") == d0


def test_unknown_item_returns_suggestions_not_crash():
    setup()
    agent._trace.set([])
    agent._request_id.set("t")
    out = agent.dispatch("record_sale", {"item_name": "xyz unicorn", "quantity": 1})
    assert out["ok"] is False and "did_you_mean" in out
    out = agent.dispatch("record_sale", {"item_name": "maggi", "quantity": 1, "bogus": 3})
    assert out["ok"] is False and "bad arguments" in out["error"]


def test_shelf_count_and_low_stock_and_reorder_draft():
    setup()
    agent._trace.set([])
    agent._request_id.set("t2")
    out = agent.dispatch("set_counted_stock", {"item_name": "Parle-G", "counted_quantity": 7, "note": "shelf photo"})
    assert out["ok"] and out["counted"] == 7
    low = agent.dispatch("get_low_stock", {})
    names = [i["item"] for v in low["by_supplier"].values() for i in v]
    assert "Parle-G Biscuit 100g" in names and "Lay's Chips 50g" in names
    assert all(i["suggested_order"] >= 1 for v in low["by_supplier"].values() for i in v)
    d = agent.dispatch("draft_supplier_order", {"supplier_name": "gupta", "message": "नमस्ते गुप्ता जी,\n10 तूर दाल\n- साई किराना"})
    assert d["ok"]
    draft = get_store().state()["drafts"][0]
    assert draft["whatsapp_link"].startswith("https://wa.me/919800000001?text=")


def test_add_new_item_and_report():
    setup()
    agent._trace.set([])
    agent._request_id.set("t3")
    out = agent.dispatch("add_new_item", {"name": "Bru Coffee 50g", "unit": "jar", "sell_price": 95,
                                          "cost_price": 82, "opening_stock": 6, "supplier_name": "Metro"})
    assert out["ok"] and out["supplier"] == "Metro FMCG Distributors"
    dup = agent.dispatch("add_new_item", {"name": "Bru Coffee 50g", "unit": "jar", "sell_price": 95})
    assert dup["ok"] is False
    rep = agent.dispatch("sales_report", {"days": 7})
    assert rep["revenue"] > 0 and rep["profit"] > 0 and rep["top_items"]
    assert get_store().undo("t3") == 1
    assert not any(i["name"] == "Bru Coffee 50g" for i in get_store().inventory())


def test_tool_declarations():
    decls = {d["name"]: d for d in map(agent._declaration, agent.TOOLS)}
    rs = decls["record_sale"]
    assert rs["parameters"]["required"] == ["item_name", "quantity"]
    assert rs["parameters"]["properties"]["quantity"]["type"] == "NUMBER"
    assert "Price per unit" in rs["parameters"]["properties"]["unit_price"]["description"]
    assert "Args" not in rs["description"]
    assert "parameters" not in decls["list_inventory"]
    assert decls["sales_report"]["parameters"]["properties"]["days"]["type"] == "INTEGER"


def test_model_fallback():
    setup()
    agent.set_client(fake_client([Resp(text="Namaste!")], fail_first_model=True))
    r = agent.run_turn(text="hi")
    assert r["reply"] == "Namaste!" and r["model"] == agent.MODELS[1]


def test_http_endpoints():
    setup()
    c = app.test_client()
    assert c.get("/healthz").json["ok"]
    assert b"ShelfTalk" in c.get("/").data
    s = c.get("/api/state").json
    assert s["inventory"] and s["report"]["revenue"] > 0
    assert c.post("/api/turn", data={"text": ""}).status_code == 400

    agent.set_client(fake_client([
        Resp([FC("list_inventory", {})]),
        Resp([FC("set_counted_stock", {"item_name": "Coca-Cola 750ml", "counted_quantity": 4})]),
        Resp(text="Shelf par 4 Coke dikhe. Record 12 tha — 8 ka farak hai."),
    ]))
    r = c.post("/api/turn", data={"text": "", "history": "[]",
                                  "images": (io.BytesIO(b"\xff\xd8fakejpeg"), "shelf.jpg", "image/jpeg")},
               content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    j = r.json
    assert j["changed"] and j["state"]["inventory"]
    assert stock("Coca-Cola 750ml") == 4
    u = c.post("/api/undo", json={"request_id": j["request_id"]}).json
    assert u["undone"] == 1 and stock("Coca-Cola 750ml") == 12

    agent.set_client(fake_client([Resp([FC("sales_report", {"days": 1}), FC("get_low_stock", {})]),
                                  Resp(text="Kal ₹2,100 ki bikri...")]))
    b = c.post("/api/brief").json
    assert b["reply"].startswith("Kal") and not b["changed"]

    class Boom:
        class models:
            @staticmethod
            def generate_content(**kw):
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
    agent.set_client(Boom)
    e = c.post("/api/turn", data={"text": "hi"})
    assert e.status_code == 502 and "rate limit" in e.json["error"]


def test_audio_webm_converted_to_wav():
    from app.main import to_wav
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    d = tempfile.mkdtemp()
    src = Path(d, "a.webm")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:a", "libopus", str(src)], check=True)
    data, mime = to_wav(src.read_bytes(), "audio/webm;codecs=opus")
    assert mime == "audio/wav" and data[:4] == b"RIFF"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
                import traceback; traceback.print_exc()
    sys.exit(1 if fails else 0)
