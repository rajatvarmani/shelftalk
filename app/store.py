"""Storage layer for ShelfTalk.

Two backends share one interface:
  * LocalStore     - a JSON file (great for local dev and demos)
  * FirestoreStore - Google Cloud Firestore (used on Cloud Run, survives restarts)

Pick with env STORE=firestore|local (default: local).
"""
from __future__ import annotations

import copy
import difflib
import json
import os
import threading
import time
import uuid
from pathlib import Path

SEED_PATH = Path(__file__).with_name("seed_data.json")


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").split())


def now_ts() -> float:
    return time.time()


class BaseStore:
    """Shared business logic. Subclasses implement load/save of raw state."""

    lock = threading.RLock()

    # ---- raw state (overridden) -------------------------------------------
    def _load(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _save(self, state: dict, changed: set[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    # ---- helpers ------------------------------------------------------------
    def state(self) -> dict:
        return self._load()

    def reset(self) -> None:
        seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
        state = build_seed_state(seed)
        with self.lock:
            self._save(state, {"items", "suppliers", "ledger", "drafts", "shop"})

    def resolve_item(self, state: dict, name: str):
        """Fuzzy-match an item by name or alias (works for Hindi/English/romanised)."""
        q = _norm(name)
        if not q:
            return None, []
        best, best_score = None, 0.0
        for item in state["items"].values():
            cands = [item["name"], *item.get("aliases", [])]
            for c in cands:
                c = _norm(c)
                if c == q:
                    return item, []
                score = difflib.SequenceMatcher(None, q, c).ratio()
                if q in c or c in q:
                    score = max(score, 0.85)
                if score > best_score:
                    best, best_score = item, score
        if best and best_score >= 0.72:
            return best, []
        names = [i["name"] for i in state["items"].values()]
        return None, difflib.get_close_matches(name, names, n=3, cutoff=0.3)

    def resolve_supplier(self, state: dict, name: str):
        q = _norm(name)
        for s in state["suppliers"].values():
            if _norm(s["name"]) == q or q in _norm(s["name"]):
                return s
        names = {s["name"]: s for s in state["suppliers"].values()}
        m = difflib.get_close_matches(name, list(names), n=1, cutoff=0.5)
        return names[m[0]] if m else None

    # ---- mutations ------------------------------------------------------------
    def apply_change(self, item_id: str, delta: float, kind: str, request_id: str,
                     unit_price: float | None = None, note: str = "") -> dict:
        with self.lock:
            state = self._load()
            item = state["items"][item_id]
            before = item["stock"]
            item["stock"] = round(max(0.0, before + delta), 3)
            entry = {
                "id": uuid.uuid4().hex[:10],
                "ts": now_ts(),
                "item_id": item_id,
                "item_name": item["name"],
                "kind": kind,  # sale | restock | count | new
                "qty": round(item["stock"] - before, 3),
                "unit_price": unit_price,
                "note": note,
                "request_id": request_id,
            }
            state["ledger"].append(entry)
            self._save(state, {"items", "ledger"})
            return {"item": item, "before": before, "after": item["stock"], "entry": entry}

    def add_item(self, item: dict, request_id: str) -> dict:
        with self.lock:
            state = self._load()
            item_id = "i_" + uuid.uuid4().hex[:8]
            item = {"id": item_id, "aliases": [], "stock": 0, **item}
            item["id"] = item_id
            state["items"][item_id] = item
            state["ledger"].append({
                "id": uuid.uuid4().hex[:10], "ts": now_ts(), "item_id": item_id,
                "item_name": item["name"], "kind": "new", "qty": item["stock"],
                "unit_price": item.get("cost_price"), "note": "new item",
                "request_id": request_id,
            })
            self._save(state, {"items", "ledger"})
            return item

    def update_item(self, item_id: str, fields: dict) -> dict:
        with self.lock:
            state = self._load()
            state["items"][item_id].update(fields)
            self._save(state, {"items"})
            return state["items"][item_id]

    def save_draft(self, draft: dict) -> dict:
        with self.lock:
            state = self._load()
            draft = {"id": uuid.uuid4().hex[:10], "ts": now_ts(), **draft}
            state["drafts"].insert(0, draft)
            state["drafts"] = state["drafts"][:20]
            self._save(state, {"drafts"})
            return draft

    def undo(self, request_id: str) -> int:
        """Revert every ledger entry written by one agent turn."""
        with self.lock:
            state = self._load()
            undone = 0
            keep = []
            for e in reversed(state["ledger"]):
                if e.get("request_id") == request_id:
                    item = state["items"].get(e["item_id"])
                    if item is not None:
                        if e["kind"] == "new":
                            state["items"].pop(e["item_id"], None)
                        else:
                            item["stock"] = round(max(0.0, item["stock"] - e["qty"]), 3)
                    undone += 1
                else:
                    keep.append(e)
            state["ledger"] = list(reversed(keep))
            if undone:
                self._save(state, {"items", "ledger"})
            return undone

    # ---- reads -----------------------------------------------------------------
    def item_status(self, item: dict) -> str:
        if item["stock"] <= 0:
            return "out"
        if item["stock"] <= item.get("reorder_level", 0):
            return "low"
        return "ok"

    def inventory(self) -> list[dict]:
        state = self._load()
        sup = state["suppliers"]
        out = []
        for i in state["items"].values():
            out.append({
                **i,
                "status": self.item_status(i),
                "supplier_name": sup.get(i.get("supplier_id"), {}).get("name"),
            })
        order = {"out": 0, "low": 1, "ok": 2}
        return sorted(out, key=lambda x: (order[x["status"]], x["name"]))

    def report(self, days: int = 7) -> dict:
        state = self._load()
        since = now_ts() - days * 86400
        revenue = profit = 0.0
        by_item: dict[str, dict] = {}
        by_day: dict[str, float] = {}
        for e in state["ledger"]:
            if e["kind"] != "sale" or e["ts"] < since:
                continue
            item = state["items"].get(e["item_id"])
            qty = -e["qty"]
            price = e.get("unit_price") or (item or {}).get("sell_price", 0)
            cost = (item or {}).get("cost_price", 0)
            revenue += qty * price
            profit += qty * (price - cost)
            b = by_item.setdefault(e["item_name"], {"item": e["item_name"], "units": 0, "revenue": 0.0, "profit": 0.0})
            b["units"] += qty
            b["revenue"] += qty * price
            b["profit"] += qty * (price - cost)
            day = time.strftime("%Y-%m-%d", time.localtime(e["ts"]))
            by_day[day] = by_day.get(day, 0) + qty * price
        top = sorted(by_item.values(), key=lambda x: -x["revenue"])
        for t in top:
            t["revenue"] = round(t["revenue"], 2)
            t["profit"] = round(t["profit"], 2)
        return {
            "days": days,
            "revenue": round(revenue, 2),
            "profit": round(profit, 2),
            "top_items": top[:8],
            "by_day": [{"day": d, "revenue": round(v, 2)} for d, v in sorted(by_day.items())],
        }


def build_seed_state(seed: dict) -> dict:
    """Expand seed_data.json into a full state with ~7 days of sales history."""
    import random

    rng = random.Random(42)
    items = {i["id"]: dict(i) for i in seed["items"]}
    suppliers = {s["id"]: dict(s) for s in seed["suppliers"]}
    ledger = []
    t0 = now_ts()
    for day in range(7, 0, -1):
        for item in items.values():
            daily = item.get("daily_sales", 0)
            if not daily:
                continue
            qty = max(0, round(rng.gauss(daily, daily * 0.3)))
            if not qty:
                continue
            ledger.append({
                "id": uuid.uuid4().hex[:10],
                "ts": t0 - day * 86400 + rng.randint(8, 20) * 3600 - 43200,
                "item_id": item["id"], "item_name": item["name"], "kind": "sale",
                "qty": -qty, "unit_price": item["sell_price"], "note": "seed",
                "request_id": "seed",
            })
    ledger.sort(key=lambda e: e["ts"])
    for item in items.values():
        item.pop("daily_sales", None)
    return {"shop": seed["shop"], "items": items, "suppliers": suppliers,
            "ledger": ledger, "drafts": []}


class LocalStore(BaseStore):
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("LOCAL_DB", "/tmp/shelftalk_db.json"))
        if not self.path.exists():
            self.reset()

    def _load(self) -> dict:
        with self.lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict, changed: set[str]) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)


class FirestoreStore(BaseStore):
    """Firestore backend: one doc per collection-chunk keeps writes cheap and atomic.

    Layout:  shelftalk/{shop_id}  -> {shop, items, suppliers, drafts}
             shelftalk/{shop_id}/ledger/{entry_id}
    The ledger is the only part that grows, so it lives in a subcollection.
    """

    def __init__(self, shop_id: str | None = None):
        from google.cloud import firestore  # imported lazily

        self.db = firestore.Client()
        self.shop_id = shop_id or os.environ.get("SHOP_ID", "demo-shop")
        self.doc = self.db.collection("shelftalk").document(self.shop_id)
        self._cache: dict | None = None
        if not self.doc.get().exists:
            self.reset()

    def _load(self) -> dict:
        with self.lock:
            snap = self.doc.get().to_dict() or {}
            ledger = [d.to_dict() for d in self.doc.collection("ledger").order_by("ts").stream()]
            return {
                "shop": snap.get("shop", {}),
                "items": snap.get("items", {}),
                "suppliers": snap.get("suppliers", {}),
                "drafts": snap.get("drafts", []),
                "ledger": ledger,
            }

    def _save(self, state: dict, changed: set[str]) -> None:
        with self.lock:
            top = {k: state[k] for k in ("shop", "items", "suppliers", "drafts") if k in changed}
            if top:
                # merge=[fields] replaces those fields wholesale (so deleted items vanish)
                self.doc.set(top, merge=list(top))
            if "ledger" in changed:
                col = self.doc.collection("ledger")
                existing = {d.id for d in col.select([]).stream()}
                wanted = {e["id"]: e for e in state["ledger"]}
                batch = self.db.batch()
                n = 0
                for eid in existing - set(wanted):
                    batch.delete(col.document(eid)); n += 1
                for eid in set(wanted) - existing:
                    batch.set(col.document(eid), wanted[eid]); n += 1
                    if n >= 450:
                        batch.commit(); batch = self.db.batch(); n = 0
                if n:
                    batch.commit()


_store: BaseStore | None = None


def get_store() -> BaseStore:
    global _store
    if _store is None:
        if os.environ.get("STORE", "local").lower() == "firestore":
            _store = FirestoreStore()
        else:
            _store = LocalStore()
    return _store
