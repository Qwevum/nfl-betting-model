"""Append-only, hash-chained records: forecast history and the placed-bet ledger.

Each line of a .jsonl store is one record:
    {"seq": n, "recorded_utc": ..., "kind": ..., "data": {...}, "prev": <hash of previous>, "hash": <sha256>}
hash = sha256(canonical JSON of the record without "hash"). Editing, deleting or
reordering any line breaks the chain, which `verify()` detects; `append()` refuses
to write to a broken store. Nothing in this module rewrites a file.

Forecasts (logs/forecasts.jsonl) hold every side of every market the model
priced, recommended or not, plus the snapshot id of the exact inputs used.
The ledger (logs/ledger.jsonl) holds only actions you took: "placed" and "void".
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from .timeutil import fmt, now_utc, parse_utc

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "logs" / "forecasts.jsonl"
LEDGER = ROOT / "logs" / "ledger.jsonl"
COLLECTIONS = ROOT / "logs" / "collections.jsonl"
WEATHER_ARCHIVE = ROOT / "logs" / "weather_forecasts.jsonl"
SNAPSHOTS = ROOT / "snapshots"
GENESIS = "0" * 64


class StoreError(RuntimeError):
    pass


def _clean(v):
    """JSON-safe scalar: NaN -> None, Timestamps -> ISO UTC, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return fmt(v) if v.tzinfo else v.isoformat()
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(path: Path) -> tuple[bool, str]:
    prev, n = GENESIS, 0
    for i, rec in enumerate(read(path)):
        if rec.get("seq") != i:
            return False, f"record {i}: sequence {rec.get('seq')} out of order"
        if rec.get("prev") != prev:
            return False, f"record {i}: previous-hash link broken"
        if _digest(rec) != rec.get("hash"):
            return False, f"record {i}: content does not match its hash (edited?)"
        prev, n = rec["hash"], i + 1
    return True, f"{n} records, chain intact"


def append(path: Path, kind: str, rows: list[dict], recorded_utc: str | None = None) -> list[dict]:
    ok, msg = verify(path)
    if not ok:
        raise StoreError(f"{path.name} failed verification ({msg}); refusing to append")
    existing = read(path)
    prev = existing[-1]["hash"] if existing else GENESIS
    seq = len(existing)
    when = recorded_utc or fmt(now_utc())
    out = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for data in rows:
            rec = {"seq": seq, "recorded_utc": when, "kind": kind,
                   "data": {k: _clean(v) for k, v in data.items()}, "prev": prev}
            rec["hash"] = _digest(rec)
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
            out.append(rec)
            prev, seq = rec["hash"], seq + 1
    return out


# ---------------------------------------------------------------- snapshots

def save_snapshot(run_id: str, files: dict[str, Path], frames: dict[str, pd.DataFrame],
                  meta: dict) -> tuple[str, Path]:
    """Copy the exact inputs of a run; returns (snapshot_id, directory).
    snapshot_id = sha256 of the manifest of file hashes."""
    d = SNAPSHOTS / run_id
    d.mkdir(parents=True, exist_ok=False)
    for name, src in files.items():
        if src is not None and Path(src).exists():
            shutil.copy2(src, d / name)
    for name, df in frames.items():
        if df is not None:
            df.to_csv(d / name, index=False)
    (d / "meta.json").write_text(json.dumps({k: _clean(v) if not isinstance(v, (dict, list)) else v
                                             for k, v in meta.items()}, indent=2, sort_keys=True, default=str))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    sid = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return sid, d


# ---------------------------------------------------------------- forecasts

FORECAST_FIELDS = ["game_id", "kickoff_utc", "season", "week", "away_team", "home_team", "market", "side",
                   "team", "book", "point", "price", "price_source", "odds_time", "reference",
                   "reference_live", "p_win", "p_push", "model_prob", "implied", "market_prob", "ev",
                   "ev[fair 0.5 worse]", "ev[market only]", "min_price", "decision", "stake_units",
                   "reasons", "is_best_side"]

TIER = {"BET": "executable", "BET IF CONFIRMED": "conditional",
        "BET IF PRICE AVAILABLE": "conditional", "NO BET": "pass"}


def record_forecasts(rows: pd.DataFrame, run: dict, path: Path = FORECASTS) -> list[dict]:
    recs = []
    for r in rows.to_dict("records"):
        d = {k: r.get(k) for k in FORECAST_FIELDS}
        d["tier"] = TIER.get(r.get("decision"), "pass")
        d.update(run)
        recs.append(d)
    return append(path, "forecast", recs, recorded_utc=run.get("run_utc"))


def forecasts_frame(path: Path = FORECASTS) -> pd.DataFrame:
    recs = read(path)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame([{**r["data"], "recorded_utc": r["recorded_utc"], "hash": r["hash"]} for r in recs])
    return df


# ---------------------------------------------------------------- ledger

def place_bet(forecast_hash: str, stake_units: float, placed_utc: str | None = None, price: float | None = None,
              point: float | None = None, book: str | None = None, note: str = "",
              forecasts_path: Path = FORECASTS, ledger_path: Path = LEDGER) -> dict:
    """Record a wager you actually placed, linked to the forecast it came from."""
    from .odds import VALID_SIDES, _valid_point, _valid_price

    ok, msg = verify(forecasts_path)
    if not ok:
        raise StoreError(f"forecast history failed verification: {msg}")
    matches = [r for r in read(forecasts_path) if r["hash"].startswith(forecast_hash)]
    if len(matches) != 1:
        raise StoreError(f"forecast id {forecast_hash!r} matches {len(matches)} records; give more characters")
    f = matches[0]["data"]
    when = parse_utc(placed_utc) if placed_utc else now_utc()
    kickoff = parse_utc(f["kickoff_utc"])
    if when >= kickoff:
        raise StoreError(f"placed time {fmt(when)} is not before kickoff {fmt(kickoff)}")
    if parse_utc(matches[0]["recorded_utc"]) > when:
        raise StoreError("placed time is before the forecast was recorded")
    if not (stake_units > 0 and math.isfinite(stake_units)):
        raise StoreError("stake must be a positive number of units")
    price = f["price"] if price is None else price
    point = f["point"] if point is None else point
    if not _valid_price(price):
        raise StoreError(f"invalid price {price!r}")
    if f["market"] != "ml" and not _valid_point(f["market"], point):
        raise StoreError(f"invalid point {point!r}")
    if f["side"] not in VALID_SIDES[f["market"]]:
        raise StoreError("forecast has an invalid market/side")
    bet_id = hashlib.sha256(f"{matches[0]['hash']}{fmt(when)}{stake_units}{price}{point}".encode()).hexdigest()[:12]
    if any(r["data"].get("bet_id") == bet_id for r in read(ledger_path)):
        raise StoreError(f"bet {bet_id} already recorded")
    data = {"bet_id": bet_id, "forecast_hash": matches[0]["hash"], "placed_utc": fmt(when),
            "game_id": f["game_id"], "kickoff_utc": f["kickoff_utc"], "market": f["market"], "side": f["side"],
            "team": f["team"], "book": book or f["book"], "point": point, "price": price,
            "stake_units": stake_units, "forecast_decision": f["decision"], "forecast_price": f["price"],
            "forecast_point": f["point"], "note": note}
    return append(ledger_path, "placed", [data], recorded_utc=fmt(now_utc()))[0]


def void_bet(bet_id: str, reason: str, ledger_path: Path = LEDGER) -> dict:
    placed = {r["data"]["bet_id"] for r in read(ledger_path) if r["kind"] == "placed"}
    voided = {r["data"]["bet_id"] for r in read(ledger_path) if r["kind"] == "void"}
    if bet_id not in placed:
        raise StoreError(f"no placed bet {bet_id}")
    if bet_id in voided:
        raise StoreError(f"bet {bet_id} already voided")
    return append(ledger_path, "void", [{"bet_id": bet_id, "reason": reason}])[0]


def ledger_frame(ledger_path: Path = LEDGER) -> pd.DataFrame:
    recs = read(ledger_path)
    placed = [r["data"] for r in recs if r["kind"] == "placed"]
    if not placed:
        return pd.DataFrame()
    voided = {r["data"]["bet_id"]: r["data"]["reason"] for r in recs if r["kind"] == "void"}
    df = pd.DataFrame(placed)
    df["voided"] = df["bet_id"].map(voided)
    return df


# ---------------------------------------------------------------- weather archive

WEATHER_FIELDS = ["game_id", "kickoff_utc", "valid_for_utc", "wind_mph", "temp_f", "source", "forecast_utc",
                  "retrieved_utc"]


def archive_weather(accepted: dict, run_id: str, path: Path = WEATHER_ARCHIVE) -> int:
    """Append validated forecasts (issue, retrieval and valid-for times, source, game time) so a
    future weather feature can be evaluated on forecasts that existed before kickoff.
    Identical forecasts already archived are not duplicated. Returns rows added."""
    seen = {tuple(r["data"].get(k) for k in WEATHER_FIELDS) for r in read(path)}
    rows = []
    for w in accepted.values():
        d = {k: _clean(w.get(k)) for k in WEATHER_FIELDS}
        key = tuple(d.get(k) for k in WEATHER_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        rows.append({**d, "run_id": run_id})
    if rows:
        append(path, "weather_forecast", rows)
    return len(rows)
