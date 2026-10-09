"""Write the weekly report (markdown) and the terminal summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decide import GameContext, _fmt_price

FEATURE_NAMES = {
    "f_mrtg": "points-margin rating", "f_epa": "EPA/play rating", "f_sr": "success-rate rating",
    "f_pass": "pass EPA rating", "f_qb": "starting-QB change", "f_hfa": "home field",
    "f_rest": "rest difference", "f_div": "division game",
}
MARKET = {"spread": "Spread", "ml": "Moneyline", "total": "Total"}


def _line(home, away, m):
    if pd.isna(m):
        return "n/a"
    if abs(m) < 0.25:
        return "PK"
    return f"{home if m > 0 else away} -{abs(m):.1f}"


def _bet(r) -> str:
    if r["market"] == "spread":
        return f"{r['team']} {r['point']:+g}"
    if r["market"] == "ml":
        return f"{r['team']} ML"
    return f"{r['team']} {r['point']:g}"


def _pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def drivers(g, coefs: dict) -> list[str]:
    contrib = {k: coefs[k] * getattr(g, k) for k in FEATURE_NAMES if k in coefs}
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:4]
    return [f"{FEATURE_NAMES[k]}: {v:+.1f} pts to {g.home_team}" for k, v in top if abs(v) >= 0.1]


def game_section(g, ctx: GameContext, rows: pd.DataFrame, coefs: dict, model, settings) -> str:
    out = [f"## {g.away_team} @ {g.home_team}", ""]
    out += ["**Verified facts** (read from dated sources)", ""] + [f"- {f}" for f in ctx.facts]
    out += ["", "**Model estimate** (reproducible calculation, see README)", ""]
    out.append(f"- Market spread {_line(g.home_team, g.away_team, g.spread_line)}, total "
               f"{'n/a' if pd.isna(g.total_line) else f'{g.total_line:g}'} (consensus)")
    out.append(f"- Model's own line {_line(g.home_team, g.away_team, g.model_margin)}; fair line after "
               f"calibration {_line(g.home_team, g.away_team, g.fair_margin)}; fair total {g.fair_total:.1f}")
    out.append(f"- {g.home_team} win probability {model.win_prob(g.fair_margin):.1%}")
    d = drivers(g, coefs)
    if d:
        out.append("- Biggest inputs to the model's own line: " + "; ".join(d))
    out += ["", "| Market | Bet | Best price | Implied | Market (no-vig) | Model | EV | EV if fair 0.5 worse | "
            "EV market only | Decision |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rows.sort_values(["market", "ev"], ascending=[True, False]).iterrows():
        when = f", {r['odds_time']}" if isinstance(r["odds_time"], str) else ", time unknown"
        out.append(f"| {MARKET[r['market']]} | {_bet(r)} | {int(r['price']):+d} ({r['book']}{when}) | "
                   f"{_pct(r['implied'])} | {_pct(r['market_prob'])} | {_pct(r['model_prob'])} | "
                   f"{r['ev']:+.1%} | {r['ev[fair 0.5 worse]']:+.1%} | {r['ev[market only]']:+.1%} | "
                   f"{r['decision']} |")
    out += ["", "**Decisions**", ""]
    for _, r in rows[rows["is_best_side"]].sort_values("market").iterrows():
        line = f"- {MARKET[r['market']]}: **{r['decision']}**"
        if r["decision"] != "NO BET":
            line += (f" {_bet(r)} {int(r['price']):+d}, stake {r['stake_units']:g}u "
                     f"({settings.kelly_fraction:g}x Kelly, cap {settings.max_stake_units:g}u); still "
                     f"+{settings.min_edge:.1%} EV at {_fmt_price(r['min_price'])} or better")
        line += f". {r['reasons'] or 'All checks passed.'}"
        out.append(line)
        if "reference" in r and isinstance(r["reference"], str):
            out.append(f"  - Market reference: {r['reference']}")
        if r["decision"] != "NO BET" and r["risks"]:
            out.append(f"  - Could be wrong because: {r['risks']}")
        sens = {k[3:-1]: v for k, v in r.items() if k.startswith("ev[")}
        out.append("  - Sensitivity (EV): " + ", ".join(f"{k} {v:+.1%}" for k, v in sens.items()))
    out += ["", "**Assumptions** (not verified)", ""] + [f"- {a}" for a in ctx.assumptions]
    out += ["", "**Missing or stale information**", ""]
    out += [f"- {m}" for m in ctx.missing] or ["- none detected"]
    if any(r for r in rows["price_source"] if r == "consensus"):
        out.append("- Prices are nflverse consensus lines with no timestamp; not a quote from a specific book")
    return "\n".join(out) + "\n"


def header(season, week, sources: dict, version: str, model, settings, validation: str | None) -> str:
    out = [f"# NFL {season} Week {week} - model report", "",
           f"Model version `{version}`. Flag threshold: EV > {settings.min_edge:.1%} at the available price. "
           f"Stakes: {settings.kelly_fraction:g} x full Kelly, capped at {settings.max_stake_units:g} units. "
           f"Model-vs-market gap rule: {settings.gap_points:g} pts.", "",
           f"All settings used: `{settings.describe()}`", "",
           "## Data sources", "", "| Source | Retrieved (UTC) | Server last-modified | Notes |", "|---|---|---|---|"]
    for name, s in sources.items():
        notes = ", ".join(f"{k}: {v}" for k, v in s.items()
                          if k not in ("url", "retrieved_utc", "last_modified"))
        out.append(f"| {name} | {s.get('retrieved_utc', '')} | {s.get('last_modified') or 'not provided'} | "
                   f"{notes} |")
    out += ["", "## How much to trust this", "",
            f"- Out of sample, {model.k_spread:.0%} of the model's disagreement with the spread and "
            f"{model.k_total:.0%} with the total held up; the fair lines keep only that share.",
            "- Against closing lines the walk-forward backtest has not shown a statistically "
            "significant edge. Treat every BET as a small, logged test."]
    if validation:
        out += ["", validation]
    return "\n".join(out) + "\n"


def terminal_summary(best: pd.DataFrame, preds: pd.DataFrame) -> str:
    g = preds.set_index("game_id")
    rows = []
    for _, r in best.sort_values(["decision", "ev"], ascending=[True, False]).iterrows():
        x = g.loc[r["game_id"]]
        rows.append([f"{x.away_team} @ {x.home_team}", MARKET[r["market"]], _bet(r),
                     f"{int(r['price']):+d}", _pct(r["implied"]), _pct(r["market_prob"]), _pct(r["model_prob"]),
                     f"{r['ev']:+.1%}", f"{r['ev[fair 0.5 worse]']:+.1%}",
                     f"{r['stake_units']:g}u" if r["decision"] != "NO BET" else "-",
                     r["decision"], (r["reasons"] or "")[:60]])
    return table(rows, ["Game", "Market", "Best side", "Odds", "Implied", "Mkt no-vig", "Model", "EV",
                        "EV -0.5pt", "Stake", "Decision", "Reason"], "llllrrrrrrll")


def table(rows: list[list[str]], header: list[str], align: str) -> str:
    """Box-drawn terminal table. align: one char per column, 'l' or 'r'."""
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(header)]

    def line(cells):
        return "│ " + " │ ".join(c.rjust(w) if a == "r" else c.ljust(w)
                                 for c, w, a in zip(cells, widths, align)) + " │"

    def rule(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r

    if not rows:
        return "(nothing to show)"
    return "\n".join([rule("┌", "┬", "┐"), line(header), rule("├", "┼", "┤")]
                     + [line(r) for r in rows] + [rule("└", "┴", "┘")])
