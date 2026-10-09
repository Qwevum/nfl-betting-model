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
    out.append("- " + headline_win_prob(g, rows))
    d = drivers(g, coefs)
    if d:
        out.append("- Biggest inputs to the model's own line: " + "; ".join(d))
    out += ["", "Probabilities (win, excluding pushes): **raw market** = no-vig reference at this line, built "
            "without the priced book; **calibrated market-only** = the calibration applied to it with no model "
            "input; **final** = the probability used for EV and the decision.", "",
            "| Market | Bet | Best price | Implied (break-even) | Raw market | Calibrated market-only | Final | EV | "
            "EV if fair 0.5 worse | EV at raw market | Status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rows.sort_values(["market", "ev"], ascending=[True, False]).iterrows():
        when = f", {r['odds_time']}" if isinstance(r["odds_time"], str) else ", time unknown"
        out.append(f"| {MARKET[r['market']]} | {_bet(r)} | {int(r['price']):+d} ({r['book']}{when}) | "
                   f"{_pct(r['implied'])} | {_pct(r['market_prob'])} | {_pct(r.get('market_cal_prob'))} | "
                   f"{_pct(r['model_prob'])} | {r['ev']:+.1%} | {r['ev[fair 0.5 worse]']:+.1%} | "
                   f"{_signed(r.get('ev_raw_market'))} | {status_label(r)} |")
    out += ["", "**Decisions**", ""]
    for _, r in rows[rows["is_best_side"]].sort_values("market").iterrows():
        line = f"- {MARKET[r['market']]}: **{status_label(r)}**"
        if r["decision"] == "BET":
            line += (f" {_bet(r)} {int(r['price']):+d}, stake {r['stake_units']:g}u "
                     f"({settings.kelly_fraction:g}x Kelly, cap {settings.max_stake_units:g}u); still "
                     f"+{settings.min_edge:.1%} EV at {_fmt_price(r['min_price'])} or better")
        line += f". {r['reasons'] or 'All checks passed.'}"
        out.append(line)
        if "reference" in r and isinstance(r["reference"], str):
            out.append(f"  - Market reference: {r['reference']}")
        if r["decision"] == "BET" and r["risks"]:
            out.append(f"  - Could be wrong because: {r['risks']}")
        sens = {k[3:-1]: v for k, v in r.items() if k.startswith("ev[")}
        out.append("  - Sensitivity (EV): " + ", ".join(f"{k} {v:+.1%}" for k, v in sens.items()))
    out += ["", "**Assumptions** (not verified)", ""] + [f"- {a}" for a in ctx.assumptions]
    out += ["", "**Missing or stale information**", ""]
    out += [f"- {m}" for m in ctx.missing] or ["- none detected"]
    if any(r for r in rows["price_source"] if r == "consensus"):
        out.append("- Prices are nflverse consensus lines with no timestamp; not a quote from a specific book")
    return "\n".join(out) + "\n"


def _signed(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.1%}"


def status_label(r) -> str:
    """Only a BET is a recommendation; everything else is labelled by what it lacks."""
    if r["decision"] == "BET":
        return "BET (actionable)"
    cat = r.get("category")
    return f"not a bet: {cat}" if isinstance(cat, str) and cat else "not a bet"


def headline_win_prob(g, rows: pd.DataFrame) -> str:
    """The SAME probability the moneyline rows use (final), with the raw market next to it."""
    ml = rows[(rows["market"] == "ml") & (rows["side"] == "home")] if len(rows) else rows
    if len(ml):
        r = ml.iloc[0]
        return (f"{g.home_team} win probability {r['model_prob']:.1%} (final, as used in the moneyline rows; "
                f"raw market {_pct(r['market_prob'])})")
    return f"{g.home_team} win probability: n/a (no moneyline priced)"


def actionable_md(rows: pd.DataFrame, settings) -> str:
    best = rows[rows["decision"] == "BET"] if len(rows) else rows
    out = ["## Actionable recommendations", "",
           "Only sides that meet **every** requirement: EV above the threshold and robust to a 0.5-pt error, "
           "a timestamped quote at a book you can use, a live market reference without that book, nothing left "
           "to confirm, and still valid when the run completed.", ""]
    if best.empty:
        out.append("**None.** No side met every requirement in this run.")
        return "\n".join(out) + "\n"
    out += ["| Game | Bet | Price (book, quote time) | Final prob | Raw market | EV | Stake |", "|---|---|---|---|---|---|---|"]
    for _, r in best.sort_values("ev", ascending=False).iterrows():
        out.append(f"| {r['game_id']} | {_bet(r)} | {int(r['price']):+d} ({r['book']}, {r['odds_time']}) | "
                   f"{_pct(r['model_prob'])} | {_pct(r['market_prob'])} | {r['ev']:+.1%} | {r['stake_units']:g}u |")
    return "\n".join(out) + "\n"


def watchlist_md(w: pd.DataFrame, settings, preds: pd.DataFrame) -> str:
    out = ["## Watchlist (conditional, NOT recommendations)", "",
           "The closest candidates that failed at least one requirement. \"Needs\" is the worst price at the "
           "**same line** that would clear the EV threshold **if the probability estimate stays the same**. A "
           "later quote at that price is not a bet by itself: rerun the analysis with fresh quotes first.", ""]
    if w.empty:
        out.append("Nothing to show.")
        return "\n".join(out) + "\n"
    names = preds.set_index("game_id")
    out += ["| Game | Side | Current price (book) | Final prob | Raw market | EV now | Needs (same line) | "
            "Category | Why not actionable |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in w.iterrows():
        x = names.loc[r["game_id"]]
        out.append(f"| {x.away_team} @ {x.home_team} | {_bet(r)} | {int(r['price']):+d} ({r['book']}) | "
                   f"{_pct(r['model_prob'])} | {_pct(r['market_prob'])} | {r['ev']:+.1%} | {r['needs_price']} | "
                   f"{r['category']} | {r['why']} |")
    return "\n".join(out) + "\n"


def diagnostics_md(rows: pd.DataFrame, settings) -> str:
    from . import diagnose
    from .validate import to_markdown
    best = rows[rows["is_best_side"].astype(bool)]
    out = ["## Why these decisions", "",
           f"Primary cause: {diagnose.primary_cause(rows, settings)}.", "",
           "Categories (all sides / best side per market):", ""]
    ca, cb = diagnose.categorize(rows, settings).value_counts(), diagnose.categorize(best, settings).value_counts()
    out += ["| category | all sides | best sides |", "|---|---|---|"]
    out += [f"| {c} | {int(ca.get(c, 0))} | {int(cb.get(c, 0))} |" for c in diagnose.CATEGORIES]
    out += ["", "Overlapping reasons (a side can have several):", ""]
    ov = diagnose.overlap(rows, settings).merge(diagnose.overlap(best, settings), on="reason",
                                                suffixes=(" (all)", " (best)"))
    out += [to_markdown(ov), "", "Sequential filter (best side per market; each stage keeps sides passing all "
            "earlier stages):", "", to_markdown(diagnose.funnel(best, settings)), "",
            "Estimated EV at the best available price, by market (all sides):", "",
            to_markdown(diagnose.ev_distribution(rows, settings),
                        {k: "{:+.1%}" for k in ("min", "p25", "median", "p75", "max")})]
    return "\n".join(out) + "\n"


def terminal_actionable(rows: pd.DataFrame, preds: pd.DataFrame) -> str:
    best = rows[rows["decision"] == "BET"] if len(rows) else rows
    if best.empty:
        return "ACTIONABLE RECOMMENDATIONS: none (no side met every requirement)."
    g = preds.set_index("game_id")
    t = [[f"{g.loc[r['game_id']].away_team} @ {g.loc[r['game_id']].home_team}", MARKET[r["market"]], _bet(r),
          f"{int(r['price']):+d}", r["book"], _pct(r["model_prob"]), _pct(r["market_prob"]), f"{r['ev']:+.1%}",
          f"{r['stake_units']:g}u"] for _, r in best.sort_values("ev", ascending=False).iterrows()]
    return "ACTIONABLE RECOMMENDATIONS\n" + table(t, ["Game", "Market", "Bet", "Odds", "Book", "Final", "Raw mkt",
                                                       "EV", "Stake"], "llllllrrr")


def terminal_watchlist(w: pd.DataFrame, preds: pd.DataFrame) -> str:
    if w.empty:
        return "WATCHLIST: nothing to show."
    g = preds.set_index("game_id")
    t = [[f"{g.loc[r['game_id']].away_team} @ {g.loc[r['game_id']].home_team}", _bet(r), f"{int(r['price']):+d}",
          _pct(r["model_prob"]), _pct(r["market_prob"]), f"{r['ev']:+.1%}", r["needs_price"], r["category"],
          r["why"][:70]] for _, r in w.iterrows()]
    return ("WATCHLIST - conditional, NOT recommendations. 'Needs' = worst price at the same line that clears the\n"
            "threshold IF the probability estimate is unchanged; rerun with fresh quotes before acting.\n"
            + table(t, ["Game", "Side", "Now", "Final", "Raw mkt", "EV", "Needs", "Category", "Why not"],
                    "llrrrrrll"))


def header(season, week, sources: dict, version: str, model, settings, validation: str | None,
           timeline: dict | None = None, simulated: bool = False) -> str:
    out = [f"# NFL {season} Week {week} - model report", ""]
    if simulated:
        out += ["> **SIMULATED RUN** (`--now`): the clock is fixed, nothing is recorded, and this is not a "
                "prospective forecast.", ""]
    if timeline:
        out += ["Run timeline (UTC): " + ", ".join(f"{k.replace('_utc', '')} {v}" for k, v in timeline.items()), ""]
    out += [
           f"Model version `{version}`. Flag threshold: EV > {settings.min_edge:.1%} at the available price. "
           f"Stakes: {settings.kelly_fraction:g} x full Kelly, capped at {settings.max_stake_units:g} units. "
           f"Model-vs-market gap rule: {settings.gap_points:g} pts.", "",
           f"All settings used: `{settings.describe()}`", "",
           f"Probability model: `{settings.probability_model}`. Actionable books: "
           + (f"`{settings.actionable_books}` (other books are used only as market references)"
              if settings.actionable_books.strip() else
              "not configured, so every book counts as actionable (set `actionable_books` in settings.toml)"), "",
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
