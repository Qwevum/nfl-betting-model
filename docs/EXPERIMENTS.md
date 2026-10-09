# Feature experiments: protocol (written before any experiment was run)

## Periods
* **Development:** walk-forward test seasons 2015–2021. All comparisons and any
  decision to continue use only these seasons.
* **Holdout:** test seasons 2022–2026 (to date). It is evaluated **once** per
  feature group, and only for groups that pass the development criteria.
  `run.py experiment --holdout` refuses a second holdout run for the same group;
  every run is recorded in the hash-chained `logs/experiments.jsonl`.
* Caveat: earlier work in this repo (before this protocol) looked at 2015–2025
  results, so the holdout is not pristine for the baseline model itself. It is
  fresh for the new feature groups.

## Method
* One feature group at a time is added to the unchanged baseline. Both arms use:
  * identical training rows
  * the same walk-forward fitting and inner calibration
  * the same ridge penalty
* No hyperparameters are tuned for the new features: they use the same
  smoothing (EWMA step, season carry-over) as the existing ratings.
* Seasons without coverage for a group (for example snap counts before 2013)
  get the neutral value 0 in **both** arms' feature tables. The baseline
  ignores the column.

## Metrics
All are computed on identical rows for both arms:
* log loss and Brier of the final (market-anchored) probability for winner,
  home covers and over hits
* the model's own-line MAE, a signal check that does not affect decisions
* a paired, game-level bootstrap 95% CI of the log-loss difference
  (experiment − baseline)
* the number of development seasons where the experiment's log loss is lower

## Acceptance criteria (development)
A group passes only if **all** of these hold:
1. for at least one target, the pooled log-loss difference is < 0 and its 95%
   CI excludes 0
2. for no target is the difference > 0 with a CI excluding 0
3. the improving target improves in at least 5 of the 7 development seasons

A group that passes is evaluated once on the holdout. It becomes a default
feature only if the holdout log-loss difference for the same target is ≤ 0.
Otherwise it stays experimental: available via `--features`, off by default.

## Feature groups

| id | group | source | timing |
|---|---|---|---|
| G1 | separate rushing efficiency (opp.-adjusted rush EPA/play, offense & defense; passing already in the model) | play-by-play | prior games only |
| G2 | neutral-situation pace (seconds per play, Q1–Q3, score within 7) | play-by-play | prior games only |
| G3 | snap-weighted availability: expected snap share (prior 4 games) of players listed Out/Doubtful, non-QB | injury reports + snap counts | the final injury report precedes kickoff by NFL rule, but the feed has **no publish timestamps**; labelled as inferred timing |
| G4 | offensive-line continuity: share of last game's OL snaps taken by the team's top-5 OL of the prior 4 games | snap counts | prior games only |
