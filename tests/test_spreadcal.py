"""Spread calibration diagnostics (nflmodel/spreadcal.py) on SYNTHETIC data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import spreadcal as sc  # noqa: E402
from nflmodel.model import fit, inner_oos  # noqa: E402
from test_forecast_uncertainty import FEAT, TRAIN, POINT  # noqa: E402

ROWS = sc.spread_rows(inner_oos(TRAIN))


class Refit(unittest.TestCase):
    def test_reproduces_the_models_own_calibration(self):
        np.testing.assert_allclose(sc.fit_legacy(ROWS), POINT.spread_cal, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(sc.fit_offset(ROWS), POINT.offset_cal["spread"][2], places=12)

    def test_rows_exclude_pushes_and_missing(self):
        oos = inner_oos(TRAIN)
        pushes = int((oos["result"] == oos["spread_line"]).sum())
        self.assertEqual(len(ROWS), len(oos.dropna(subset=["blend_margin", "result", "spread_line"])) - pushes)
        self.assertTrue(set(ROWS["y"].unique()) <= {0.0, 1.0})
        self.assertTrue(ROWS["season"].min() > TRAIN["season"].min())   # inner walk-forward skips first seasons

    def test_inner_oos_is_out_of_sample(self):
        """Changing a season's outcomes cannot change that season's inner predictions."""
        t2 = TRAIN.copy()
        last = t2["season"] == t2["season"].max()
        t2.loc[last, "result"] += 20
        a, b = inner_oos(TRAIN), inner_oos(t2)
        np.testing.assert_array_equal(a.loc[a["season"] == TRAIN["season"].max(), "blend_margin"],
                                      b.loc[b["season"] == TRAIN["season"].max(), "blend_margin"])


class Bootstrap(unittest.TestCase):
    B = sc.bootstrap(ROWS, n=200, seed=3)

    def test_intervals_and_nulls(self):
        b = self.B
        self.assertEqual(list(b["param"]), ["a", "b", "c", "c"])
        self.assertTrue((b["lo"] <= b["hi"]).all())
        self.assertTrue(((b["lo"] <= b["estimate"] + 1e-9) & (b["estimate"] <= b["hi"] + 1e-9)).all())
        bb = b[b["param"] == "b"].iloc[0]
        self.assertEqual(bool(bb["CI includes null"]), bool(bb["lo"] <= 1 <= bb["hi"]))

    def test_reproducible(self):
        pd.testing.assert_frame_equal(self.B, sc.bootstrap(ROWS, n=200, seed=3))

    def test_by_era_splits(self):
        e = sc.by_era(ROWS, {"early": (2014, 2015), "late": (2016, 2017), "none": (1990, 1991)}, n=50)
        self.assertEqual(set(e["era"]), {"early", "late"})


class Decompose(unittest.TestCase):
    D = sc.decompose(POINT, POINT.predict(FEAT[FEAT["season"] == 2018].head(10)))

    def test_parts_add_up_and_sides_mirror(self):
        d = self.D
        np.testing.assert_allclose(d["legacy: remap (a,b)"] + d["legacy: model (c*edge)"], d["legacy: total"],
                                   atol=1e-12)
        home, away = d.iloc[0::2].reset_index(drop=True), d.iloc[1::2].reset_index(drop=True)
        for c in ("legacy: total", "offset (live): model (c*edge)", "edge (pts)"):
            np.testing.assert_allclose(home[c], -away[c], atol=1e-12)
        np.testing.assert_allclose(home["no-vig"] + away["no-vig"], 1.0)

    def test_zero_edge_means_no_model_part(self):
        m = fit(TRAIN)
        m.offset_cal = {k: np.array([0.0, 1.0, 0.0]) for k in m.offset_cal}
        m.spread_cal = np.array([0.0, 1.0, 0.0])
        d = sc.decompose(m, m.predict(FEAT[FEAT["season"] == 2018].head(5)))
        np.testing.assert_allclose(d["legacy: total"], 0.0, atol=1e-12)
        np.testing.assert_allclose(d["offset (live): model (c*edge)"], 0.0, atol=1e-12)


class CompareCover(unittest.TestCase):
    def frame(self, p):
        rng = np.random.default_rng(0)
        n = 400
        res = rng.integers(-20, 21, n).astype(float)
        return pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "season": 2018, "result": res,
                             "spread_line": 0.5, "p_cover": p if np.ndim(p) else np.full(n, p)}), res

    def test_identical_arms(self):
        a, _ = self.frame(0.5)
        c = sc.compare_cover(a, a, n_boot=200)
        self.assertTrue((c["offset - legacy"].abs() < 1e-12).all())
        self.assertTrue(c["CI includes 0"].all())

    def test_better_arm_is_negative(self):
        a, res = self.frame(0.5)
        good = a.assign(p_cover=np.where(res > 0.5, 0.7, 0.3))
        c = sc.compare_cover(a, good, n_boot=200)
        self.assertTrue((c["offset - legacy"] < 0).all())
        self.assertFalse(c["CI includes 0"].any())


if __name__ == "__main__":
    unittest.main()
