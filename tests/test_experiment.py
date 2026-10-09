"""Experiment protocol: acceptance rule and the one-time holdout guard."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import experiment, store  # noqa: E402


def res(winner=(-0.002, -0.004, -0.0005, 6), cover=(0.0, -0.001, 0.001, 3), over=(0.0, -0.001, 0.001, 4)):
    def t(x):
        return {"d_logloss": x[0], "ci_lo": x[1], "ci_hi": x[2], "seasons_improved": x[3], "seasons": 7}
    return {"winner": t(winner), "home covers": t(cover), "over hits": t(over)}


class Acceptance(unittest.TestCase):
    def test_passes_only_with_ci_below_zero_and_consistency(self):
        self.assertTrue(experiment.passes_dev(res())[0])
        self.assertFalse(experiment.passes_dev(res(winner=(-0.002, -0.004, 0.0002, 6)))[0])   # CI crosses 0
        self.assertFalse(experiment.passes_dev(res(winner=(-0.002, -0.004, -0.0005, 4)))[0])  # 4/7 seasons
        self.assertFalse(experiment.passes_dev(res(cover=(0.002, 0.0005, 0.004, 1)))[0])      # worse elsewhere


class HoldoutGuard(unittest.TestCase):
    def test_holdout_requires_dev_pass_and_runs_once(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "e.jsonl"
            with mock.patch.object(experiment, "LOG", log):
                with self.assertRaises(experiment.ProtocolError):
                    experiment.run(None, "G1", holdout=True, last_season=2026, model_version="x")
                store.append(log, "experiment", [{"group": "G1", "period": "development", "passed": True,
                                                  "passed_targets": ["winner"]}])
                store.append(log, "experiment", [{"group": "G1", "period": "holdout", "passed": False,
                                                  "passed_targets": []}])
                with self.assertRaises(experiment.ProtocolError):
                    experiment.run(None, "G1", holdout=True, last_season=2026, model_version="x")
                with self.assertRaises(experiment.ProtocolError):
                    experiment.run(None, "G9", holdout=False, last_season=2026, model_version="x")


if __name__ == "__main__":
    unittest.main()
