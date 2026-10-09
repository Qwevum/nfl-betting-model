"""Python requirement is stated once and enforced consistently."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nflmodel  # noqa: E402


class PythonRequirement(unittest.TestCase):
    def test_running_interpreter_meets_requirement(self):
        self.assertGreaterEqual(sys.version_info[:2], nflmodel.MIN_PYTHON)
        self.assertEqual(nflmodel.MIN_PYTHON, (3, 11))

    def test_docs_and_requirements_agree(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("Python 3.11+", readme)
        self.assertIsNone(re.search(r"Python 3\.10", readme))
        self.assertIn("Python 3.11+", (ROOT / "requirements.txt").read_text())

    def test_tomllib_is_the_only_toml_reader(self):
        self.assertIn("import tomllib", (ROOT / "nflmodel" / "config.py").read_text())


if __name__ == "__main__":
    unittest.main()
