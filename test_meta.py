"""Project-metadata pinning tests (stdlib only — no yaml dependency).

Keeps CITATION.cff in lockstep with the code: the cited version must always equal
benchy_common.__version__, so a release bump in one place without the other fails CI.
CITATION.cff is parsed line-wise on purpose (top-level `key: value` lines only) to keep
the zero-pip-deps guarantee.
"""
import os
import unittest

import benchy_common

HERE = os.path.dirname(os.path.abspath(__file__))
CFF_PATH = os.path.join(HERE, "CITATION.cff")


def cff_top_level_field(name):
    """Return the value of a top-level `name: value` line in CITATION.cff, or None.

    Only unindented lines count: nested mappings (e.g. under `references:`) are indented
    in this file, so a hypothetical nested `version:` can never shadow the project one.
    Surrounding single/double quotes are stripped.
    """
    prefix = name + ":"
    with open(CFF_PATH, encoding="utf-8") as f:
        for line in f:
            if line[:1].isspace() or line.startswith("#"):
                continue
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                return value
    return None


class TestCitationMetadata(unittest.TestCase):
    def test_citation_file_exists(self):
        self.assertTrue(os.path.isfile(CFF_PATH), "CITATION.cff missing")

    def test_version_matches_benchy_common(self):
        cff_version = cff_top_level_field("version")
        self.assertIsNotNone(cff_version, "CITATION.cff has no top-level version field")
        self.assertEqual(
            cff_version, benchy_common.__version__,
            "CITATION.cff version (%s) != benchy_common.__version__ (%s) — "
            "bump both together" % (cff_version, benchy_common.__version__))


if __name__ == "__main__":
    unittest.main()
