"""Test runner that discovers both test roots.

Why this exists
---------------
``python manage.py test`` with no arguments discovered only 143 tests. The
other 142 — everything under ``src/tests/`` — were silently skipped.

The cause is that ``src/`` has no ``__init__.py``. It therefore imports as a
*namespace* package, and ``unittest``'s discovery deliberately does not descend
into namespace packages: it walks directories, and a directory without
``__init__.py`` is not a package as far as it is concerned. The modules are
importable (``import src.cjps_train.run`` works fine), which is exactly what
makes the failure quiet — the code under test imports fine, the tests for it
just are not collected.

The danger is not the missed tests, it is that nobody notices. A contributor
following the README runs one command, sees green, and ships.

So this runner discovers both roots explicitly, and asserts that each one
actually contributed tests. An empty root is an error, not a pass — that way a
future rename or move cannot quietly empty a suite.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from django.test.runner import DiscoverRunner

#: Directories holding tests, relative to the repository root. Each must yield
#: at least one test; see :meth:`RunTests.assert_no_silent_skips`.
TEST_ROOTS: tuple[str, ...] = ("ctmj", "src")


def _count(suite) -> int:
    """Number of tests in a (possibly nested) suite."""
    total = 0
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            total += _count(item)
        else:
            total += 1
    return total


def _iter_tests(suite):
    """Yield every leaf test in a possibly nested suite."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


class RunTests(DiscoverRunner):
    """``DiscoverRunner`` that also collects ``src/tests``."""

    def build_suite(self, test_labels=None, extra_tests=None, **kwargs):
        if test_labels:
            # An explicit label is an explicit request; do not second-guess it.
            return super().build_suite(
                test_labels, extra_tests=extra_tests, **kwargs
            )

        combined = unittest.TestSuite()
        counts: dict[str, int] = {}
        for label in TEST_ROOTS:
            suite = super().build_suite([label], extra_tests=extra_tests, **kwargs)
            counts[label] = _count(suite)
            combined.addTest(suite)

        self.assert_no_silent_skips(counts)
        self.collected_per_root = counts
        return combined

    def assert_no_silent_skips(self, counts: dict[str, int]) -> None:
        """Every declared root must have contributed at least one test.

        Counting happens per label, on the suite built for that label alone.
        Grouping the combined suite by ``type(test).__module__`` does not work:
        Django sets ``top_level_dir`` to the discovered directory, so tests
        under ``src/tests`` are named ``tests.test_*`` rather than
        ``src.tests.test_*`` and cannot be told apart from the app's modules by
        their import path.

        Raises ``AssertionError``, which Django reports as an error, so a root
        that collects nothing cannot present as a passing run.
        """
        empty = sorted(label for label, count in counts.items() if count == 0)
        if empty:
            raise AssertionError(
                f"No tests were collected from {', '.join(empty)} "
                f"(collected: {counts}). A test root that collects nothing "
                "would otherwise pass silently."
            )
