"""Tests for the test runner itself.

A test runner that silently collects nothing is worse than no runner: the
command a contributor trusts reports green while testing nothing. These tests
pin the behaviour that prevents that.
"""

from __future__ import annotations

import unittest

from django.test import SimpleTestCase

from ctmj import testrunner


class CountingTests(SimpleTestCase):
    def test_count_handles_nested_suites(self):
        """A suite inside a suite must be walked, not counted as one item."""
        inner = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        outer = unittest.TestSuite([
            inner,
            unittest.FunctionTestCase(lambda: None),
        ])
        self.assertEqual(testrunner._count(outer), 2)

    def test_count_of_empty_suite_is_zero(self):
        self.assertEqual(testrunner._count(unittest.TestSuite()), 0)

    def test_iter_tests_flattens(self):
        inner = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        outer = unittest.TestSuite([
            inner,
            unittest.FunctionTestCase(lambda: None),
        ])
        self.assertEqual(len(list(testrunner._iter_tests(outer))), 2)

    def test_grouping_is_by_module_path(self):
        """Each test is attributed to exactly one root.

        Kept as a direct unit of ``_count``-style bookkeeping: the guard itself
        counts per label rather than by module path, because Django names the
        modules under ``src/tests`` as ``tests.*`` and not ``src.tests.*``.
        """
        class InCtmj(unittest.TestCase):
            def test_a(self):
                pass

        class InSrc(unittest.TestCase):
            def test_b(self):
                pass

        InCtmj.__module__ = "ctmj.tests.test_thing"
        InSrc.__module__ = "tests.test_thing"

        suite = unittest.TestSuite([InCtmj("test_a"), InSrc("test_b")])
        self.assertEqual(testrunner._count(suite), 2)
        modules = sorted(type(t).__module__ for t in testrunner._iter_tests(suite))
        self.assertEqual(modules, ["ctmj.tests.test_thing", "tests.test_thing"])


class DiscoveryGuardTests(SimpleTestCase):
    def test_empty_root_is_rejected(self):
        """A root that collects nothing must be an error, not a pass.

        This is the property that makes the runner worth having: without it, a
        renamed or moved test directory produces a green run.
        """
        with self.assertRaises(AssertionError) as ctx:
            testrunner.RunTests().assert_no_silent_skips(
                {"ctmj": 152, "src": 0}
            )
        message = str(ctx.exception)
        self.assertIn("src", message)
        self.assertIn("pass silently", message)

    def test_fully_covered_roots_are_accepted(self):
        # Must not raise.
        testrunner.RunTests().assert_no_silent_skips({"ctmj": 152, "src": 142})

    def test_entirely_empty_run_is_rejected(self):
        with self.assertRaises(AssertionError):
            testrunner.RunTests().assert_no_silent_skips({"ctmj": 0, "src": 0})

    def test_both_roots_are_declared(self):
        self.assertIn("ctmj", testrunner.TEST_ROOTS)
        self.assertIn("src", testrunner.TEST_ROOTS)

    def test_the_suite_actually_contains_tests_from_both_roots(self):
        """The guard is not merely present, it is satisfied by this repository.

        Reads the real test modules' ``__module__`` values rather than
        re-running discovery, so it cannot be affected by import order — which
        is what made an earlier version of the check unreliable.

        The lookup descends one level because ``iter_modules`` on a package
        yields its subpackages, not the modules inside them: both roots hold a
        ``tests`` package, and the test modules live beneath that.
        """
        import importlib
        import pkgutil

        seen: set[str] = set()
        for root in testrunner.TEST_ROOTS:
            tests_package = importlib.import_module(f"{root}.tests")
            for info in pkgutil.iter_modules(tests_package.__path__):
                if not info.name.startswith("test"):
                    continue
                module = importlib.import_module(f"{root}.tests.{info.name}")
                found_any = False
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, unittest.TestCase)
                        and obj.__module__ == module.__name__
                    ):
                        found_any = True
                        break
                if found_any:
                    seen.add(root)
                    break

        self.assertEqual(
            seen,
            set(testrunner.TEST_ROOTS),
            msg="a declared test root contains no test cases",
        )
