"""Tests for the Hugging Face path construction.

The HF loading path had never been exercised end to end, and it was wrong: the
artefact path carried a ``buckets/`` prefix, a Google Cloud Storage convention
that does not apply to ``HfFileSystem``. Because that prefix shifted the
repository namespace by one segment, every load failed with
``FileNotFoundError: repository not found`` — a message that points squarely at
the token or the repository name, neither of which was at fault.

These tests pin the path format. They are hermetic: no token, no network, no
private repository. The one test that does need the network is opt-in, because
the rest of the suite is guaranteed to run without it.
"""

from __future__ import annotations

import unittest

from django.test import SimpleTestCase

from ctmj.services.registry import ModelRegistry

REPO = "quanghuynh0122/datamining_models"


class HFPathTests(SimpleTestCase):
    def _registry(self, repo: str = REPO) -> ModelRegistry:
        return ModelRegistry(
            config={
                "MODEL_FILES": {"dbscan": "dbscan_clustering_model.pkl"},
                "MODEL_SOURCE": "hf",
                "MODEL_DIR": "models",
                "HF_REPO": repo,
            }
        )

    def test_path_is_namespace_then_file(self):
        """Two segments of namespace, then the file within the repo."""
        registry = self._registry()
        self.assertEqual(
            registry.hf_path("dbscan_clustering_model.pkl"),
            f"{REPO}/dbscan_clustering_model.pkl",
        )

    def test_path_has_no_storage_service_prefix(self):
        """The regression.

        ``buckets/`` is a GCS convention. On the Hub it is read as part of the
        repository namespace, so the loader asked for a repository literally
        named ``buckets/quanghuynh0122`` and got a 404 for it.

        The check is on leading segments rather than a substring, because the
        configured repository is itself called ``datamining_models`` — a
        substring test for ``models/`` would flag its own name.
        """
        registry = self._registry()
        for filename in (
            "dbscan_clustering_model.pkl",
            "spectral_clustering_model.pkl",
            "GradientBoostingClassifier_model.pkl",
            "user_data_preprocessor.pkl",
            "s1_predicting_preprocessor.pkl",
        ):
            with self.subTest(artefact=filename):
                path = registry.hf_path(filename)
                segments = path.split("/")
                for leaked in ("buckets", "models", "datasets", "spaces", "gs"):
                    self.assertNotIn(
                        leaked, segments[:2],
                        f"{leaked!r} leaked into the repository namespace",
                    )
                for scheme in ("gs://", "hf://", "http://", "https://"):
                    self.assertFalse(path.startswith(scheme), path)

    def test_first_two_segments_are_the_namespace(self):
        """Matches how ``HfFileSystem.resolve_path`` splits the path.

        That method reads the repository namespace as ``path.split("/")[:2]``.
        A path with any extra leading segment therefore names a different
        repository, which is exactly the failure this guards.
        """
        registry = self._registry()
        path = registry.hf_path("dbscan_clustering_model.pkl")
        segments = path.split("/")
        self.assertEqual(len(segments), 3)
        self.assertEqual("/".join(segments[:2]), REPO)
        self.assertEqual(segments[2], "dbscan_clustering_model.pkl")

    def test_every_configured_artefact_gets_a_three_segment_path(self):
        from django.conf import settings

        registry = self._registry()
        for key, filename in settings.CTMJ["MODEL_FILES"].items():
            with self.subTest(artefact=key):
                segments = registry.hf_path(filename).split("/")
                self.assertEqual(len(segments), 3, filename)
                self.assertEqual("/".join(segments[:2]), REPO)

    def test_repo_without_a_namespace_is_rejected_up_front(self):
        """A bare repository name cannot address a Hub repo.

        It would silently resolve against the *caller's* personal namespace
        instead, so it is a configuration error rather than something to let
        the 404 discover.
        """
        registry = self._registry(repo="just-a-name")
        with self.assertRaises(ValueError) as ctx:
            registry.validate_hf_repo()
        self.assertIn("namespace", str(ctx.exception).lower())

    @unittest.skipUnless(
        __import__("os").environ.get("CTMJ_TEST_HF_NETWORK"),
        "set CTMJ_TEST_HF_NETWORK=1 to check path resolution against the Hub",
    )
    def test_path_resolves_against_a_real_repository(self):
        """The only test that proves the format against the live library.

        Uses a public model repository so it needs no credentials. Off by
        default: the rest of the suite runs without network access.
        """
        from huggingface_hub import HfFileSystem

        registry = self._registry(repo="openai-community/gpt2")
        resolved = HfFileSystem().resolve_path(registry.hf_path("config.json"))
        self.assertEqual(resolved.repo_id, "openai-community/gpt2")
        self.assertEqual(resolved.path_in_repo, "config.json")
