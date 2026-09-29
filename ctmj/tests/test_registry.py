"""Artefact loading, failure handling and status reporting.

The registry replaced an ``input()`` prompt in ``AppConfig.ready()`` that
blocked startup and made the feature unreachable in development. These tests
pin the replacement behaviour: no prompting, no crashing, and a status the UI
can act on.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import joblib
from django.test import SimpleTestCase

from ctmj.services.registry import (
    REQUIRED_KEYS,
    LoadState,
    ModelRegistry,
    RegistryStatus,
)

#: A minimal contract mirroring settings.CTMJ["MODEL_FILES"].
FILES = {
    "dbscan": "dbscan.pkl",
    "spectral": "spectral.pkl",
    "gradient_boosting": "gb.pkl",
    "user_data_preprocessor": "user_pre.pkl",
    "predicting_preprocessor": "predict_pre.pkl",
}


def _config(source: str, model_dir: Path) -> dict:
    return {
        "MODEL_SOURCE": source,
        "MODEL_DIR": model_dir,
        "HF_REPO": "someone/models",
        "MODEL_FILES": FILES,
        "LOAD_TIMEOUT_SECONDS": 5,
    }


class _TempDirTestCase(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name)

    def write_artefacts(self, only: set[str] | None = None, value=object()) -> None:
        for key, filename in FILES.items():
            if only is not None and key not in only:
                continue
            joblib.dump(value, self.model_dir / filename)


class ConstructionTestCase(SimpleTestCase):
    def test_empty_file_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelRegistry({"MODEL_FILES": {}})

    def test_source_none_starts_disabled(self):
        registry = ModelRegistry(_config("none", Path(".")))
        self.assertIs(registry.status.state, LoadState.DISABLED)

    def test_status_serialises_for_json(self):
        payload = RegistryStatus(state=LoadState.READY, message="ok").as_dict()
        self.assertEqual(payload["state"], "ready")
        self.assertEqual(payload["loaded"], [])


class DisabledTestCase(_TempDirTestCase):
    def test_start_background_load_is_a_no_op_when_disabled(self):
        registry = ModelRegistry(_config("none", self.model_dir))
        registry.start_background_load()
        self.assertIs(registry.status.state, LoadState.DISABLED)
        self.assertFalse(registry.ready())

    def test_load_sync_is_a_no_op_when_disabled(self):
        registry = ModelRegistry(_config("none", self.model_dir))
        self.assertIs(registry.load_sync().state, LoadState.DISABLED)


class LocalLoadTestCase(_TempDirTestCase):
    def test_loads_every_artefact(self):
        self.write_artefacts(value={"tag": "artefact"})
        registry = ModelRegistry(_config("local", self.model_dir))
        status = registry.load_sync()
        self.assertIs(status.state, LoadState.READY)
        self.assertEqual(set(status.loaded), set(FILES.values()))
        self.assertEqual(status.missing, [])

    def test_registry_reports_ready(self):
        self.write_artefacts(value={"tag": "artefact"})
        registry = ModelRegistry(_config("local", self.model_dir))
        registry.load_sync()
        self.assertTrue(registry.ready())
        for key in REQUIRED_KEYS:
            self.assertIsNotNone(registry.get(key))

    def test_missing_artefacts_are_reported_not_raised(self):
        self.write_artefacts(only={"dbscan"})
        registry = ModelRegistry(_config("local", self.model_dir))
        status = registry.load_sync()
        self.assertIs(status.state, LoadState.FAILED)
        # ``missing`` holds filenames, matching what the user must supply.
        self.assertEqual(
            set(status.missing), set(FILES.values()) - {"dbscan.pkl"}
        )
        self.assertFalse(registry.ready())

    def test_missing_error_names_the_files(self):
        self.write_artefacts(only={"dbscan"})
        registry = ModelRegistry(_config("local", self.model_dir))
        status = registry.load_sync()
        self.assertIn("spectral.pkl", status.error)
        self.assertIn("dbscan.pkl", status.loaded)

    def test_empty_directory_fails_cleanly(self):
        registry = ModelRegistry(_config("local", self.model_dir))
        status = registry.load_sync()
        self.assertIs(status.state, LoadState.FAILED)
        self.assertTrue(status.error)

    def test_non_directory_fails_cleanly(self):
        registry = ModelRegistry(_config("local", self.model_dir / "missing"))
        status = registry.load_sync()
        self.assertIs(status.state, LoadState.FAILED)

    def test_corrupt_artefact_is_reported_not_raised(self):
        (self.model_dir / "dbscan.pkl").write_bytes(b"not a pickle")
        self.write_artefacts(only=set(FILES) - {"dbscan"})
        registry = ModelRegistry(_config("local", self.model_dir))
        status = registry.load_sync()
        self.assertIs(status.state, LoadState.FAILED)
        self.assertIn("dbscan.pkl", status.missing)

    def test_start_background_load_completes(self):
        self.write_artefacts(value={"tag": "artefact"})
        registry = ModelRegistry(_config("local", self.model_dir))
        registry.start_background_load()
        self.assertTrue(registry.wait_until_ready(timeout=10))
        self.assertIs(registry.status.state, LoadState.READY)

    def test_repeated_starts_do_not_reload(self):
        self.write_artefacts(value={"tag": "artefact"})
        registry = ModelRegistry(_config("local", self.model_dir))
        registry.load_sync()
        loaded = registry.get("dbscan")
        registry.start_background_load()
        self.assertIs(registry.get("dbscan"), loaded)


class SourceResolutionTestCase(_TempDirTestCase):
    def test_auto_prefers_local_when_present(self):
        self.write_artefacts()
        registry = ModelRegistry(_config("auto", self.model_dir))
        self.assertEqual(registry._resolve_source(), "local")

    def test_auto_falls_back_to_hub_when_a_token_is_present(self):
        registry = ModelRegistry(_config("auto", self.model_dir))
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}, clear=False):
            self.assertEqual(registry._resolve_source(), "hf")

    def test_auto_gives_up_without_local_files_or_token(self):
        registry = ModelRegistry(_config("auto", self.model_dir))
        env = {k: v for k, v in os.environ.items() if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(registry._resolve_source())

    def test_auto_with_nothing_available_reports_disabled(self):
        registry = ModelRegistry(_config("auto", self.model_dir))
        env = {k: v for k, v in os.environ.items() if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
        with mock.patch.dict(os.environ, env, clear=True):
            status = registry.load_sync()
        self.assertIs(status.state, LoadState.DISABLED)
        # The message must tell the operator where to put the artefacts.
        self.assertIn(str(self.model_dir), status.message)
        self.assertIn(".pkl", status.message)

    def test_explicit_local_never_touches_the_network(self):
        registry = ModelRegistry(_config("local", self.model_dir))
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}, clear=False):
            self.assertEqual(registry._resolve_source(), "local")


class HubLoadTestCase(_TempDirTestCase):
    """The Hub path needs no network in tests - the bucket client is mocked."""

    def _payload(self, tag: str = "from-hub") -> bytes:
        import io

        buffer = io.BytesIO()
        joblib.dump({"tag": tag}, buffer)
        return buffer.getvalue()

    def test_loads_every_artefact_through_the_hub_client(self):
        """All five come from the same address, one download each.

        ``resolve_kind`` is mocked as well as the download: the client probes
        the address to learn whether it is a bucket or a repository, and an
        unmocked probe reaches the network.
        """
        payload = self._payload()
        seen: list[str] = []

        def fake_fetch(bucket, filename, token=None, timeout=None, **kwargs):
            seen.append(filename)
            return payload

        registry = ModelRegistry(_config("hf", self.model_dir))
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}, clear=False):
            with mock.patch("ctmj.services.hub.resolve_kind", return_value="bucket"):
                with mock.patch(
                    "ctmj.services.hub.fetch_artefact", side_effect=fake_fetch
                ):
                    status = registry.load_sync()

        self.assertIs(status.state, LoadState.READY)
        self.assertEqual(registry.get("dbscan"), {"tag": "from-hub"})
        self.assertEqual(len(seen), 5)

    def test_anonymous_read_is_allowed(self):
        """The bucket is public, so a token must not be a precondition.

        The old loader refused without ``HF_TOKEN`` and reported "models
        unavailable". On a correctly configured deployment that was a false
        alarm: the read would have succeeded anonymously.
        """
        payload = self._payload("anonymous")
        registry = ModelRegistry(_config("hf", self.model_dir))
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("ctmj.services.hub.resolve_kind", return_value="bucket"):
                with mock.patch(
                    "ctmj.services.hub.fetch_artefact", return_value=payload
                ) as fetch:
                    status = registry.load_sync()

        self.assertIs(status.state, LoadState.READY)
        self.assertEqual(registry.get("dbscan"), {"tag": "anonymous"})
        self.assertIsNone(fetch.call_args.kwargs.get("token"))

    def test_a_download_failure_names_the_artefact_and_the_reason(self):
        """``missing`` says which file; ``errors`` says why.

        Both are needed and they answer different questions. A 403, a file that
        is not in the bucket, and an artefact pickled against a different
        scikit-learn all land in ``missing``, and the remedy differs in each
        case.
        """
        from ctmj.services import hub

        def fake_fetch(bucket, filename, token=None, timeout=None, **kwargs):
            if filename.startswith("dbscan"):
                raise hub.HubError("not there")
            return self._payload()

        registry = ModelRegistry(_config("hf", self.model_dir))
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_fake"}, clear=False):
            with mock.patch("ctmj.services.hub.resolve_kind", return_value="bucket"):
                with mock.patch(
                    "ctmj.services.hub.fetch_artefact", side_effect=fake_fetch
                ):
                    status = registry.load_sync()

        self.assertIs(status.state, LoadState.FAILED)
        # The filename list stays bare: that is what an operator supplies.
        self.assertEqual(status.missing, ["dbscan.pkl"])
        # The reason is alongside it, keyed by the same filename.
        self.assertIn("dbscan.pkl", status.errors)
        self.assertIn("not there", status.errors["dbscan.pkl"])
        self.assertEqual(set(status.as_dict()["errors"]), {"dbscan.pkl"})
