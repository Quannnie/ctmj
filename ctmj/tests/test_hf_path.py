"""Tests for the Hugging Face storage-bucket client.

The bucket path was misdiagnosed twice before this module existed: once as a
Google Cloud Storage prefix to be removed, and once as a bad token. Both
readings came from the same error message,
``FileNotFoundError: repository not found``, which is what a bad token looks
like and also what a wrong address looks like.

These tests pin the two facts that settle it, neither of which needs a token or
the network: the addressing rules, and the shape of the two endpoints.
"""

from __future__ import annotations

import unittest
from unittest import mock

from django.test import SimpleTestCase

from ctmj.services import hub

BUCKET = "quanghuynh0122/datamining_models"

#: A model repository, addressed differently from a bucket. Publishing requires
#: one -- buckets are read-only mirrors -- so this is a real target rather than a
#: hypothetical one.
MODEL_REPO = "quanghuynh0122/ctmj-demo-models"


class BucketIdTests(SimpleTestCase):
    def test_owner_and_bucket_name_are_both_required(self):
        self.assertEqual(hub.validate_bucket_id(BUCKET), BUCKET)

    def test_surrounding_slashes_and_space_are_tolerated(self):
        for messy in (f"  {BUCKET}  ", f"/{BUCKET}", f"{BUCKET}/", f"/{BUCKET}/"):
            with self.subTest(value=messy):
                self.assertEqual(hub.validate_bucket_id(messy), BUCKET)

    def test_bare_name_is_rejected(self):
        """One segment has no owner, so the URL would name the wrong thing.

        The resulting 404 names neither the bucket nor the owner, so the
        mistake would be diagnosed as a missing repository.
        """
        for bad in ("datamining_models", "", "   "):
            with self.subTest(value=bad):
                with self.assertRaises(hub.HubError) as ctx:
                    hub.validate_bucket_id(bad)
                self.assertIn("owner", str(ctx.exception))

    def test_three_segments_are_rejected(self):
        """``buckets/...`` is added by the URL builder, not part of the id.

        Prepending it again produces ``buckets/buckets/...`` and a 404 that
        looks like a missing bucket.
        """
        with self.assertRaises(hub.HubError):
            hub.validate_bucket_id(f"buckets/{BUCKET}")

    def test_error_message_names_the_setting(self):
        with self.assertRaises(hub.HubError) as ctx:
            hub.validate_bucket_id("just-a-name")
        self.assertIn("CTMJ_HF_REPO", str(ctx.exception))


class UrlTests(SimpleTestCase):
    def test_bucket_url_includes_the_buckets_segment(self):
        """The regression, in the direction it originally went wrong.

        ``buckets`` is the bucket namespace in a Hub URL, not a storage-vendor
        prefix. An earlier revision removed it on the assumption that it was
        one, which broke a path that was otherwise correct.
        """
        url = hub.artefact_url(BUCKET, "dbscan_clustering_model.pkl", kind=hub.BUCKET)
        self.assertEqual(
            url,
            f"https://huggingface.co/buckets/{BUCKET}/resolve/dbscan_clustering_model.pkl",
        )

    def test_bucket_url_has_no_revision_segment(self):
        """Where a bucket differs from a repository.

        A repository serves ``/resolve/{revision}/{path}``. A bucket serves
        ``/resolve/{path}``. Probing the repository shape returns 404 for a file
        that is present, which is easy to read as a missing artefact.
        """
        url = hub.artefact_url(BUCKET, "dbscan_clustering_model.pkl", kind=hub.BUCKET)
        self.assertIn("/resolve/dbscan_clustering_model.pkl", url)
        self.assertNotIn("/resolve/main", url)

    def test_repository_url_carries_the_revision(self):
        """The other half of the difference.

        This is the shape a model repository uses, and the one a loader that
        only spoke the bucket scheme 404s on for every file -- which is
        indistinguishable from the upload having failed.
        """
        url = hub.artefact_url(MODEL_REPO, "dbscan_clustering_model.pkl", kind=hub.REPO)
        self.assertEqual(
            url,
            f"https://huggingface.co/{MODEL_REPO}/resolve/main/dbscan_clustering_model.pkl",
        )
        self.assertNotIn("/buckets/", url)

    def test_repository_revision_is_configurable(self):
        url = hub.artefact_url(
            MODEL_REPO, "x.pkl", kind=hub.REPO, revision="a1b2c3d"
        )
        self.assertIn("/resolve/a1b2c3d/x.pkl", url)

    def test_the_two_schemes_are_not_interchangeable(self):
        """Stated as a test because the difference is one path segment.

        Reading a file from the wrong scheme produces a 404 that names the
        file, not the address, so the natural conclusion is that the file is
        missing.
        """
        bucket_url = hub.artefact_url(BUCKET, "x.pkl", kind=hub.BUCKET)
        repo_url = hub.artefact_url(MODEL_REPO, "x.pkl", kind=hub.REPO)
        self.assertNotEqual(bucket_url, repo_url)
        self.assertIn("/resolve/x.pkl", bucket_url)
        self.assertIn("/resolve/main/x.pkl", repo_url)

    def test_every_artefact_gets_a_valid_url(self):
        from django.conf import settings

        for key, filename in settings.CTMJ["MODEL_FILES"].items():
            with self.subTest(artefact=key):
                for kind in (hub.BUCKET, hub.REPO):
                    url = hub.artefact_url(BUCKET, filename, kind=kind)
                    self.assertTrue(url.startswith("https://huggingface.co/"))
                    self.assertIn(f"/{filename}", url)

    def test_filename_is_percent_encoded(self):
        url = hub.artefact_url(BUCKET, "a file with spaces.pkl", kind=hub.BUCKET)
        self.assertNotIn(" ", url)
        self.assertIn("%20", url)

    def test_invalid_bucket_produces_no_url(self):
        with self.assertRaises(hub.HubError):
            hub.artefact_url("nope", "x.pkl", kind=hub.BUCKET)


class HeaderTests(SimpleTestCase):
    def test_no_token_means_no_header(self):
        """The bucket is public, so an anonymous read is legitimate.

        Requiring a token made a correctly configured deployment report
        "models unavailable" for no reason.
        """
        import os

        saved = {
            k: os.environ.pop(k, None)
            for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }
        try:
            self.assertEqual(hub._headers(None), {})
            self.assertEqual(hub._headers(""), {})
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_token_is_sent_as_bearer(self):
        headers = hub._headers("hf_example")
        self.assertEqual(headers, {"Authorization": "Bearer hf_example"})

    def test_environment_token_is_picked_up(self):
        import os

        os.environ["HF_TOKEN"] = "hf_from_env"
        try:
            self.assertEqual(
                hub._headers(None), {"Authorization": "Bearer hf_from_env"}
            )
        finally:
            os.environ.pop("HF_TOKEN", None)


class ListingTests(SimpleTestCase):
    def test_bucket_file_parses_a_listing_entry(self):
        entry = hub.BucketFile.from_json(
            {
                "type": "file",
                "path": "dbscan_clustering_model.pkl",
                "size": 705495,
                "uploadedAt": "2026-05-29T08:00:32.009Z",
                "xetHash": "abc",
            }
        )
        self.assertEqual(entry.path, "dbscan_clustering_model.pkl")
        self.assertEqual(entry.size, 705495)
        self.assertEqual(entry.uploaded_at, "2026-05-29T08:00:32.009Z")

    def test_missing_fields_default_safely(self):
        entry = hub.BucketFile.from_json({"path": "x.pkl"})
        self.assertEqual(entry.size, 0)
        self.assertEqual(entry.uploaded_at, "")

    def test_listing_hits_the_bucket_tree_endpoint(self):
        """A bucket is not a repository, so no repo endpoint applies.

        ``/api/models/...``, ``/api/datasets/...`` and ``/api/spaces/...`` all
        answer 404 for a bucket, which reads exactly like "it does not exist".

        Asserted by observing the request rather than by reading the source:
        the URL is assembled from constants, so a source-text check would be
        asserting on the implementation's spelling.
        """
        import requests

        from ctmj.services import hub as hub_module

        seen: dict = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return [{"type": "file", "path": "a.pkl", "size": 10}]

        def fake_get(url, headers=None, timeout=None, **kwargs):
            seen["url"] = url
            seen["headers"] = headers
            return _Response()

        original = requests.get
        requests.get = fake_get
        try:
            entries = hub_module.list_bucket(BUCKET)
        finally:
            requests.get = original

        self.assertEqual(
            seen["url"],
            f"https://huggingface.co/api/buckets/{BUCKET}/tree",
        )
        for repo_endpoint in ("/api/models/", "/api/datasets/", "/api/spaces/"):
            self.assertNotIn(repo_endpoint, seen["url"])
        self.assertEqual([e.path for e in entries], ["a.pkl"])

    def test_401_with_a_credential_is_reported_as_one(self):
        """Not as a missing bucket.

        The two produce different remedies, and conflating them is what made
        this cost a full debugging cycle: the original error said "repository
        not found", which sent the investigation to the token.

        A token has to be in the environment for this to be the case being
        tested -- see the anonymous-401 test for the other half.
        """
        import os

        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()

        class _Response:
            status_code = 401
            text = "unauthorized"

        original = requests.get
        requests.get = lambda *a, **k: _Response()
        try:
            with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_example"}, clear=False):
                with self.assertRaises(hub_module.HubError) as ctx:
                    hub_module.list_bucket(BUCKET)
        finally:
            requests.get = original
            hub_module.forget_kind()
        message = str(ctx.exception)
        self.assertIn("401", message)
        self.assertIn("HF_TOKEN", message)

    def test_wrong_address_says_so_rather_than_404(self):
        """The address being wrong is a different problem from a file missing.

        ``resolve_kind`` probes both schemes, so a name that is neither a bucket
        nor a repository fails the probe and never reaches the listing call. The
        message says exactly that, which is the diagnosis that was missing for
        most of this investigation -- a 404 naming a file sends you looking for
        the file.
        """
        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()

        class _Response:
            status_code = 404
            text = "Repository not found"

        import os

        original = requests.get
        requests.get = lambda *a, **k: _Response()
        try:
            with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_example"}, clear=False):
                with self.assertRaises(hub_module.HubError) as ctx:
                    hub_module.list_bucket(BUCKET)
        finally:
            requests.get = original
            hub_module.forget_kind()
        message = str(ctx.exception)
        self.assertIn("neither a readable storage bucket", message)
        self.assertIn(BUCKET, message)
        # With a credential in play, the anonymity caveat does not apply.
        self.assertNotIn("No HF_TOKEN was sent", message)

    def test_probe_order_prefers_the_bucket(self):
        """The bucket endpoint is tried first.

        It is the scheme the real artefacts use, and its endpoint is
        unambiguous. Getting this wrong would cost a wasted round trip on every
        cold start.
        """
        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()
        seen: list[str] = []

        class _Response:
            status_code = 404
            text = "not found"

        def fake_get(url, headers=None, timeout=None, **kwargs):
            seen.append(url)
            return _Response()

        original = requests.get
        requests.get = fake_get
        try:
            with self.assertRaises(hub_module.HubError):
                hub_module.resolve_kind(BUCKET)
        finally:
            requests.get = original
            hub_module.forget_kind()

        self.assertEqual(len(seen), 2)
        self.assertIn("/api/buckets/", seen[0])
        self.assertIn("/api/models/", seen[1])

    def test_a_401_during_the_probe_is_an_auth_problem_not_a_wrong_address(self):
        """The two send an operator in opposite directions.

        Treating 401 as "not this kind of address" makes the probe try the
        other scheme, fail again, and report that the address is wrong -- which
        is how an expired token came to be investigated as a naming problem.
        """
        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()
        calls: list[str] = []

        class _Response:
            status_code = 401
            text = "unauthorized"

        def fake_get(url, headers=None, timeout=None, **kwargs):
            calls.append(url)
            return _Response()

        import os

        original = requests.get
        requests.get = fake_get
        try:
            with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_example"}, clear=False):
                with self.assertRaises(hub_module.HubError) as ctx:
                    hub_module.resolve_kind(BUCKET)
        finally:
            requests.get = original
            hub_module.forget_kind()

        self.assertEqual(len(calls), 1, "401 should stop the probe, not continue it")
        self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_an_anonymous_401_is_not_blamed_on_the_token(self):
        """Without a credential, 401 means private-or-absent, not "bad token".

        An unconditional credential error would send the reader to fix a token
        that was never sent, so the probe has to keep trying instead.
        """
        import os

        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()
        calls: list[str] = []

        class _Response:
            status_code = 401
            text = "unauthorized"

        def fake_get(url, headers=None, timeout=None, **kwargs):
            calls.append(url)
            return _Response()

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        }
        original = requests.get
        requests.get = fake_get
        try:
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(hub_module.HubError) as ctx:
                    hub_module.resolve_kind(BUCKET)
        finally:
            requests.get = original
            hub_module.forget_kind()

        self.assertEqual(len(calls), 2, "the probe should try both schemes")
        message = str(ctx.exception)
        self.assertIn("No HF_TOKEN was sent", message)
        self.assertIn("neither a readable storage bucket", message)

    def test_probe_result_is_remembered(self):
        """One probe per address, not one per artefact.

        Five artefacts at five probes would be five needless round trips on
        every cold start.
        """
        import requests

        from ctmj.services import hub as hub_module

        hub_module.forget_kind()
        calls: list[str] = []

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return [{"path": "x.pkl", "size": 1}]

        def fake_get(url, headers=None, timeout=None, **kwargs):
            calls.append(url)
            return _Response()

        original = requests.get
        requests.get = fake_get
        try:
            for _ in range(4):
                self.assertEqual(
                    hub_module.resolve_kind(BUCKET, token="t"), hub_module.BUCKET
                )
        finally:
            requests.get = original
            hub_module.forget_kind()

        self.assertEqual(len(calls), 1, "the probe result was not reused")

    def test_forget_kind_clears_the_cache(self):
        from ctmj.services import hub as hub_module

        hub_module.forget_kind()
        self.assertIsNone(hub_module._recall("anything/at-all"))
        hub_module._remember(hub_module.BUCKET, "anything/at-all")
        self.assertEqual(hub_module._recall("anything/at-all"), hub_module.BUCKET)
        hub_module.forget_kind()
        self.assertIsNone(hub_module._recall("anything/at-all"))


class HuggingFaceHubIncompatibilityTests(SimpleTestCase):
    """The reason this client exists rather than a two-line fsspec call."""

    def test_huggingface_hub_rejects_a_bucket_id(self):
        """``huggingface_hub`` validates the id before making any request."""
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HFValidationError

        with self.assertRaises((HFValidationError, ValueError)):
            HfApi(token="hf_dummy").repo_info(f"buckets/{BUCKET}", repo_type="model")


class NetworkTests(SimpleTestCase):
    """Opt-in checks against the live bucket."""

    def setUp(self):
        super().setUp()
        if not __import__("os").environ.get("CTMJ_TEST_HF_NETWORK"):
            self.skipTest("set CTMJ_TEST_HF_NETWORK=1 to hit the Hub")

    def test_bucket_lists_the_expected_artefacts(self):
        listing = {item.path: item.size for item in hub.list_bucket(BUCKET)}
        from django.conf import settings

        for key, filename in settings.CTMJ["MODEL_FILES"].items():
            with self.subTest(artefact=key):
                self.assertIn(filename, listing)
                self.assertGreater(listing[filename], 0)

    def test_small_artefact_downloads_and_deserialises(self):
        from django.conf import settings

        import joblib

        filename = settings.CTMJ["MODEL_FILES"]["predicting_preprocessor"]
        payload = hub.fetch_artefact(BUCKET, filename)
        self.assertGreater(len(payload), 0)
        self.assertEqual(payload[:2], b"\x80\x04", "not a pickle stream")
        joblib.loads(payload)  # must not raise

    def test_missing_file_is_named(self):
        with self.assertRaises(hub.HubError) as ctx:
            hub.fetch_artefact(BUCKET, "definitely_not_here.pkl")
        self.assertIn("definitely_not_here.pkl", str(ctx.exception))


del unittest  # only imported for the skip decorator's readability
