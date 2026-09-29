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

from django.test import SimpleTestCase

from ctmj.services import hub

BUCKET = "quanghuynh0122/datamining_models"


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
    def test_url_includes_the_buckets_segment(self):
        """The regression, in the direction it originally went wrong.

        ``buckets`` is the bucket namespace in a Hub URL, not a storage-vendor
        prefix. An earlier revision removed it on the assumption that it was
        one, which broke a path that was otherwise correct.
        """
        url = hub.artefact_url(BUCKET, "dbscan_clustering_model.pkl")
        self.assertEqual(
            url,
            f"https://huggingface.co/buckets/{BUCKET}/resolve/dbscan_clustering_model.pkl",
        )
        self.assertIn("/buckets/", url)

    def test_url_has_no_revision_segment(self):
        """Where a bucket differs from a repository.

        A repository serves ``/resolve/{revision}/{path}``. A bucket serves
        ``/resolve/{path}``. Probing the repository shape returns 404 for a file
        that is present, which is easy to read as a missing artefact.
        """
        url = hub.artefact_url(BUCKET, "dbscan_clustering_model.pkl")
        self.assertIn("/resolve/dbscan_clustering_model.pkl", url)
        self.assertNotIn("/resolve/main/", url)
        self.assertNotIn("/resolve/main", url)

    def test_every_artefact_gets_a_valid_url(self):
        from django.conf import settings

        for key, filename in settings.CTMJ["MODEL_FILES"].items():
            with self.subTest(artefact=key):
                url = hub.artefact_url(BUCKET, filename)
                self.assertTrue(url.startswith("https://huggingface.co/buckets/"))
                self.assertIn(f"/{filename}", url)

    def test_filename_is_percent_encoded(self):
        """Names are concatenated into a path, so anything unusual is escaped."""
        url = hub.artefact_url(BUCKET, "a file with spaces.pkl")
        self.assertNotIn(" ", url)
        self.assertIn("%20", url)

    def test_invalid_bucket_produces_no_url(self):
        with self.assertRaises(hub.HubError):
            hub.artefact_url("nope", "x.pkl")


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

    def test_401_is_reported_as_a_credential_problem(self):
        """Not as a missing bucket.

        The two produce different remedies, and conflating them is what made
        this cost a full debugging cycle: the original error said "repository
        not found", which sent the investigation to the token.
        """
        import requests

        from ctmj.services import hub as hub_module

        class _Response:
            status_code = 401
            text = "unauthorized"

        original = requests.get
        requests.get = lambda *a, **k: _Response()
        try:
            with self.assertRaises(hub_module.HubError) as ctx:
                hub_module.list_bucket(BUCKET)
        finally:
            requests.get = original
        message = str(ctx.exception)
        self.assertIn("401", message)
        self.assertIn("HF_TOKEN", message)

    def test_404_is_reported_as_a_missing_bucket(self):
        import requests

        from ctmj.services import hub as hub_module

        class _Response:
            status_code = 404
            text = "Repository not found"

        original = requests.get
        requests.get = lambda *a, **k: _Response()
        try:
            with self.assertRaises(hub_module.HubError) as ctx:
                hub_module.list_bucket(BUCKET)
        finally:
            requests.get = original
        self.assertIn("404", str(ctx.exception))


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
