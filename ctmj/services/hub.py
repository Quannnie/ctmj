"""Reading model artefacts from Hugging Face.

Two surfaces, and which one applies decides the whole URL scheme
---------------------------------------------------------------
The artefacts may live in a **storage bucket** (``buckets/{owner}/{name}``) or
in an ordinary **model repository** (``{owner}/{name}``). They are addressed
differently, and a loader that only knows one of them fails on the other with a
404 that names neither the token nor the file:

    bucket   list       GET /api/buckets/{id}/tree
             download   GET /buckets/{id}/resolve/{filename}
             -- no revision segment

    repo     list       GET /api/models/{id}/tree/{revision}
             download   GET /{id}/resolve/{revision}/{filename}
             -- a revision segment, defaulting to ``main``

Nothing in the request says which one it is, so it is determined by probing
once and then remembered. Either can be tried first: a bucket answers 404 on
the repository endpoints and a repository answers 404 on the bucket endpoints.
The bucket is tried first because that is where the real artefacts live, and
because its endpoint is unambiguous.

Why ``huggingface_hub`` is not used
-----------------------------------
It cannot address a bucket at all. The id validator accepts ``repo_name`` or
``namespace/repo_name``, no ``repo_type`` fits, and ``HfApi.repo_info`` raises
``HFValidationError`` before making a request. ``HfFileSystem`` is the same, so
the ``fsspec.filesystem("hf")`` path this module originally used could not have
read a bucket under any circumstances. A bucket is not a repository. The HTTP
client is therefore spelled out here, and the repository case goes through the
same code path rather than through a second implementation.

What is not possible
--------------------
Buckets are read-only mirrors of external cloud storage. There is no upload
endpoint: eight candidate routes answer 404, the page has no upload control, and
its embedded state carries ``canReadRepoContent`` with no write flag. Every
"upload" string on a bucket page is the ``uploadedAt`` column. Publishing
artefacts means creating a model repository instead.

Corrections worth recording, because each cost a debugging cycle
----------------------------------------------------------------
* The ``buckets/`` segment was twice taken for a Google Cloud Storage
  convention and "fixed" away. It is the bucket namespace.
* The failure was attributed to the token, because
  ``FileNotFoundError: repository not found`` is what a bad token looks like --
  and also what a wrong address looks like. The bucket was public throughout.
* A loader that spoke only the bucket scheme was the reason a freshly uploaded
  set of artefacts 404'd on every file while the upload itself had succeeded.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote

logger = logging.getLogger("ctmj.hub")

#: Base for every Hugging Face endpoint.
HF_BASE = "https://huggingface.co"

#: Path segment that identifies a storage bucket in a URL. Part of the address,
#: not a storage-vendor prefix.
BUCKET_SEGMENT = "buckets"

#: Default branch for a model repository. Buckets have no equivalent, which is
#: precisely the difference between the two URL schemes.
DEFAULT_REVISION = "main"

#: How long a probe result is remembered. The kind of a repository does not
#: change, and a request per artefact would double the round trips.
PROBE_TIMEOUT = 300

#: Longest download we will accept. The spectral artefact is ~488 MB because
#: ``SpectralClustering`` persists its n-by-n affinity matrix, so a limit below a
#: few hundred megabytes would reject the real set.
MAX_DOWNLOAD_BYTES = 2_000_000_000

DEFAULT_TIMEOUT = 900

BUCKET = "bucket"
REPO = "repo"


class HubError(RuntimeError):
    """Raised when artefacts cannot be read. Message is explicit."""


@dataclass(frozen=True)
class BucketFile:
    """One entry from a listing."""

    path: str
    size: int
    uploaded_at: str = ""

    @classmethod
    def from_json(cls, payload: dict) -> "BucketFile":
        return cls(
            path=str(payload.get("path", "")),
            size=int(payload.get("size") or 0),
            uploaded_at=str(payload.get("uploadedAt") or ""),
        )


def validate_bucket_id(bucket: str) -> str:
    """Normalise and check the id.

    A bucket or repository is addressed as ``{owner}/{name}``. A single segment
    has no owner, and the request would be built against the wrong URL, giving a
    404 that names neither the owner nor the artefact.
    """
    cleaned = (bucket or "").strip().strip("/")
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) != 2:
        raise HubError(
            f"CTMJ_HF_REPO must be '{'{owner}'}/{'{name}'}', got {bucket!r}. "
            "A Hugging Face bucket or repository is addressed by owner and name."
        )
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Which kind of address is this?
# ---------------------------------------------------------------------------

_probe_cache: dict[str, tuple[str, float]] = {}
_probe_lock = threading.Lock()


def _remember(kind: str, bucket: str) -> None:
    import time

    with _probe_lock:
        _probe_cache[bucket] = (kind, time.monotonic())


def _recall(bucket: str) -> str | None:
    import time

    with _probe_lock:
        entry = _probe_cache.get(bucket)
    if entry is None:
        return None
    kind, when = entry
    if time.monotonic() - when > PROBE_TIMEOUT:
        return None
    return kind


def _headers(token: str | None) -> dict[str, str]:
    resolved = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    # No header when there is no token. The production bucket is public, so an
    # anonymous read is legitimate and a missing token is not by itself an
    # error -- treating it as one made a working setup look broken.
    if not resolved:
        return {}
    return {"Authorization": f"Bearer {resolved}"}


def _list_url(kind: str, bucket: str, revision: str) -> str:
    if kind == BUCKET:
        return f"{HF_BASE}/api/{BUCKET_SEGMENT}/{bucket}/tree"
    return f"{HF_BASE}/api/models/{bucket}/tree/{quote(revision, safe='')}"


def _download_url(kind: str, bucket: str, filename: str, revision: str) -> str:
    name = quote(filename, safe="")
    if kind == BUCKET:
        # No revision: that is the whole difference from a repository.
        return f"{HF_BASE}/{BUCKET_SEGMENT}/{bucket}/resolve/{name}"
    return f"{HF_BASE}/{bucket}/resolve/{quote(revision, safe='')}/{name}"


def resolve_kind(bucket: str, token: str | None = None, revision: str = DEFAULT_REVISION) -> str:
    """Decide whether ``bucket`` names a storage bucket or a model repository.

    Probed once and remembered. A 401 is not a "wrong kind" answer -- it is an
    authentication failure and must be reported as one, because "check your
    token" and "this is not a bucket" send an operator in opposite directions.
    """
    import requests

    bucket = validate_bucket_id(bucket)
    cached = _recall(bucket)
    if cached is not None:
        return cached

    headers = _headers(token)
    for kind in (BUCKET, REPO):
        url = _list_url(kind, bucket, revision)
        try:
            response = requests.get(url, headers=headers, timeout=60)
        except Exception as exc:  # noqa: BLE001
            raise HubError(f"Could not reach {url}: {type(exc).__name__}: {exc}") from exc

        if response.status_code == 200:
            _remember(kind, bucket)
            logger.info("Resolved %s as a Hugging Face %s", bucket, kind)
            return kind
        if response.status_code in (401, 403) and "Authorization" in headers:
            # Only a *credential* failure when a credential was actually sent.
            # An anonymous request for a private or non-existent name answers 401
            # as well, and treating that as "your token is wrong" points the
            # reader at the token when the address is the problem -- the exact
            # misdirection this module exists to avoid.
            raise HubError(
                f"Hugging Face rejected the credentials ({response.status_code}) "
                f"for {bucket!r}. Set HF_TOKEN and check that it has not expired."
            )
        # 404, or a 401 with no credential, means "not this kind of address".

    detail = ""
    if "Authorization" not in headers:
        detail = (
            " No HF_TOKEN was sent, so a private repository is "
            "indistinguishable from a non-existent one."
        )
    raise HubError(
        f"{bucket!r} is neither a readable storage bucket nor a model "
        f"repository. Check the owner and name.{detail}"
    )


def forget_kind(bucket: str | None = None) -> None:
    """Drop a cached probe result. Used by tests and by the management command."""
    with _probe_lock:
        if bucket is None:
            _probe_cache.clear()
        else:
            _probe_cache.pop(bucket, None)


def artefact_url(
    bucket: str,
    filename: str,
    kind: str | None = None,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
) -> str:
    """The download URL for one artefact.

    ``kind`` may be supplied to skip the probe; otherwise it is resolved.
    """
    bucket = validate_bucket_id(bucket)
    kind = kind or resolve_kind(bucket, token=token, revision=revision)
    return _download_url(kind, bucket, filename, revision)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_bucket(
    bucket: str,
    token: str | None = None,
    revision: str = DEFAULT_REVISION,
    kind: str | None = None,
) -> list[BucketFile]:
    """List the artefacts' contents.

    Listing first means a missing artefact is reported by name rather than
    surfacing as a 404 on the first attempt to load it.
    """
    import requests

    bucket = validate_bucket_id(bucket)
    kind = kind or resolve_kind(bucket, token=token, revision=revision)
    url = _list_url(kind, bucket, revision)
    headers = _headers(token)

    try:
        response = requests.get(url, headers=headers, timeout=60)
    except Exception as exc:  # noqa: BLE001
        raise HubError(f"Could not reach {url}: {type(exc).__name__}: {exc}") from exc

    if response.status_code in (401, 403):
        raise HubError(
            f"Hugging Face rejected the credentials ({response.status_code}). "
            "Set HF_TOKEN and check that it has not expired."
        )
    if response.status_code == 404:
        raise HubError(f"{bucket!r} not found (404). Check the owner and name.")
    if response.status_code != 200:
        raise HubError(
            f"Listing {bucket!r} failed: HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HubError(f"Listing for {bucket!r} was not JSON") from exc

    if not isinstance(payload, list):
        raise HubError(f"Unexpected listing shape: {type(payload).__name__}")

    return [BucketFile.from_json(item) for item in payload if item.get("path")]


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def fetch_artefact(
    bucket: str,
    filename: str,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    revision: str = DEFAULT_REVISION,
    kind: str | None = None,
) -> bytes:
    """Download one artefact into memory.

    If the address turns out to be the other kind, the probe result is dropped
    and the download retried once. That is what makes a stale probe harmless:
    the answer cannot be wrong for long.
    """
    import requests

    bucket = validate_bucket_id(bucket)
    resolved = kind or resolve_kind(bucket, token=token, revision=revision)

    for attempt in (resolved, BUCKET if resolved == REPO else REPO):
        url = _download_url(attempt, bucket, filename, revision)
        try:
            with requests.get(
                url, headers=_headers(token), timeout=timeout, stream=True
            ) as response:
                if response.status_code in (401, 403):
                    raise HubError(
                        f"Access denied downloading {filename!r} (HTTP "
                        f"{response.status_code}). Check that HF_TOKEN is valid "
                        "and can read this repository."
                    )
                if response.status_code == 404:
                    last = response.status_code
                    continue
                if response.status_code != 200:
                    raise HubError(
                        f"Downloading {filename!r} failed: HTTP {response.status_code}"
                    )

                declared = response.headers.get("content-length")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise HubError(
                            f"{filename!r} exceeded the "
                            f"{MAX_DOWNLOAD_BYTES:,}-byte download ceiling; "
                            "refusing to continue."
                        )
                    chunks.append(chunk)

                if declared and int(declared) != total:
                    raise HubError(
                        f"{filename!r} downloaded incompletely: {total:,} of "
                        f"{int(declared):,} bytes."
                    )
                return b"".join(chunks)
        except requests.RequestException as exc:
            raise HubError(f"Could not download {filename!r}: {exc}") from exc

    if attempt != resolved:
        # The first scheme 404'd; try the other once, then remember the answer.
        kind = attempt
        _remember(attempt, bucket)
        logger.info(
            "Re-resolved %s: the %s scheme 404'd, so it is a %s",
            bucket, resolved, attempt,
        )
        return fetch_artefact(
            bucket, filename, token=token, timeout=timeout,
            revision=revision, kind=attempt,
        )

    raise HubError(
        f"{filename!r} could not be fetched from {bucket!r} (HTTP 404). "
        "Run the hub check command to list what is actually there."
    )


def load_artefact(
    bucket: str,
    filename: str,
    token: str | None = None,
    revision: str = DEFAULT_REVISION,
    kind: str | None = None,
) -> Any:
    """Download one artefact and deserialise it."""
    import io

    import joblib

    payload = fetch_artefact(
        bucket, filename, token=token, revision=revision, kind=kind
    )
    try:
        return joblib.load(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001
        raise HubError(
            f"{filename!r} downloaded but could not be deserialised: "
            f"{type(exc).__name__}: {exc}. A scikit-learn version mismatch is "
            "the usual cause; the artefact records the version it was trained with."
        ) from exc


def available_artefacts(
    bucket: str, filenames: Iterable[str], token: str | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict[str, bool]:
    """Which of ``filenames`` are present, without downloading them.

    A single listing answers the whole question, where one request per file
    would cost a round trip each to learn the same thing.
    """
    present = {item.path for item in list_bucket(bucket, token=token, revision=revision)}
    return {name: name in present for name in filenames}
