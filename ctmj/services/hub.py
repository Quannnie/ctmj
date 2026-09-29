"""Reading model artefacts from a Hugging Face *storage bucket*.

What the artefacts actually are
-------------------------------
They live at ``buckets/quanghuynh0122/datamining_models``, and that is a
Hugging Face **storage bucket** — ``"repoType": "bucket"`` — not a model
repository. The distinction is not cosmetic; it decides which tools work:

* ``huggingface_hub`` refuses the id outright. Its validator accepts
  ``repo_name`` or ``namespace/repo_name`` and there is no ``repo_type`` that
  fits, so ``HfApi.repo_info`` raises ``HFValidationError`` before any request
  is made. The same applies to ``HfFileSystem``, so the ``fsspec.filesystem("hf")``
  path this module used to take could not have loaded a bucket under any
  circumstances.
* ``/api/models/...``, ``/api/datasets/...`` and ``/api/spaces/...`` all answer
  404 for a bucket, which reads exactly like "the repository does not exist".

Two URL shapes are in play, and they are not interchangeable with a
repository's:

    list       GET /api/buckets/{bucket}/tree
    download   GET /buckets/{bucket}/resolve/{filename}

Note there is **no revision segment**. A repository serves
``/resolve/{revision}/{path}``; a bucket serves ``/resolve/{path}``. Probing
``/resolve/main/{file}`` returns 404, which is easy to misread as a missing
file when the file is in fact right there.

A correction worth recording
----------------------------
An earlier revision of this code built ``buckets/{repo}/{filename}`` and
``HfFileSystem`` was used to open it. Two conclusions were drawn from that and
both were wrong. The ``buckets/`` segment was taken for a Google Cloud Storage
convention when it is in fact the bucket namespace, so a "fix" removed it and
broke a path that was one missing piece from working. And the failure was
attributed to the token, because ``FileNotFoundError: repository not found``
is what a bad token looks like. The bucket was public the whole time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

logger = logging.getLogger("ctmj.hub")

#: Base for the bucket REST endpoints.
HF_BASE = "https://huggingface.co"

#: Path segment that identifies a storage bucket in a URL. Part of the address,
#: not a storage-vendor prefix.
BUCKET_SEGMENT = "buckets"

#: Longest side we will download without complaint. The spectral artefact is
#: ~488 MB because ``SpectralClustering`` persists its n-by-n affinity matrix, so
#: a limit below a few hundred megabytes would reject the real set.
DEFAULT_TIMEOUT = 900


class HubError(RuntimeError):
    """Raised when a bucket or artefact cannot be read. Message is explicit."""


@dataclass(frozen=True)
class BucketFile:
    """One entry from a bucket listing."""

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
    """Normalise and check a bucket id.

    A bucket is addressed as ``{owner}/{bucket}``. A single segment has no
    owner, and the request would then be built against the wrong URL, producing
    a 404 that names neither the bucket nor the owner.
    """
    cleaned = (bucket or "").strip().strip("/")
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) != 2:
        raise HubError(
            f"CTMJ_HF_REPO must be '{'{owner}'}/{'{bucket}'}', got {bucket!r}. "
            "A Hugging Face storage bucket is addressed by owner and bucket name."
        )
    return "/".join(parts)


def list_bucket(bucket: str, token: str | None = None) -> list[BucketFile]:
    """List a bucket's contents.

    Used by the ``check`` management command and by ``download_all``. Listing
    first means a missing artefact is reported by name instead of surfacing as a
    404 on the first attempt to load it.
    """
    import requests

    bucket = validate_bucket_id(bucket)
    url = f"{HF_BASE}/api/{BUCKET_SEGMENT}/{bucket}/tree"
    headers = _headers(token)

    try:
        response = requests.get(url, headers=headers, timeout=60)
    except Exception as exc:  # noqa: BLE001
        raise HubError(f"Could not reach {url}: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 401:
        raise HubError(
            "Hugging Face rejected the credentials (401). Set HF_TOKEN, and "
            "check that it has not expired and can read the bucket."
        )
    if response.status_code == 404:
        raise HubError(
            f"Bucket {bucket!r} not found (404). If it exists but is private, "
            "HF_TOKEN must belong to an account with read access."
        )
    if response.status_code != 200:
        raise HubError(
            f"Listing {bucket!r} failed: HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HubError(f"Bucket listing for {bucket!r} was not JSON") from exc

    if not isinstance(payload, list):
        raise HubError(f"Unexpected bucket listing shape: {type(payload).__name__}")

    return [BucketFile.from_json(item) for item in payload if item.get("path")]


def artefact_url(bucket: str, filename: str) -> str:
    """The download URL for one artefact.

    ``/buckets/{bucket}/resolve/{filename}`` — deliberately **no revision
    segment**, which is where this differs from a repository's
    ``/resolve/{rev}/{path}``. The filename is percent-encoded because artefact
    names contain no slashes but the path is assembled by string concatenation.
    """
    bucket = validate_bucket_id(bucket)
    return f"{HF_BASE}/{BUCKET_SEGMENT}/{bucket}/resolve/{quote(filename)}"


def _headers(token: str | None) -> dict[str, str]:
    resolved = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    # The header is only attached when a token exists. This bucket is public, so
    # an anonymous read is legitimate and a missing token is not by itself an
    # error -- treating it as one previously made a working setup look broken.
    if not resolved:
        return {}
    return {"Authorization": f"Bearer {resolved}"}


def fetch_artefact(
    bucket: str,
    filename: str,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> bytes:
    """Download one artefact into memory.

    Streaming with a size cap so a wrong URL cannot stream an unbounded body
    into the process. The 488 MB spectral artefact is deserialised anyway, so a
    generous ceiling is appropriate.
    """
    import requests

    url = artefact_url(bucket, filename)
    try:
        with requests.get(
            url, headers=_headers(token), timeout=timeout, stream=True
        ) as response:
            if response.status_code in (401, 403):
                raise HubError(
                    f"Access denied downloading {filename!r} (HTTP "
                    f"{response.status_code}). Check that HF_TOKEN is valid and "
                    "can read the bucket."
                )
            if response.status_code == 404:
                raise HubError(
                    f"{filename!r} is not in bucket {bucket!r} (404). "
                    "Run 'python manage.py check_hub_bucket' to list its contents."
                )
            if response.status_code != 200:
                raise HubError(
                    f"Downloading {filename!r} failed: HTTP {response.status_code}"
                )

            declared = response.headers.get("content-length")
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                total += len(chunk)
                # 2 GB: far above the largest real artefact, low enough to stop
                # a misdirected request from filling memory.
                if total > 2_000_000_000:
                    raise HubError(
                        f"{filename!r} exceeded the 2 GB download ceiling; "
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


def load_artefact(bucket: str, filename: str, token: str | None = None) -> Any:
    """Download one artefact and deserialise it."""
    import io

    import joblib

    payload = fetch_artefact(bucket, filename, token=token)
    try:
        return joblib.load(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001
        raise HubError(
            f"{filename!r} downloaded but could not be deserialised: "
            f"{type(exc).__name__}: {exc}. A scikit-learn version mismatch is "
            "the usual cause; the artefact records the version it was trained with."
        ) from exc
