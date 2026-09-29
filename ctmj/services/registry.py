"""Thread-safe, failure-tolerant loader for the trained model artefacts.

Why this exists
---------------
The original implementation called :func:`input` from ``AppConfig.ready()`` to
ask for a Hugging Face token. That had two failure modes:

  1. Under ``runserver`` the reloader's double-import meant the prompt appeared
     twice, and any non-interactive context (Docker, CI, systemd) hung or
     crashed on EOF.
  2. The guard was ``RUN_MAIN == 'true' or not DEBUG``, which is *false* during
     ordinary local development. Models therefore never loaded in DEBUG, and
     every prediction request failed with "models not loaded". The feature was
     effectively dead in both development and production.

This module removes the prompt entirely: credentials come from the environment,
loading happens on a background thread, and any failure is recorded as state
rather than raised, so the site always starts and can explain the situation.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from django.conf import settings

logger = logging.getLogger("ctmj.registry")


class LoadState(str, Enum):
    """Lifecycle of the artefact set."""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"


#: Human-readable copy for each state, surfaced in the UI.
STATE_MESSAGES: dict[LoadState, str] = {
    LoadState.IDLE: "Mô hình chưa được nạp.",
    LoadState.LOADING: "Đang nạp mô hình vào bộ nhớ, vui lòng chờ trong giây lát.",
    LoadState.READY: "Mô hình sẵn sàng.",
    LoadState.FAILED: "Không nạp được bộ mô hình.",
    LoadState.DISABLED: "Tính năng dự báo đang tạm tắt theo cấu hình.",
}

#: Which artefacts must be present for inference to run.
REQUIRED_KEYS: tuple[str, ...] = (
    "dbscan",
    "spectral",
    "gradient_boosting",
    "user_data_preprocessor",
    "predicting_preprocessor",
)


@dataclass
class RegistryStatus:
    """Serialisable snapshot of the loader, safe to hand to a template."""

    state: LoadState = LoadState.IDLE
    message: str = ""
    source: str = ""
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.state is LoadState.READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "message": self.message,
            "source": self.source,
            "loaded": list(self.loaded),
            "missing": list(self.missing),
            "error": self.error,
        }


class ModelRegistry:
    """Holds the five joblib artefacts and the state of their loading.

    The instance is a process-wide singleton created in ``AppConfig.ready()``.
    Reads are lock-free (a plain dict read is atomic under the GIL) and writes
    happen once, so concurrent requests never block on the loader.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config if config is not None else getattr(settings, "CTMJ", {})
        self._config = cfg
        self._model_files: dict[str, str] = dict(cfg.get("MODEL_FILES", {}))
        self._source: str = str(cfg.get("MODEL_SOURCE", "auto")).lower()
        self._model_dir: Path = Path(cfg.get("MODEL_DIR", Path("models")))
        self._hf_repo: str = str(cfg.get("HF_REPO", ""))

        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = RegistryStatus(source=self._source)

        if not self._model_files:
            raise ValueError("CTMJ.MODEL_FILES is empty; the registry has no contract to load.")

        self._status.state = (
            LoadState.DISABLED if self._source == "none" else LoadState.IDLE
        )
        self._status.message = STATE_MESSAGES[self._status.state]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def status(self) -> RegistryStatus:
        with self._lock:
            return self._status

    def get(self, key: str) -> Any:
        """Return a loaded artefact, or ``None`` when unavailable."""
        return self._models.get(key)

    def ready(self) -> bool:
        return all(self._models.get(key) is not None for key in REQUIRED_KEYS)

    def start_background_load(self) -> None:
        """Kick off loading on a worker thread and return immediately.

        Safe to call more than once; subsequent calls are no-ops while a load
        is in flight or has already completed.

        The thread is deliberately *not* a daemon. Deserialising a joblib
        artefact imports modules referenced by the pickle, and killing the
        interpreter mid-import can deadlock the import lock for the main
        thread. A bounded ``atexit`` join lets a normal exit wait for the
        load to finish without risking an indefinite hang.
        """
        if self._status.state is LoadState.DISABLED:
            logger.info("Model loading disabled via CTMJ_MODEL_SOURCE=none")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._status.state is LoadState.READY:
                return
            self._status.state = LoadState.LOADING
            self._status.message = STATE_MESSAGES[LoadState.LOADING]
            self._status.error = ""
            self._thread = threading.Thread(
                target=self._load_guarded, name="ctmj-model-loader", daemon=False
            )
            self._thread.start()
        atexit.register(self._join_on_exit)
        logger.info("Loading model artefacts in background thread (source=%s)", self._source)

    def _join_on_exit(self) -> None:
        """Wait briefly for the loader at interpreter shutdown."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        timeout = float(self._config.get("LOAD_TIMEOUT_SECONDS", 180))
        logger.info("Waiting up to %.0fs for model loading to finish at exit", timeout)
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("Model loading still running at exit; abandoning it.")

    def load_sync(self) -> RegistryStatus:
        """Load on the calling thread. Used by management commands and tests."""
        if self._status.state is LoadState.DISABLED:
            return self.status
        with self._lock:
            self._status.state = LoadState.LOADING
            self._status.message = STATE_MESSAGES[LoadState.LOADING]
        return self._load_guarded()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the load finishes. Returns True when ready."""
        if timeout is None:
            timeout = float(self._config.get("LOAD_TIMEOUT_SECONDS", 180))
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.ready()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_guarded(self) -> RegistryStatus:
        try:
            source = self._resolve_source()
            if source is None:
                with self._lock:
                    self._status.state = LoadState.DISABLED
                    self._status.message = (
                        "Không tìm thấy bộ mô hình. Đặt các tệp .pkl vào "
                        f"{self._model_dir} hoặc cấu hình CTMJ_MODEL_SOURCE=hf."
                    )
                    self._status.error = "no artefacts found"
                logger.warning(self._status.message)
                return self.status

            with self._lock:
                self._status.source = source

            loaded, missing = self._load_from(source)
            with self._lock:
                self._models.update(loaded)
                # ``loaded`` and ``missing`` both report *filenames*, so an
                # operator reading /health/ sees a consistent picture of which
                # artefacts are present and which to supply.
                self._status.loaded = sorted(self._model_files[k] for k in loaded)
                self._status.missing = sorted(missing)
                if missing:
                    self._status.state = LoadState.FAILED
                    self._status.error = f"missing artefacts: {', '.join(sorted(missing))}"
                    self._status.message = (
                        "Thiếu " + str(len(missing)) + " tệp mô hình: " + ", ".join(sorted(missing))
                    )
                    logger.error(self._status.message)
                else:
                    self._status.state = LoadState.READY
                    self._status.error = ""
                    self._status.message = STATE_MESSAGES[LoadState.READY]
                    logger.info("Model artefacts ready from %s: %s", source, ", ".join(sorted(loaded)))
        except Exception as exc:  # noqa: BLE001 - never let loading kill startup
            with self._lock:
                self._status.state = LoadState.FAILED
                self._status.error = f"{type(exc).__name__}: {exc}"
                self._status.message = (
                    "Không nạp được mô hình từ nguồn cấu hình. "
                    "Trang vẫn hoạt động nhưng tính năng dự báo tạm thời không dùng được."
                )
            logger.exception("Model loading failed: %s", exc)
        return self.status

    def _resolve_source(self) -> str | None:
        """Pick a concrete source, or ``None`` when there is nothing to load."""
        if self._source == "local":
            return "local"
        if self._source == "hf":
            return "hf"
        if self._source != "auto":
            raise ValueError(f"Unsupported CTMJ_MODEL_SOURCE={self._source!r}")

        # auto: prefer local artefacts, fall back to the Hub.
        if self._local_available():
            return "local"
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return "hf"
        return None

    def _local_available(self) -> bool:
        if not self._model_dir.is_dir():
            return False
        return all((self._model_dir / name).is_file() for name in self._model_files.values())

    def _load_from(self, source: str) -> tuple[dict[str, Any], list[str]]:
        """Load every artefact from ``source``; never raises for a missing file."""
        loader: Callable[[str, Path], Any]
        if source == "local":
            loader = self._load_local
        elif source == "hf":
            loader = self._load_hf
        else:  # pragma: no cover - guarded by _resolve_source
            raise ValueError(f"Unknown source {source!r}")

        loaded: dict[str, Any] = {}
        missing: list[str] = []
        for key, filename in self._model_files.items():
            try:
                loaded[key] = loader(key, Path(filename))
            except FileNotFoundError:
                missing.append(filename)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load %s (%s): %s", key, filename, exc)
                missing.append(filename)
        return loaded, missing

    def _load_local(self, key: str, filename: Path) -> Any:
        import joblib  # imported lazily: keeps `manage.py` commands cheap

        path = self._model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(str(path))
        logger.debug("Loading %s from %s", key, path)
        return joblib.load(path)

    def _load_hf(self, key: str, filename: Path) -> Any:
        import joblib

        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fsspec is required for CTMJ_MODEL_SOURCE=hf. "
                "Install it with: pip install fsspec huggingface_hub"
            ) from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise FileNotFoundError(
                "HF_TOKEN is not set; cannot read the private model repository."
            )

        remote = f"buckets/{self._hf_repo}/{filename}"
        logger.debug("Loading %s from hub://%s", key, remote)
        fs = fsspec.filesystem("hf", token=token)
        with fs.open(remote, "rb") as handle:
            return joblib.load(handle)
