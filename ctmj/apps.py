"""Application configuration.

Model artefacts are loaded lazily on a background thread. ``ready()`` returns
immediately so the first HTTP response is never delayed by a multi-second
joblib deserialisation, and a loading failure degrades to a visible status
message instead of a crash. See :mod:`ctmj.services.registry`.
"""

from __future__ import annotations

import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger("ctmj")


class CtmjConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ctmj"
    verbose_name = "Hệ thống Dự báo CJPS"

    #: Process-wide model registry. Populated during ``ready()``.
    registry = None

    def ready(self) -> None:
        # Importing models here would trigger AppRegistryNotReady during
        # migrations, so the registry is constructed from settings only.
        from ctmj.services.registry import ModelRegistry

        # Assign through the class, not ``self``. ``ready()`` runs on an
        # AppConfig *instance*, so ``self.registry = ...`` would create an
        # instance attribute while ``CtmjConfig.registry`` — the name the views
        # use — still reads ``None``.
        cls = type(self)
        if cls.registry is None:
            cls.registry = ModelRegistry()

        if self._should_defer_loading():
            logger.debug("Skipping model loading for this process (%s).", self._process_kind())
            return

        cls.registry.start_background_load()

    # Commands that never serve HTTP requests. Starting a background download
    # for these would make `migrate` and `test` hang on a cold model cache.
    _NON_SERVING_COMMANDS = frozenset(
        {
            "migrate",
            "makemigrations",
            "test",
            "collectstatic",
            "shell",
            "check",
            "seed_reference_data",
            "build_demo_models",
        }
    )

    @classmethod
    def _process_kind(cls) -> str:
        """Best-effort identification of the running command."""
        argv = sys.argv[1:] if sys.argv and sys.argv[0].endswith((".py", ".exe")) else []
        for arg in argv:
            token = arg.lstrip("-")
            if token in cls._NON_SERVING_COMMANDS:
                return token
        return "serve" if "runserver" in argv or not argv else "unknown"

    @classmethod
    def _should_defer_loading(cls) -> bool:
        if os.environ.get("CTMJ_SKIP_MODEL_LOAD", "").lower() in {"1", "true", "yes"}:
            return True
        return cls._process_kind() in cls._NON_SERVING_COMMANDS
