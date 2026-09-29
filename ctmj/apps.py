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

        # Registered before the deferral check below, and deliberately so.
        # Invalidation has nothing to do with model loading: any process that
        # writes a lookup row — a test, a management command, a worker — needs
        # it, and every one of those paths defers loading. Registering it after
        # the early return meant a stale label survived in exactly the processes
        # that could most easily have caused it.
        self._register_cache_invalidation()

        if self._should_defer_loading():
            logger.debug("Skipping model loading for this process (%s).", self._process_kind())
            return

        cls.registry.start_background_load()

    @staticmethod
    def _register_cache_invalidation() -> None:
        """Drop the reference-data cache whenever a lookup row changes.

        The lookup tables are cached in memory (see
        :mod:`ctmj.services.lookups`), and ``/admin/`` is a live editor for
        them. Without this, correcting a touchpoint's name in the admin would
        have no effect until the cache expired — and the person who just made
        the correction is the one most likely to go and check it worked.

        ``bulk_create`` and ``queryset.update`` bypass these signals, so the
        management commands that use them invalidate explicitly.
        """
        from django.db.models.signals import post_delete, post_save

        from ctmj import models
        from ctmj.services.lookups import invalidate_all

        def _invalidate(sender, **kwargs):  # noqa: ANN001, ARG001
            invalidate_all()

        for model in models.REFERENCE_MODELS:
            post_save.connect(_invalidate, sender=model, dispatch_uid=f"ctmj.cache.save.{model.__name__}")
            post_delete.connect(_invalidate, sender=model, dispatch_uid=f"ctmj.cache.delete.{model.__name__}")

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
            "make_favicon",
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
