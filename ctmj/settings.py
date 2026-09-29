"""
Django settings for the CJPS project (Customer Journey Prediction System).

Configuration is environment-driven so the same code runs in development, CI
and production without edits. A ``.env`` file at the repository root is read
automatically when python-dotenv is installed; otherwise plain environment
variables are used. See ``.env.example`` for the supported variables.

Design rules followed here:
  * No secret is ever committed. SECRET_KEY falls back to a generated ephemeral
    value in DEBUG, and *refuses to boot* in production without a real key.
  * ALLOWED_HOSTS / CSRF_TRUSTED_ORIGINS are explicit, never wildcarded
    implicitly.
  * Static files are served by WhiteNoise in production so no external
    web server is required for a single-container deploy.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
try:  # optional dependency — keeps the project runnable without python-dotenv
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without dotenv
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

load_dotenv(BASE_DIR / ".env", override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Core security settings
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "").strip()
DEBUG = _env_bool("DJANGO_DEBUG", default=True)

if not SECRET_KEY:
    if DEBUG:
        # Ephemeral key: sessions/CSRF break across restarts, but nothing leaks
        # and no real credential is ever written to disk or source control.
        SECRET_KEY = secrets.token_urlsafe(64)
    else:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set when DJANGO_DEBUG=False. "
            "Generate one with: python -c \"from django.core.management.utils "
            "import get_random_secret_key as k; print(k())\""
        )

ALLOWED_HOSTS = _env_list("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost" if DEBUG else "")

if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must list at least one host when DJANGO_DEBUG=False."
    )

CSRF_TRUSTED_ORIGINS = [
    origin for origin in _env_list("DJANGO_CSRF_TRUSTED_ORIGINS") if origin
]

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
# Discovers both ctmj/tests and src/tests. The default runner found only the
# former, because src/ imports as a namespace package and unittest's discovery
# does not descend into those. See ctmj/testrunner.py.
TEST_RUNNER = "ctmj.testrunner.RunTests"

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ctmj.apps.CtmjConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "ctmj.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ctmj.wsgi.application"
ASGI_APPLICATION = "ctmj.asgi.application"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3"),
        "OPTIONS": {
            # Survive concurrent reads while a prediction is running.
            "timeout": 20,
        },
        "TEST": {"NAME": None},  # in-memory database during tests
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------------------------
# Internationalisation — the UI is Vietnamese
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "vi"
TIME_ZONE = "Asia/Ho_Chi_Minh"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": (
            "whitenoise.storage.CompressedManifestStaticFilesStorage"
            if not DEBUG
            else "django.contrib.staticfiles.storage.StaticFilesStorage"
        )
    },
}

# ---------------------------------------------------------------------------
# Transport security — enforced only behind TLS in production
# ---------------------------------------------------------------------------
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = not DEBUG
SECURE_HSTS_SECONDS = 0 if DEBUG else 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = not DEBUG
SECURE_HSTS_PRELOAD = not DEBUG
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

# ---------------------------------------------------------------------------
# Sessions & messages
# ---------------------------------------------------------------------------
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

MESSAGE_STORAGE = "django.contrib.messages.storage.session.SessionStorage"

# ---------------------------------------------------------------------------
# Logging
#
# Application code uses ``logging.getLogger("ctmj")`` instead of print(). The
# console handler is quiet in production and verbose in development.
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(levelname)s %(name)s: %(message)s"},
        "verbose": {
            "format": "%(asctime)s %(levelname)-7s %(name)s [%(module)s:%(lineno)d] %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose" if DEBUG else "plain",
        }
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
    "loggers": {
        "ctmj": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}

# ---------------------------------------------------------------------------
# CJPS specific configuration
# ---------------------------------------------------------------------------
CTMJ = {
    # "auto" | "local" | "hf" | "none"
    "MODEL_SOURCE": os.environ.get("CTMJ_MODEL_SOURCE", "auto").strip().lower(),
    "MODEL_DIR": Path(os.environ.get("CTMJ_MODEL_DIR", BASE_DIR / "models")),
    "HF_REPO": os.environ.get("CTMJ_HF_REPO", "quanghuynh0122/datamining_models"),
    # Contract between the training pipeline (src/main.py) and the web app.
    # Changing a key here without retraining breaks inference.
    "MODEL_FILES": {
        "dbscan": "dbscan_clustering_model.pkl",
        "spectral": "spectral_clustering_model.pkl",
        "gradient_boosting": "GradientBoostingClassifier_model.pkl",
        "user_data_preprocessor": "user_data_preprocessor.pkl",
        "predicting_preprocessor": "s1_predicting_preprocessor.pkl",
    },
    # Loading happens off the request thread; a slow first load should never
    # delay the first HTTP response.
    "LOAD_TIMEOUT_SECONDS": float(os.environ.get("CTMJ_LOAD_TIMEOUT", "180")),
}

if CTMJ["MODEL_SOURCE"] not in {"auto", "local", "hf", "none"}:
    raise ImproperlyConfigured(
        f"CTMJ_MODEL_SOURCE must be auto|local|hf|none, got {CTMJ['MODEL_SOURCE']!r}"
    )
