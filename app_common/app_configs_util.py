import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

LOGGER = logging.getLogger(__name__)

class AppConfigs:
    """
    A configuration manager that loads environment variables (from .env or system)
    and provides typed accessors for common data types.
    """

    def __init__(self):
        self._configs = {}

    def __str__(self):
        return f"AppConfigs({self._configs})"

    def load_app_configs(self, dotenv_path: str = "../.env"):
        LOGGER.info(f"STARTED Loading application configurations from {dotenv_path} and environment variables.")
        """Load all environment variables and store in _configs dict."""

        load_dotenv(dotenv_path)
        for key, value in os.environ.items():
            self._configs[key] = value

        for key, value in self._configs.items():
            LOGGER.info(f"Config loaded: {key}={value}")

        LOGGER.info(f"Loaded {len(self._configs)} configuration entries. ")
        LOGGER.info(f"COMPLETED Loading application configurations from {dotenv_path} and environment variables.")

    def get(self, key: str, default=None):
        """Return the raw config value as-is."""
        return self._configs.get(key, default)

    def get_str(self, key: str, default: str = None) -> str:
        """Return value as string."""
        value = self._configs.get(key, default)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = None) -> int:
        """Return value as int."""
        value = self._configs.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Return value as bool."""
        value = str(self._configs.get(key, "")).lower()
        if value in ("true", "1", "yes", "y"):
            return True
        elif value in ("false", "0", "no", "n"):
            return False
        return default

    def get_date(self, key: str, fmt: str = "%Y-%m-%d", default=None):
        """Return value as datetime.date."""
        value = self._configs.get(key)
        if not value:
            return default
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            return default

    def get_json(self, key: str, default=None):
        """Return value parsed as JSON/dict."""
        value = self._configs.get(key)
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def set(self, key: str, value):
        """Set or update a config key-value."""
        if isinstance(value, (dict, list)):
            self._configs[key] = json.dumps(value)
        elif isinstance(value, datetime):
            self._configs[key] = value.isoformat()
        else:
            self._configs[key] = str(value)
        os.environ[key] = self._configs[key]

    def all(self):
        """Return all configs as a dict."""
        return dict(self._configs)
