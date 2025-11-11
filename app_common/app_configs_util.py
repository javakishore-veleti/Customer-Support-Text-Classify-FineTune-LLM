from __future__ import annotations

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging
from threading import Lock

# Below is to support forward references in type hints i.e. return AppConfigs as a Builder pattern

LOGGER = logging.getLogger(__name__)


class Builder:
    def __init__(self, cls):
        self._instance = cls()

    def __getattr__(self, name):
        # automatically chain any callable method on the instance
        attr = getattr(self._instance, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # only chain if the original method returns None or self
                return self if result is None or result is self._instance else result

            return wrapper
        return attr

    def build(self):
        return self._instance


class AppConfigs:

    _instance = None
    _lock = Lock()

    """
    A configuration manager that loads environment variables (from .env or system)
    and provides typed accessors for common data types.
    """

    def __init__(self):
        self._configs = {}
        self.configs_loaded: bool = False

    def __str__(self):
        return f"AppConfigs({self._configs})"

    # ------------------------------
    # Singleton Accessor
    # ------------------------------
    @classmethod
    def get_instance(cls) -> "AppConfigs":
        """
        Return a singleton instance of AppConfigs.
        Automatically loads configs once on first use.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    LOGGER.info("Creating new AppConfigs singleton instance.")
                    cls._instance = AppConfigs()
                    cls._instance.load_app_configs()
        return cls._instance

    # ------------------------------
    # Builder Implementation
    # ------------------------------
    @classmethod
    def builder(cls):
        return Builder(cls)

    # ------------------------------
    # Loaders and Accessors
    # ------------------------------
    def load_app_configs(self, dotenv_path: str = "../.env") -> "AppConfigs":
        if self.configs_loaded:
            LOGGER.info("App configurations already loaded. Skipping reload.")
            return self

        LOGGER.info(f"STARTED Loading application configurations from {dotenv_path} and environment variables.")
        load_dotenv(dotenv_path)

        for key, value in os.environ.items():
            self._configs[key] = value

        self.configs_loaded = True
        LOGGER.info(f"Loaded {len(self._configs)} configuration entries.")
        return self

    def get(self, key: str, default=None):
        return self._configs.get(key, default)

    def get_str(self, key: str, default: str = None) -> str:
        value = self._configs.get(key, default)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = None) -> int:
        value = self._configs.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = str(self._configs.get(key, "")).lower()
        if value in ("true", "1", "yes", "y"):
            return True
        elif value in ("false", "0", "no", "n"):
            return False
        return default

    def get_json(self, key: str, default=None):
        value = self._configs.get(key)
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def set(self, key: str, value) -> "AppConfigs":
        if isinstance(value, (dict, list)):
            self._configs[key] = json.dumps(value)
        elif isinstance(value, datetime):
            self._configs[key] = value.isoformat()
        else:
            self._configs[key] = str(value)
        os.environ[key] = self._configs[key]
        return self

    def all(self):
        return dict(self._configs)

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get_str(key, str(default)))
        except (ValueError, TypeError):
            return default
