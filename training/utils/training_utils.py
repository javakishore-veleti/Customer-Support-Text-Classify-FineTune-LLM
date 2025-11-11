import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from app_common.app_configs_util import AppConfigs

LOGGER = logging.getLogger(__name__)


class TrainingUtils:
    """
    Singleton utility for reading training-related configurations.
    Reads values from AppConfigs singleton and returns results as dicts.
    Does NOT store configs on the instance.
    """

    _instance = None
    _lock = Lock()

    def __init__(self):
        # Prevent accidental re-init
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

    # ------------------------------
    # Singleton Accessor
    # ------------------------------
    @classmethod
    def get_instance(cls) -> "TrainingUtils":
        """
        Return the singleton instance of TrainingUtils.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    LOGGER.info("Creating new TrainingUtils singleton instance.")
                    cls._instance = TrainingUtils()
        return cls._instance

    # ------------------------------
    # Public Config Loader
    # ------------------------------
    def load_model_configs(self) -> dict:
        """
        Reads model-related configs from AppConfigs singleton and returns a dict:
            {
                "models_output_dir": <Path>,
                "model_name": <str>,
                "model_version": <str>,
                "model_output_path": <Path>
            }
        """
        app_configs = AppConfigs.get_instance()

        models_output_dir = self._resolve_models_output_dir(app_configs)
        model_name = self._resolve_model_name(app_configs)
        model_version = self._resolve_model_version(app_configs)

        model_output_path = models_output_dir / model_name / model_version
        model_output_path.mkdir(parents=True, exist_ok=True)

        result = {
            "models_output_dir": str(models_output_dir),
            "model_name": model_name,
            "model_version": model_version,
            "model_output_path": str(model_output_path.resolve()),
        }

        LOGGER.info(f"Resolved training configs: {result}")
        return result

    # ------------------------------
    # Private Helpers
    # ------------------------------
    def _resolve_models_output_dir(self, app_configs: AppConfigs) -> Path:
        """
        MODELS_OUTPUT_DIR:
        - If absolute, use as-is.
        - If relative, resolve relative to current working directory.
        - Default: ./outputs/model_outputs
        """
        dir_value = app_configs.get_str("MODELS_OUTPUT_DIR")

        if dir_value:
            dir_path = Path(dir_value)
            if not dir_path.is_absolute():
                dir_path = Path.cwd() / dir_path
        else:
            dir_path = Path.cwd() / "outputs" / "model_outputs"

        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path.resolve()

    def _resolve_model_name(self, app_configs: AppConfigs) -> str:
        """
        MODEL_NAME_DIR default: 'customer_support_distiled'
        """
        return app_configs.get_str("MODEL_NAME_DIR", "customer_support_distiled")

    def _resolve_model_version(self, app_configs: AppConfigs) -> str:
        """
        MODEL_VERSION default: timestamp-based like 0.0.YYYYMMDD_HHMMSS
        """
        version = app_configs.get_str("MODEL_VERSION")
        if version:
            return version
        return f"0.0.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
