from pathlib import Path
from typing import Optional, Dict
import datetime


class TrainingUtils:

    @staticmethod
    def create_output_dirs(req_dto, resp_dto):
        pass


# New utility function to resolve/create model output directory and version
def resolve_model_output_config(project_root: Optional[str] = None,
                                env_path: Optional[str] = None,
                                model_name_dir_default: str = "customer_support_distilled") -> Dict[str, str]:
    """
    Resolve configuration for model outputs from a .env file (project root by default), apply defaults,
    ensure the model output directory exists, and create a MODEL_VERSION if missing (persisted to .env).

    Returns a dict with keys:
      - models_output_dir: absolute path to the parent models output dir
      - model_name_dir: the name of the model directory (not full path)
      - model_dir: absolute path to the model directory (models_output_dir / model_name_dir)
      - model_version: resolved or newly created version string

    Behavior and defaults:
      - Reads .env in project root (or `env_path` if provided). Lines starting with '#' or '//' are ignored.
      - MODELS_OUTPUT_DIR: if present and absolute -> used as-is; if present and relative -> joined to project_root.
        If missing, defaults to project_root/outputs/model_outputs
      - MODEL_NAME_DIR: default is `customer_support_distiled` if not present in .env
      - MODEL_VERSION: if missing a timestamp-based version like '0.0.1+YYYYMMDDHHMMSS' is generated and appended
        to the .env file so future runs will reuse it.
    """

    # determine project root (two levels up from this file by repo layout)
    if project_root:
        root = Path(project_root).expanduser().resolve()
    else:
        # training/utils/<this file> -> parents[2] gives project root
        root = Path(__file__).resolve().parents[2]

    # env file path
    env_file = Path(env_path).expanduser().resolve() if env_path else (root / ".env")

    # simple .env parser
    env = {}
    if env_file.exists():
        try:
            with env_file.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    env[key] = val
        except Exception:
            # if reading fails, proceed with empty env (we'll surface errors on write/dir creation)
            env = {}

    # MODELS_OUTPUT_DIR
    mod_out_value = env.get("MODELS_OUTPUT_DIR")
    if mod_out_value:
        mod_out_path = Path(mod_out_value)
        if not mod_out_path.is_absolute():
            mod_out_path = (root / mod_out_path).resolve()
    else:
        mod_out_path = (root / "outputs" / "model_outputs").resolve()

    # MODEL_NAME_DIR
    model_name_dir = env.get("MODEL_NAME_DIR", model_name_dir_default)

    # final model directory
    model_dir = (mod_out_path / model_name_dir).resolve()

    # ensure dirs exist
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create model directory '{model_dir}': {e}")

    # MODEL_VERSION
    model_version = env.get("MODEL_VERSION")
    if not model_version:
        # generate a timezone-aware UTC timestamped version and persist to .env
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        model_version = f"0.0.1+{ts}"
        try:
            # append to .env (create file if it doesn't exist)
            with env_file.open("a", encoding="utf-8") as f:
                if env_file.stat().st_size > 0:
                    f.write("\n")
                f.write(f"MODEL_VERSION={model_version}\n")
        except Exception as e:
            # If we cannot persist, still return the generated version but surface a warning via exception
            raise RuntimeError(f"Could not persist MODEL_VERSION to {env_file}: {e}")

    return {
        "models_output_dir": str(mod_out_path),
        "model_name_dir": str(model_name_dir),
        "model_dir": str(model_dir),
        "model_version": str(model_version),
    }
