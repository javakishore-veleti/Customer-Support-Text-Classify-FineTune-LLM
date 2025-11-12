from app_common.app_configs_util import AppConfigs
from training.training_pipeline import TrainingPipelineImpl
from training.dtos import TrainingReqDTO, TrainingResDTO
import logging
from dotenv import load_dotenv
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load .env from repo root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure the root logger or specific loggers
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('../app.log')])

LOGGER = logging.getLogger(__name__)


def main():
    LOGGER.info("Loading AppConfigs started...")
    configs = AppConfigs.get_instance()
    configs.load_app_configs()
    LOGGER.info("AppConfigs loaded successfully.")

    req_dto = TrainingReqDTO()
    res_dto = TrainingResDTO()

    TrainingPipelineImpl().run(req_dto=req_dto, res_dto=res_dto)


if __name__ == "__main__":
    main()
