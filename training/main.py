from app_common.app_configs_util import AppConfigs
from training.dtos import TrainingReqDTO, TrainingResDTO
from training.training_pipeline import TrainingPipelineImpl
import logging

from training.utils.training_utils import TrainingUtils

# Configure the root logger or specific loggers
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('app.log')])

LOGGER = logging.getLogger(__name__)


def main():
    LOGGER.info("Loading AppConfigs started...")
    configs = AppConfigs.get_instance()
    configs.load_app_configs()

    LOGGER.info("Loading TrainingConfigs started...")
    training_utils = TrainingUtils.get_instance()
    training_utils.load_model_configs()

    LOGGER.info("Training started...")
    # Add training logic here

    req_dto: TrainingReqDTO = TrainingReqDTO()
    res_dto: TrainingResDTO = TrainingResDTO()

    TrainingPipelineImpl().run(req_dto, res_dto)

    LOGGER.info("Training completed.")


if __name__ == "__main__":
    main()
