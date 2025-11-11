from app_common.app_configs_util import AppConfigs
from training.dtos import TrainingReqDTO, TrainingResDTO
import logging
from training.training_pipeline_loader import TrainingPipelineLoader
from training.wf_tasks.interfaces import TrainingPipelineTask
from app_common.app_constants import TrainingConstants

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

    loader = TrainingPipelineLoader("../training_pipeline_wfs.json")
    training_pipeline_name = configs.get_str(TrainingConstants.KEY_TRAINING_PIPELINE_NAME, TrainingConstants.DEFAULT_TRAINING_PIPELINE_NAME)
    tasks = loader.load_pipeline(training_pipeline_name)

    req_dto: TrainingReqDTO = TrainingReqDTO()
    res_dto: TrainingResDTO = TrainingResDTO()

    for task in tasks:
        task_obj: TrainingPipelineTask = task
        LOGGER.info(f"STARTED Running task: {task.name()}")

        task_obj.execute(req_dto=req_dto, res_dto=res_dto)

        LOGGER.info(f"COMPLETED Running task: {task.name()}")

    # LOGGER.info("Loading TrainingConfigs started...")
    # training_utils = TrainingUtils.get_instance()
    # training_utils.load_model_configs()

    # LOGGER.info("Training started...")
    # Add training logic here

    # TrainingPipelineImpl().run(req_dto, res_dto)

    LOGGER.info("Training completed.")


if __name__ == "__main__":
    main()
