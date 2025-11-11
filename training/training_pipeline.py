from overrides import override
from training.interfaces import TrainingPipelineWf
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_configs_util import AppConfigs
from training.dtos import TrainingReqDTO, TrainingResDTO
from training.training_pipeline_loader import TrainingPipelineLoader
from training.wf_tasks.interfaces import TrainingPipelineTask
from app_common.app_constants import TrainingConstants
import logging

LOGGER = logging.getLogger(__name__)


class TrainingPipelineImpl(TrainingPipelineWf):
    def __init__(self):
        pass

    @override
    def run(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO):
        LOGGER.info("STARTED Training Pipeline")

        configs = AppConfigs.get_instance()
        wfs_json_path: str = configs.get_str(TrainingConstants.KEY_TRAINING_PIPELINE_WORKFLOWS_JSON_PATH,
                                             "../aws_data_fine_tuning_pipeline_wfs.json")

        loader = TrainingPipelineLoader(wfs_json_path)
        training_pipeline_name = configs.get_str(TrainingConstants.KEY_TRAINING_PIPELINE_NAME,
                                                 TrainingConstants.DEFAULT_TRAINING_PIPELINE_NAME)
        tasks = loader.load_pipeline(training_pipeline_name)

        req_dto: TrainingReqDTO = TrainingReqDTO()
        res_dto: TrainingResDTO = TrainingResDTO()

        for task in tasks:
            task_obj: TrainingPipelineTask = task
            LOGGER.info(f"STARTED Running task: {task.name()}")

            task_obj.execute(req_dto=req_dto, res_dto=res_dto)

            LOGGER.info(f"COMPLETED Running task: {task.name()}")

        # Load data

        # Train the model

        # Evaluate the model

        LOGGER.info("COMPLETED Training Pipeline")

        return
