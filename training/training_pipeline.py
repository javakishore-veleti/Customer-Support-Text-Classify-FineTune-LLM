from overrides import override
from training.interfaces import TrainingPipelineWf
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_configs_util import AppConfigs
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

        # --- Load workflow config ---
        configs = AppConfigs.get_instance()
        wfs_json_path: str = configs.get_str(
            TrainingConstants.KEY_TRAINING_PIPELINE_WORKFLOWS_JSON_PATH,
            "../aws_data_fine_tuning_pipeline_wfs.json"
        )

        loader = TrainingPipelineLoader(wfs_json_path)
        training_pipeline_name = configs.get_str(
            TrainingConstants.KEY_TRAINING_PIPELINE_NAME,
            TrainingConstants.DEFAULT_TRAINING_PIPELINE_NAME
        )
        tasks = loader.load_pipeline(training_pipeline_name)

        executed = set()

        # --- Run top-level tasks only ---
        for task in tasks:
            task_obj: TrainingPipelineTask = task
            task_meta = getattr(task_obj, "metadata", {})

            # Skip tasks that belong to another workflow (avoid duplicates)
            if task_meta.get("is_belongs_to_task", False):
                LOGGER.debug(f"Skipping subtask '{task_obj.name()}' (belongs to {task_meta.get('belongs_to_list')})")
                continue

            task_name = task_obj.name()
            if task_name in executed:
                LOGGER.debug(f"Already executed task '{task_name}', skipping duplicate.")
                continue

            LOGGER.info(f"STARTED Running task: {task_name}")
            try:
                task_obj.execute(req_dto=req_dto, res_dto=res_dto)
                LOGGER.info(f"COMPLETED Running task: {task_name}")
                executed.add(task_name)
            except Exception as e:
                LOGGER.exception(f"Failed during execution of task '{task_name}': {e}")
                raise

        LOGGER.info("COMPLETED Training Pipeline")
        return
