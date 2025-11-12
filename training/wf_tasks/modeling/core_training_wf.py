from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.training_pipeline_loader import TrainingPipelineLoader
from app_common.app_configs_util import AppConfigs
from app_common.app_constants import TrainingConstants, WfResponses
from training.dtos import TrainingReqDTO, TrainingResDTO
import logging

LOGGER = logging.getLogger(__name__)


class CoreTrainingWF(TrainingPipelineTask):
    """
    CoreTrainingWF acts as a meta-task that orchestrates
    all training & modeling related tasks that belong to it.

    It queries the workflow JSON for tasks where:
       "is_belongs_to_task": true
       and "belongs_to_list": includes this task name.
    """

    def __init__(self):
        super().__init__()

    def name(self):
        return "core_training_wf"

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED CoreTrainingWF execution")

        configs = AppConfigs.get_instance()
        wfs_json_path = configs.get_str(
            TrainingConstants.KEY_TRAINING_PIPELINE_WORKFLOWS_JSON_PATH,
            "../aws_data_fine_tuning_pipeline_wfs.json"
        )
        pipeline_name = configs.get_str(
            TrainingConstants.KEY_TRAINING_PIPELINE_NAME,
            TrainingConstants.DEFAULT_TRAINING_PIPELINE_NAME
        )

        # Load workflow JSON
        loader = TrainingPipelineLoader(wfs_json_path)
        all_tasks = loader.load_pipeline(pipeline_name, include_all_metadata=True)

        # Filter child tasks belonging to this meta-task
        my_children = [
            t for t in all_tasks
            if getattr(t, "metadata", {}).get("is_belongs_to_task")
            and "core_training_wf" in getattr(t, "metadata", {}).get("belongs_to_list", [])
        ]

        if not my_children:
            LOGGER.warning("No child tasks found for CoreTrainingWF.")
            return WfResponses.SKIPPED

        LOGGER.info(f"Found {len(my_children)} sub-tasks under CoreTrainingWF.")

        # Run children sequentially (or parallel in future)
        for task in my_children:
            LOGGER.info(f"STARTED Sub-task: {task.name()}")
            result = task.execute(req_dto=req_dto, res_dto=res_dto)
            if result == WfResponses.FAILURE:
                LOGGER.error(f"Sub-task failed: {task.name()}. Aborting CoreTrainingWF.")
                return WfResponses.FAILURE
            LOGGER.info(f"COMPLETED Sub-task: {task.name()}")

        LOGGER.info("COMPLETED CoreTrainingWF execution.")
        return WfResponses.SUCCESS
