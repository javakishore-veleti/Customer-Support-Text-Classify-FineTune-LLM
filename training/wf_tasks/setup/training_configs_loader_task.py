from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
import logging

LOGGER = logging.getLogger(__name__)


class TrainingConfigsLoaderTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "training_configs_loader"

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto:TrainingReqDTO, res_dto:TrainingResDTO) -> int:

        if not AppConfigs.get_instance().get_bool("TRAINING_ENABLE_CLASSIFICATION", False):
            LOGGER.info("Classification DISABLED in .env — skipping.")
            req_dto.training_clustering_enabled = False
            return WfResponses.SUCCESS

        from training.utils.training_utils import TrainingUtils

        training_utils = TrainingUtils.get_instance()
        training_utils.load_model_configs()

        return WfResponses.SUCCESS
