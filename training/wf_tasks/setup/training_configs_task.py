from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses


class TrainingConfigsTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto:TrainingReqDTO, res_dto:TrainingResDTO) -> int:
        from training.utils.training_utils import TrainingUtils

        training_utils = TrainingUtils.get_instance()
        training_utils.load_model_configs()

        return WfResponses.SUCCESS