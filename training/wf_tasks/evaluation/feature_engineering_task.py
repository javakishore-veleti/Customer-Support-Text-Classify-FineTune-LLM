from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses


class FeatureEngineeringTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "feature_engineering"

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto:TrainingReqDTO, res_dto:TrainingResDTO) -> int:

        return WfResponses.SUCCESS
