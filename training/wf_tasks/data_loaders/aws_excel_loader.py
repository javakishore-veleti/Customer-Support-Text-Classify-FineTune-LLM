from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses


class AWSExcelTrainingDataLoaderTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "aws_excel_training_data_loader"

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto:TrainingReqDTO, res_dto:TrainingResDTO) -> int:

        return WfResponses.SUCCESS
