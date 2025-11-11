from app_common.app_constants import WfResponses
from training.dtos import TrainingReqDTO, TrainingResDTO


class TrainingPipelineTask:
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        return WfResponses.SUCCESS
