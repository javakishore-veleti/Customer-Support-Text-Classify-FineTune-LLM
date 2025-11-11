from app_common.app_constants import WfResponses
from training.dtos import TrainingReqDTO, TrainingResDTO

from abc import ABC, abstractmethod


class TrainingPipelineTask(ABC):
    """Base interface for all workflow tasks."""

    def __init__(self):
        pass

    @abstractmethod
    def name(self) -> str:
        """Unique task name identifier."""
        pass

    @abstractmethod
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        return WfResponses.SUCCESS
