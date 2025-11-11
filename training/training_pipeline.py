from overrides import override

from training.interfaces import TrainingPipelineWf
from training.dtos import TrainingReqDTO, TrainingResDTO


class TrainingPipelineImpl(TrainingPipelineWf):
    def __init__(self):
        pass

    @override
    def run(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO):
        # Load data

        # Train the model

        # Evaluate the model

        return
