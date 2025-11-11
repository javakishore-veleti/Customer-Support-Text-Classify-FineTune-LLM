from training.dtos import TrainingReqDTO, TrainingResDTO


class TrainingPipelineWf:
    def __init__(self):
        self.wf_tasks = []

    def run(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO):
        pass


