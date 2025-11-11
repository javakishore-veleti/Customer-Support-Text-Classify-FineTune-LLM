import json
import logging
from training.core.task_registry import TaskRegistry

LOGGER = logging.getLogger(__name__)


class TrainingPipelineLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.registry = TaskRegistry()

    def load_pipeline(self, workflow_name: str = "CustomerSupportFineTuningTrainingPipeline"):
        with open(self.json_path, "r") as f:
            config = json.load(f)

        workflow = next((wf for wf in config["workflows"] if wf["name"] == workflow_name), None)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found in JSON.")

        loaded_tasks = []
        for task in workflow["tasks"]:
            try:
                task_cls = self.registry.get(task["task_name"])
                LOGGER.debug(f"Found registered task: {task['task_name']}")
            except KeyError:
                LOGGER.info(f"Task '{task['task_name']}' not found, loading dynamically.")
                task_cls = self.registry.load_dynamic(task["class_path"])
            loaded_tasks.append(task_cls())

        LOGGER.info(f"Loaded workflow '{workflow_name}' with {len(loaded_tasks)} tasks.")
        return loaded_tasks
