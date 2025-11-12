import importlib
import logging
from training.core.task_registry import TaskRegistry

LOGGER = logging.getLogger(__name__)


class TrainingPipelineLoader:
    def __init__(self, wfs_json_path: str):
        from training.core.workflow_json_parser import WorkflowJsonParser
        self.parser = WorkflowJsonParser(wfs_json_path)

    def load_pipeline(self, pipeline_name: str, include_all_metadata: bool = False):
        """
        Loads all tasks for the specified pipeline.

        Args:
            pipeline_name (str): Name of the workflow.
            include_all_metadata (bool): If True, attaches each task's JSON metadata to the instance.

        Returns:
            List of TrainingPipelineTask objects.
        """
        workflow = self.parser.get_workflow_by_name(pipeline_name)
        if not workflow:
            LOGGER.error(f"Workflow '{pipeline_name}' not found in JSON.")
            return []

        registry = TaskRegistry.get_instance()  # ✅ get the singleton instance

        tasks = []
        for task_config in workflow.get("tasks", []):
            task_name = task_config.get("task_name")
            class_path = task_config.get("class_path")

            # Load dynamically if not registered yet
            if not registry.has_task(task_name):
                LOGGER.info(f"Task '{task_name}' not found, loading dynamically.")
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                clazz = getattr(module, class_name)
                registry.register(task_name, clazz)  # call via instance
                LOGGER.debug(f"Task '{task_name}' registered successfully.")

            task_class = registry.get_task_class(task_name)
            if not task_class:
                LOGGER.warning(f"Skipping unregistered task: {task_name}")
                continue

            task_instance = task_class()
            if include_all_metadata:
                task_instance.metadata = task_config  # Attach JSON metadata

            tasks.append(task_instance)

        LOGGER.info(f"Loaded workflow '{pipeline_name}' with {len(tasks)} tasks.")
        return tasks
