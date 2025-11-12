import importlib
import logging
from typing import Dict, Type
from training.wf_tasks.interfaces import TrainingPipelineTask

LOGGER = logging.getLogger(__name__)


class TaskRegistry:
    """Singleton registry for all registered workflow tasks."""

    _instance = None
    _tasks: Dict[str, Type[TrainingPipelineTask]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, name: str, cls_ref: Type[TrainingPipelineTask]):
        """Register a task class explicitly."""
        if name in self._tasks:
            LOGGER.warning(f"Task '{name}' already registered, overriding.")
        self._tasks[name] = cls_ref
        LOGGER.debug(f"Task '{name}' registered successfully.")

    def get(self, name: str) -> Type[TrainingPipelineTask]:
        """Retrieve a registered task class by name."""
        if name not in self._tasks:
            raise KeyError(f"Task '{name}' not found in registry.")
        return self._tasks[name]

    def load_dynamic(self, class_path: str):
        """Dynamically load a class from its module path."""
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls_ref = getattr(module, class_name)

        LOGGER.info(f"module_name '{module_name}' module {module}")

        if not issubclass(cls_ref, TrainingPipelineTask):
            raise TypeError(f"{class_name} must inherit from TrainingPipelineTask.")
        task_instance = cls_ref()
        self.register(task_instance.name(), cls_ref)
        LOGGER.info(f"Dynamically loaded and registered task: {class_name}")
        return cls_ref

    def all_tasks(self):
        return list(self._tasks.keys())

    # ✅ Added helper methods for loader compatibility
    @classmethod
    def get_instance(cls):
        """Global accessor for the singleton instance."""
        if cls._instance is None:
            cls._instance = TaskRegistry()
        return cls._instance

    @classmethod
    def has_task(cls, name: str) -> bool:
        """Check if a task name is already registered."""
        return name in cls.get_instance()._tasks

    @classmethod
    def get_task_class(cls, name: str) -> Type[TrainingPipelineTask]:
        """Retrieve a task class for the given name (class-level access)."""
        return cls.get_instance().get(name)
