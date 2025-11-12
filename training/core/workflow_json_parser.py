import json
import logging
from typing import Dict, Any, List
from pathlib import Path

LOGGER = logging.getLogger(__name__)

"""
   Parses workflow definition JSON files for training pipelines.
   Example structure:
   {
     "workflows": [
       {
         "name": "CustomerSupportFineTuningTrainingPipeline",
         "tasks": [
           {"task_name": "training_data_loader", "class_path": "..."},
           ...
         ]
       }
     ]
   }
   """


class WorkflowJsonParser:
    """
    Parses workflow definition JSON files for training pipelines.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.workflows = self._load_json()

    def _load_json(self) -> Dict[str, Any]:
        """Loads the workflow JSON file from disk, resolving from repo root if needed."""
        try:
            # Resolve path to handle relative locations
            path = Path(self.json_path)
            if not path.exists():
                # Try to locate relative to repo root (parent of 'training' folder)
                repo_root = Path(__file__).resolve().parent.parent.parent
                alt_path = repo_root / self.json_path.lstrip("./")
                if alt_path.exists():
                    path = alt_path
                else:
                    LOGGER.error(f"Workflow JSON not found: {path} or {alt_path}")
                    return {"workflows": []}

            with open(path, "r") as f:
                data = json.load(f)
                LOGGER.debug(f"Loaded workflow JSON: {path}")
                return data

        except Exception as e:
            LOGGER.error(f"Failed to load workflow JSON: {self.json_path} — {e}")
            return {"workflows": []}

    def get_workflows(self) -> List[Dict[str, Any]]:
        """Returns all workflows."""
        return self.workflows.get("workflows", [])

    def get_workflow_by_name(self, name: str) -> Dict[str, Any]:
        """Finds and returns the workflow matching the given name."""
        for wf in self.get_workflows():
            if wf.get("name") == name:
                return wf
        LOGGER.warning(f"Workflow '{name}' not found in {self.json_path}")
        return {}
