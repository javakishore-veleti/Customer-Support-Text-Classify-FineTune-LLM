from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
from pathlib import Path
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)


class TrainingDataExcelsDataLoaderTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "training_data_excels_data_loader"

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto:TrainingReqDTO, res_dto:TrainingResDTO) -> int:

        self.read_all_file_names_using_configs(req_dto)
        LOGGER.info(f"Excel File Path set to: {req_dto.training_data_excel_filePath}")
        LOGGER.info(f"Excel File Names set to: {req_dto.training_data_excel_file_names}")

        dataframes = self.load_all_excel_sheets(req_dto.training_data_excel_filePath, req_dto.training_data_excel_file_names)
        req_dto.training_data_dataframes = dataframes

        LOGGER.info(f"COMPLETED Loading Excel Files as DataFrames. Total files loaded: {len(dataframes)}")
        LOGGER.info(f"Excel File Names set to: {req_dto.training_data_excel_file_names}")

        return WfResponses.SUCCESS

    # noinspection PyMethodMayBeStatic
    def read_all_file_names_using_configs(self, req_dto):
        excel_file_dir_path = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_EXCEL_FILE_PATH", "z_datasets/training_datasets/aws_services_questions"
        )
        excel_files_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_EXCEL_FILE_NAMES_CSV", "ALL_FILES"
        ).strip()
        if excel_files_csv != "ALL_FILES":
            excel_file_names = excel_files_csv.strip().split(",")
        else:
            LOGGER.info(f"Loading all Excel files from the specified directory {excel_file_dir_path}")
            folder = Path(excel_file_dir_path)
            excel_file_names = [file.name for file in folder.glob("*.xlsx") if file.is_file()]
            # Placeholder for all files logic
        req_dto.training_data_excel_filePath = excel_file_dir_path
        req_dto.training_data_excel_file_names = excel_file_names

    # ---------------------------------------------------------------------
    # STEP 2 — Load all Excel sheets into pandas DataFrames
    # ---------------------------------------------------------------------
    def load_all_excel_sheets(self, folder_path: str, file_names: list):
        """
        Loads every worksheet in each Excel file into DataFrames.

        Returns:
            List of dicts like:
            [
                {
                    "file_name": "aws_service_training_dataset.xlsx",
                    "sheets": {
                        "Amazon_S3": <DataFrame>,
                        "Amazon_EC2": <DataFrame>,
                        ...
                    }
                },
                ...
            ]
        """
        folder = Path(folder_path)
        dataframes = []

        for file_name in file_names:
            file_path = folder / file_name
            try:
                # Load all worksheets in one go
                sheets_dict = pd.read_excel(file_path, sheet_name=None)

                dataframes.append({
                    "file_name": file_name,
                    "sheets": sheets_dict
                })

                LOGGER.info(
                    f"Loaded {file_name} "
                    f"with {len(sheets_dict)} worksheets: {list(sheets_dict.keys())}"
                )
            except Exception as e:
                LOGGER.error(f"Failed to read Excel file {file_name}: {e}", exc_info=True)

        return dataframes
