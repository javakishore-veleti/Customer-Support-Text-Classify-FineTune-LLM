from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
from pathlib import Path
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)


def normalize_sheet_name(name: str) -> str:
    """Normalize sheet name for consistent matching across AWS/Amazon variants."""
    name = name.strip().lower()
    # Normalize AWS_ and Amazon_ prefixes to a common base
    name = name.replace("aws_", "").replace("amazon_", "")
    return name

class TrainingDataExcelsDataLoaderTask(TrainingPipelineTask):
    """
    Loads Excel training datasets from configured directories,
    skipping any worksheets listed in DATASET_WORKSHEETS_TO_SKIP_CSV.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "training_data_excels_data_loader"

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingDataExcelsDataLoaderTask")

        # Step 1: Resolve Excel file paths and names
        self.read_all_file_names_using_configs(req_dto)
        LOGGER.info(f"Excel File Path: {req_dto.training_data_excel_filePath}")
        LOGGER.info(f"Excel File Names: {req_dto.training_data_excel_file_names}")

        # Step 2: Load Excel sheets into DataFrames
        dataframes = self.load_all_excel_sheets(
            req_dto.training_data_excel_filePath,
            req_dto.training_data_excel_file_names
        )
        req_dto.training_data_dataframes = dataframes

        LOGGER.info(f"COMPLETED Excel Loading. Total files loaded: {len(dataframes)}")
        return WfResponses.SUCCESS

    # ---------------------------------------------------------------------
    # STEP 1 — Read file names from config
    # ---------------------------------------------------------------------
    def read_all_file_names_using_configs(self, req_dto):
        """Reads dataset file names from config directory or CSV list."""
        excel_file_dir_path = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_EXCEL_FILE_PATH",
            "z_datasets/training_datasets/aws_services_questions"
        )
        excel_files_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_EXCEL_FILE_NAMES_CSV",
            "ALL_FILES"
        ).strip()

        if excel_files_csv != "ALL_FILES":
            excel_file_names = [f.strip() for f in excel_files_csv.split(",") if f.strip()]
        else:
            LOGGER.info(f"📂 Loading all Excel files from: {excel_file_dir_path}")
            folder = Path(excel_file_dir_path)
            excel_file_names = [file.name for file in folder.glob("*.xlsx") if file.is_file()]

        req_dto.training_data_excel_filePath = excel_file_dir_path
        req_dto.training_data_excel_file_names = excel_file_names



    # ---------------------------------------------------------------------
    # STEP 2 — Load all Excel sheets (skip-based filtering only)
    # ---------------------------------------------------------------------
    def load_all_excel_sheets(self, folder_path: str, file_names: list):
        folder = Path(folder_path)
        dataframes = []

        skip_csv = AppConfigs.get_instance().get_str("DATASET_WORKSHEETS_TO_SKIP_CSV", "")
        if len(skip_csv) > 0:
            LOGGER.info(f"Sheets to skip (raw): {skip_csv}")
            skip_sheets = [normalize_sheet_name(name=s) for s in skip_csv.split(",") if s.strip()]
        else:
            skip_sheets = []

        LOGGER.info(f"skip_csv {skip_csv} Sheets to skip (normalized): {skip_sheets or ['(none)']}")

        for file_name in file_names:
            file_path = folder / file_name
            try:
                sheets_dict = pd.read_excel(file_path, sheet_name=None)
                filtered_sheets, skipped = {}, []

                for sheet_name, df in sheets_dict.items():
                    if normalize_sheet_name(sheet_name) in skip_sheets:
                        skipped.append(sheet_name)
                        continue
                    filtered_sheets[sheet_name] = df

                if skipped:
                    LOGGER.warning(f"Skipped sheets in {file_name}: {skipped}")

                dataframes.append({
                    "file_name": file_name,
                    "sheets": filtered_sheets
                })

                LOGGER.info(
                    f"Loaded {file_name} with {len(filtered_sheets)} worksheet(s): {list(filtered_sheets.keys())}"
                )

            except Exception as e:
                LOGGER.error(f"Failed to read Excel file {file_name}: {e}", exc_info=True)

        total_sheets = sum(len(d['sheets']) for d in dataframes)
        LOGGER.info(f"COMPLETED Excel load — total worksheets kept: {total_sheets}")

        # Final summary
        kept = [s for d in dataframes for s in d['sheets'].keys()]
        LOGGER.info(f"Final kept worksheets: {kept if kept else 'NONE — all skipped'}")

        return dataframes
