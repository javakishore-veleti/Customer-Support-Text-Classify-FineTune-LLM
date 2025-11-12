from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
import logging

LOGGER = logging.getLogger(__name__)


class TrainingDataLabelsMapperTask(TrainingPipelineTask):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """Unique task name identifier."""
        return "training_data_labels_mapper"

    @overrides
    # noinspection PyMethodMayBeStatic
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingDataLabelsMapperTask execution.")

        excels_worksheets_as_dfs_list = req_dto.training_data_dataframes

        label2ids_col_names_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_CLASSIFICATION_COLUMN_NAMES_CSV", "category")
        label2ids_col_names = [col_name.strip() for col_name in label2ids_col_names_csv.split(",")]

        # Example excel_sheets_columns_mappings_dict = {"excel_file_name": {"sheet_name": {"a_label2id_col_name": {}}}}
        excel_sheets_columns_mappings_dict = {}
        req_dto.training_data_labels_mapping = excel_sheets_columns_mappings_dict

        for an_excel_worksheets_info in excels_worksheets_as_dfs_list:
            excel_file_name = an_excel_worksheets_info.get("file_name")
            sheets_dict: dict = an_excel_worksheets_info.get("sheets")

            # Example {"a_label2id_col_name": {}}}
            temp = {}
            excel_sheets_columns_mappings_dict.update({excel_file_name: temp})

            for sheet_name, sheet_as_df in sheets_dict.items():
                LOGGER.info(f"Categories 2 Ids and Ids 2 Categories for file: {excel_file_name}, sheet: {sheet_name}")

                # Example temp2 = {"a_label2id_col_name": {}}
                temp2 = {}
                temp.update({sheet_name: temp2})

                for a_label2id_col_name in label2ids_col_names:
                    if a_label2id_col_name not in sheet_as_df.columns:
                        LOGGER.warning(f"Column '{a_label2id_col_name}' not found in sheet '{sheet_name}' of file '{excel_file_name}'. Skipping.")
                        continue

                    categories = sheet_as_df[a_label2id_col_name].unique().tolist()
                    LOGGER.info(f"Unique Categories for file: {excel_file_name}, sheet: {sheet_name} are {categories}")

                    label2ids_map = {category: idx for idx, category in enumerate(categories)}
                    ids2label_map = {idx: category for category, idx in label2ids_map.items()}

                    temp2.update({a_label2id_col_name: {
                        "column_name": a_label2id_col_name,
                        "label2ids_map": label2ids_map,
                        "ids2label_map": ids2label_map
                    }})
                # LOGGER.info(f"sheet_name {sheet_name} temp2 mapping: {temp2}")

        for excel_file_name, sheets_mapping in excel_sheets_columns_mappings_dict.items():
            LOGGER.info(f"Excel File: {excel_file_name}")
            for sheet_name, cols_mapping in sheets_mapping.items():
                LOGGER.info(f"  Sheet Name: {sheet_name}")
                for col_name, mappings in cols_mapping.items():
                    LOGGER.info(f"    Column Name: {col_name}")
                    LOGGER.info(f"      Label2Ids Map: {mappings['label2ids_map']}")
                    LOGGER.info(f"      Ids2Label Map: {mappings['ids2label_map']}")

        LOGGER.info("COMPLETED TrainingDataLabelsMapperTask execution.")
        return WfResponses.SUCCESS
