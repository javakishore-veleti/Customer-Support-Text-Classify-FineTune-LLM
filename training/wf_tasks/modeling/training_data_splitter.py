from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


class TrainingDataSplitterTask(TrainingPipelineTask):
    """
    Splits each worksheet's DataFrame into train/test subsets using configurable ratios.

    Configuration (.env):
        TRAINING_DATASET_TRAIN_RATIO=80
        TRAINING_DATASET_TEST_RATIO=20
        TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV=category,text
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_data_splitter"

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingDataSplitterTask execution.")

        excels_worksheets_as_dfs_list = req_dto.training_data_dataframes
        if not excels_worksheets_as_dfs_list:
            LOGGER.error("❌ No training_data_dataframes found in req_dto.")
            return WfResponses.FAILURE

        # --- Configuration driven ratios ---
        train_ratio = AppConfigs.get_instance().get_int("TRAINING_DATASET_TRAIN_RATIO", 80)
        test_ratio = AppConfigs.get_instance().get_int("TRAINING_DATASET_TEST_RATIO", 20)

        total_ratio = train_ratio + test_ratio
        if total_ratio != 100:
            LOGGER.warning(f"⚠️ TRAINING_DATASET_TRAIN_RATIO + TRAINING_DATASET_TEST_RATIO != 100 (got {total_ratio}). Normalizing ratios automatically.")
            train_ratio = (train_ratio / total_ratio) * 100
            test_ratio = (test_ratio / total_ratio) * 100

        train_fraction = train_ratio / 100.0
        test_fraction = test_ratio / 100.0

        LOGGER.info(f"Configured Train/Test Split → Train: {train_ratio}%, Test: {test_ratio}%")

        # --- Column(s) to use for label mapping consistency ---
        label2ids_col_names_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV", "category"
        )
        label2ids_col_names = [col.strip() for col in label2ids_col_names_csv.split(",")]

        # --- Process each workbook/sheet ---
        for workbook in excels_worksheets_as_dfs_list:
            excel_file_name = workbook.get("file_name")
            sheets_dict = workbook.get("sheets", {})

            for sheet_name, sheet_df in sheets_dict.items():
                if sheet_df.empty:
                    LOGGER.warning(f"Skipping empty sheet '{sheet_name}' in file '{excel_file_name}'.")
                    continue

                LOGGER.info(f"Splitting file: {excel_file_name}, sheet: {sheet_name}")

                for label_col in label2ids_col_names:
                    if label_col not in sheet_df.columns:
                        LOGGER.warning(f"⚠️ Column '{label_col}' not found in sheet '{sheet_name}' → skipping.")
                        continue

                    df = sheet_df.copy()
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

                    # --- Stratified split per category if possible ---
                    train_splits, test_splits = [], []
                    unique_labels = df[label_col].unique().tolist()

                    for label_value in unique_labels:
                        label_subset = df[df[label_col] == label_value]
                        n_train = max(1, int(len(label_subset) * train_fraction))
                        n_test = len(label_subset) - n_train

                        train_part = label_subset.iloc[:n_train]
                        test_part = label_subset.iloc[-n_test:] if n_test > 0 else pd.DataFrame(columns=df.columns)

                        train_splits.append(train_part)
                        test_splits.append(test_part)

                    train_df = pd.concat(train_splits, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
                    test_df = pd.concat(test_splits, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

                    # --- Store back in req_dto structure ---
                    split_key = f"{label_col}_split"
                    sheets_dict[sheet_name] = {
                        "data": sheet_df,
                        "train_df": train_df,
                        "test_df": test_df,
                        "train_count": len(train_df),
                        "test_count": len(test_df),
                        "split_ratio": {"train": train_ratio, "test": test_ratio},
                    }

                    LOGGER.info(
                        f"Split {sheet_name} by '{label_col}' → "
                        f"Train={len(train_df)}, Test={len(test_df)} (Train {train_ratio}%, Test {test_ratio}%)"
                    )

        LOGGER.info("COMPLETED TrainingDataSplitterTask execution.")
        return WfResponses.SUCCESS
