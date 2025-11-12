from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
from transformers import AutoTokenizer
from datasets import Dataset
import logging

LOGGER = logging.getLogger(__name__)


class TrainingTokenizerTask(TrainingPipelineTask):
    """
    Builds a context-rich combined text field using only training-relevant columns
    (defined in DATASET_COLUMN_NAMES_FOR_TRAINING), then tokenizes each worksheet.

    Output per worksheet:
        sheet_data["tokenized_datasets"] = {
            "tokenized_train_dataset": Dataset,
            "tokenized_test_dataset": Dataset,
            "train_sample_count": int,
            "test_sample_count": int
        }
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_data_tokenizer"

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingTokenizerTask execution.")

        if not req_dto.training_data_dataframes:
            LOGGER.error("No training_data_dataframes found in req_dto.")
            return WfResponses.FAILURE

        # --- Load model & tokenizer config ---
        model_name = AppConfigs.get_instance().get_str("TRAINING_MODEL_NAME", "distilbert-base-uncased")
        max_seq_len = AppConfigs.get_instance().get_int("TRAINING_MODEL_MAX_SEQ_LEN", 128)

        # --- Column configuration ---
        excel_cols_csv = AppConfigs.get_instance().get_str(
            "DATASET_EXCEL_COLUMN_NAMES",
            "category,category_serial_number,sample_question,sample_question_type"
        )
        excel_cols = [c.strip() for c in excel_cols_csv.split(",") if c.strip()]

        train_cols_csv = AppConfigs.get_instance().get_str(
            "DATASET_COLUMN_NAMES_FOR_TRAINING",
            "category,sample_question,sample_question_type"
        )
        train_cols = [c.strip() for c in train_cols_csv.split(",") if c.strip()]

        label_cols_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_CLASSIFICATION_COLUMN_NAMES_CSV", "category"
        )
        label_cols = [c.strip() for c in label_cols_csv.split(",") if c.strip()]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        req_dto.tokenizer = tokenizer

        LOGGER.info(f"Tokenizer Model: {model_name}")
        LOGGER.info(f"Max Sequence Length: {max_seq_len}")
        LOGGER.info(f"Excel Columns: {excel_cols}")
        LOGGER.info(f"Training Columns: {train_cols}")
        LOGGER.info(f"Label Columns: {label_cols}")

        # --- Combine context text from selected training columns ---
        def build_combined_text(row):
            parts = []
            for col in train_cols:
                val = str(row.get(col, "")).strip()
                tag = f"[{col.upper()}]"
                parts.append(f"{tag} {val}")
            return " ".join(parts)

        def tokenize_fn(examples):
            return tokenizer(
                examples["__combined_text__"],
                truncation=True,
                padding="max_length",
                max_length=max_seq_len,
            )

        # --- Iterate over workbooks ---
        for workbook in req_dto.training_data_dataframes:
            excel_file_name = workbook.get("file_name")
            sheets_dict = workbook.get("sheets", {})

            for sheet_name, sheet_data in sheets_dict.items():
                if not isinstance(sheet_data, dict) or "train_df" not in sheet_data:
                    LOGGER.warning(f"Skipping sheet '{sheet_name}' — no train/test split present.")
                    continue

                train_df = sheet_data["train_df"].copy()
                test_df = sheet_data["test_df"].copy()

                # Build combined text for both splits
                train_df["__combined_text__"] = train_df.apply(build_combined_text, axis=1)
                test_df["__combined_text__"] = test_df.apply(build_combined_text, axis=1)

                # Keep only combined text + label columns
                all_cols = ["__combined_text__"] + [c for c in label_cols if c in train_df.columns]
                try:
                    train_ds = Dataset.from_pandas(train_df[all_cols])
                    test_ds = Dataset.from_pandas(test_df[all_cols])
                except Exception as e:
                    LOGGER.exception(f"Failed converting sheet '{sheet_name}' to HuggingFace Dataset: {e}")
                    continue

                # Tokenize
                train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["__combined_text__"])
                test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=["__combined_text__"])

                # Format columns
                keep_cols = ["input_ids", "attention_mask"] + label_cols
                train_tok.set_format(type="torch", columns=[c for c in keep_cols if c in train_tok.column_names])
                test_tok.set_format(type="torch", columns=[c for c in keep_cols if c in test_tok.column_names])

                # ✅ Store clean consistent dict (expected by fine-tuner)
                sheet_data["tokenizer_model_name"] = model_name
                sheet_data["max_seq_len"] = max_seq_len
                sheet_data["tokenized_datasets"] = {
                    "tokenized_train_dataset": train_tok,
                    "tokenized_test_dataset": test_tok,
                    "train_sample_count": len(train_tok),
                    "test_sample_count": len(test_tok),
                }
                sheet_data["combined_input_example"] = train_df["__combined_text__"].iloc[0][:250]

                LOGGER.info(
                    f"Tokenized sheet '{sheet_name}' using columns {train_cols} "
                    f"(Train={len(train_tok)}, Test={len(test_tok)})"
                )

            LOGGER.info(f"Completed tokenization for workbook: {excel_file_name}")

        LOGGER.info("COMPLETED TrainingTokenizerTask execution.")
        return WfResponses.SUCCESS
