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
    Tokenizes all configured text columns (from .env) for every worksheet.

    Configuration (.env):
        TRAINING_MODEL_NAME=distilbert-base-uncased
        TRAINING_MODEL_MAX_SEQ_LEN=128
        DATASET_EXCEL_COLUMN_NAMES="category,category_serial_number,sample_question,sample_question_type"
        TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV="category"
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

        model_name = AppConfigs.get_instance().get_str(
            "TRAINING_MODEL_NAME", "distilbert-base-uncased"
        )
        max_seq_len = AppConfigs.get_instance().get_int(
            "TRAINING_MODEL_MAX_SEQ_LEN", 128
        )
        dataset_cols_csv = AppConfigs.get_instance().get_str(
            "DATASET_EXCEL_COLUMN_NAMES",
            "category,category_serial_number,sample_question,sample_question_type",
        )
        dataset_cols = [c.strip() for c in dataset_cols_csv.split(",") if c.strip()]

        label_cols_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV", "category"
        )
        label_cols = [c.strip() for c in label_cols_csv.split(",") if c.strip()]

        LOGGER.info(f"Tokenizer Model: {model_name}")
        LOGGER.info(f"Max Sequence Length: {max_seq_len}")
        LOGGER.info(f"Configured Columns to Tokenize: {dataset_cols}")
        LOGGER.info(f"Label Columns: {label_cols}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        req_dto.tokenizer = tokenizer

        def tokenize_fn(examples, text_col_name):
            texts = examples[text_col_name]
            if not isinstance(texts, list):
                texts = [str(texts)]
            else:
                texts = ["" if v is None else str(v) for v in texts]
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_len,
            )

        for workbook in req_dto.training_data_dataframes:
            excel_file_name = workbook.get("file_name")
            sheets_dict = workbook.get("sheets", {})

            for sheet_name, sheet_data in sheets_dict.items():
                if not isinstance(sheet_data, dict) or "train_df" not in sheet_data:
                    LOGGER.warning(f"Skipping sheet '{sheet_name}' — no train/test split present.")
                    continue

                train_df = sheet_data["train_df"]
                test_df = sheet_data["test_df"]

                tokenized_datasets = {}

                for col_name in dataset_cols:
                    if col_name not in train_df.columns:
                        LOGGER.debug(f"Skipping column '{col_name}' — not found in {excel_file_name}->{sheet_name}")
                        continue

                    # If column is entirely numeric or NaN, skip it
                    if train_df[col_name].dropna().apply(lambda x: isinstance(x, (int, float))).all():
                        LOGGER.warning(f"Skipping column '{col_name}' — purely numeric, not suitable for tokenization.")
                        continue

                    LOGGER.info(f"Tokenizing column '{col_name}' in {excel_file_name}->{sheet_name}")

                    selected_cols = list({col_name, *[c for c in label_cols if c in train_df.columns]})
                    LOGGER.debug(f"Selected columns for tokenization → {selected_cols}")

                    # Convert non-string types to strings
                    for c in selected_cols:
                        train_df[c] = train_df[c].astype(str)
                        test_df[c] = test_df[c].astype(str)

                    try:
                        train_ds = Dataset.from_pandas(train_df[selected_cols])
                        test_ds = Dataset.from_pandas(test_df[selected_cols])
                    except Exception as e:
                        LOGGER.exception(f"Failed converting to Dataset for {col_name} in {sheet_name}: {e}")
                        continue

                    train_tok = train_ds.map(
                        lambda x: tokenize_fn(x, col_name),
                        batched=True,
                        remove_columns=[col_name],
                    )
                    test_tok = test_ds.map(
                        lambda x: tokenize_fn(x, col_name),
                        batched=True,
                        remove_columns=[col_name],
                    )

                    keep_cols = ["input_ids", "attention_mask"] + label_cols
                    train_tok.set_format(type="torch", columns=[c for c in keep_cols if c in train_tok.column_names])
                    test_tok.set_format(type="torch", columns=[c for c in keep_cols if c in test_tok.column_names])

                    tokenized_datasets[col_name] = {
                        "tokenized_train_dataset": train_tok,
                        "tokenized_test_dataset": test_tok,
                        "train_sample_count": len(train_tok),
                        "test_sample_count": len(test_tok),
                    }

                    LOGGER.info(
                        f"Completed tokenizing column '{col_name}' in sheet '{sheet_name}' "
                        f"(Train={len(train_tok)}, Test={len(test_tok)})"
                    )

                sheet_data["tokenizer_model_name"] = model_name
                sheet_data["max_seq_len"] = max_seq_len
                sheet_data["tokenized_datasets"] = tokenized_datasets

            LOGGER.info(f"Completed tokenization for workbook: {excel_file_name}")

        LOGGER.info("COMPLETED TrainingTokenizerTask execution.")
        return WfResponses.SUCCESS

