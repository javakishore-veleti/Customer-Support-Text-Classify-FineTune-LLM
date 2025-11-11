from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import os
import json
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


class TrainingModelFineTuneTask(TrainingPipelineTask):
    """
    Fine-tunes a Hugging Face model for each sheet/column combination
    dynamically driven by .env configuration.

    Expected Configuration (.env):

        MODEL_NAME=distilbert-base-uncased
        MODEL_NAME_DIR=customer-support-distilbert
        MODEL_VERSION=1.0.0
        MODELS_OUTPUT_DIR=../outputs/model_outputs
        TRAIN_NUM_EPOCHS=12
        TRAIN_BATCH_SIZE=2
        EVAL_BATCH_SIZE=4
        TRAIN_WARMUP_STEPS=200
        TRAIN_LEARNING_RATE=8e-5
        TRAIN_WEIGHT_DECAY=0.01
        TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV=category
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_model_fine_tune"

    # --- Metrics ------------------------------------------------------------
    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    # --- Core Execution -----------------------------------------------------
    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingModelFineTuneTask execution.")

        # ---------------------------------------------------------------------
        # Config-driven model + hyperparameters
        # ---------------------------------------------------------------------
        model_base = AppConfigs.get_instance().get_str("MODEL_NAME", "distilbert-base-uncased")
        model_name_dir = AppConfigs.get_instance().get_str("MODEL_NAME_DIR", "customer-support-distilbert")
        model_version = AppConfigs.get_instance().get_str("MODEL_VERSION", "1.0.0")
        output_root = AppConfigs.get_instance().get_str("MODELS_OUTPUT_DIR", "../outputs/model_outputs")

        num_epochs = AppConfigs.get_instance().get_int("TRAIN_NUM_EPOCHS", 12)
        train_batch_size = AppConfigs.get_instance().get_int("TRAIN_BATCH_SIZE", 2)
        eval_batch_size = AppConfigs.get_instance().get_int("EVAL_BATCH_SIZE", 4)
        warmup_steps = AppConfigs.get_instance().get_int("TRAIN_WARMUP_STEPS", 200)
        learning_rate = float(AppConfigs.get_instance().get_str("TRAIN_LEARNING_RATE", "8e-5"))
        weight_decay = float(AppConfigs.get_instance().get_str("TRAIN_WEIGHT_DECAY", "0.01"))

        label_cols_csv = AppConfigs.get_instance().get_str(
            "TRAINING_DATASET_LABELS2IDS_COLUMN_NAMES_CSV", "category"
        )
        label_cols = [c.strip() for c in label_cols_csv.split(",") if c.strip()]

        LOGGER.info(f"Model Base: {model_base}")
        LOGGER.info(f"Version: {model_version}")
        LOGGER.info(f"Output Root: {output_root}")
        LOGGER.info(f"Label Columns: {label_cols}")

        tokenizer = getattr(req_dto, "tokenizer", None)
        if tokenizer is None:
            LOGGER.error("Tokenizer not found in req_dto. Ensure TrainingTokenizerTask ran successfully.")
            return WfResponses.FAILURE

        # ---------------------------------------------------------------------
        # Process each workbook
        # ---------------------------------------------------------------------
        for workbook in req_dto.training_data_dataframes:
            excel_file_name = workbook.get("file_name")
            sheets_dict = workbook.get("sheets", {})

            LOGGER.info(f"Starting fine-tuning for workbook: {excel_file_name}")

            for sheet_name, sheet_data in sheets_dict.items():
                tokenized_datasets = sheet_data.get("tokenized_datasets", {})
                if not tokenized_datasets:
                    LOGGER.warning(f"No tokenized datasets for {sheet_name}. Skipping.")
                    continue

                for col_name, ds_info in tokenized_datasets.items():
                    train_ds = ds_info.get("tokenized_train_dataset")
                    test_ds = ds_info.get("tokenized_test_dataset")

                    if train_ds is None or test_ds is None:
                        LOGGER.warning(f"Missing tokenized datasets for {sheet_name}->{col_name}. Skipping.")
                        continue

                    # --- Find appropriate label column ---
                    active_label_col = None
                    for lbl in label_cols:
                        if lbl in train_ds.column_names:
                            active_label_col = lbl
                            break
                    if not active_label_col:
                        LOGGER.warning(f"No label column found in tokenized dataset for {sheet_name}->{col_name}.")
                        continue

                    # --- Get label mappings from req_dto.training_data_labels_mapping ---
                    labels_mapping = (
                        req_dto.training_data_labels_mapping
                        .get(excel_file_name, {})
                        .get(sheet_name, {})
                        .get(active_label_col, {})
                    )

                    label2ids = labels_mapping.get("label2ids_map", {})
                    ids2label = labels_mapping.get("ids2label_map", {})

                    if not label2ids or not ids2label:
                        LOGGER.warning(f"No label mappings found for {sheet_name}->{active_label_col}. Skipping.")
                        continue

                    # --- Convert string labels → numeric IDs safely ---
                    def map_labels(example):
                        value = example[active_label_col]
                        if isinstance(value, list):
                            return {active_label_col: [label2ids.get(v, 0) for v in value]}
                        else:
                            return {active_label_col: label2ids.get(value, 0)}

                    train_ds = train_ds.map(map_labels)
                    test_ds = test_ds.map(map_labels)

                    # --- Rename to 'labels' (required by Trainer) ---
                    train_ds = train_ds.rename_column(active_label_col, "labels")
                    test_ds = test_ds.rename_column(active_label_col, "labels")

                    # --- Output directory per workbook/sheet/column ---
                    model_out_dir = os.path.join(
                        output_root,
                        model_name_dir,
                        model_version,
                        excel_file_name.replace(".xlsx", ""),
                        sheet_name,
                        col_name,
                    )
                    os.makedirs(model_out_dir, exist_ok=True)

                    LOGGER.info(
                        f"Fine-tuning model for {excel_file_name}->{sheet_name}->{col_name} "
                        f"with label column '{active_label_col}' → {model_out_dir}"
                    )

                    # --- Initialize model ---
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_base,
                        num_labels=len(label2ids),
                        id2label=ids2label,
                        label2id=label2ids,
                    )

                    # --- Training arguments ---
                    training_args = TrainingArguments(
                        output_dir=model_out_dir,
                        num_train_epochs=num_epochs,
                        per_device_train_batch_size=train_batch_size,
                        per_device_eval_batch_size=eval_batch_size,
                        warmup_steps=warmup_steps,
                        weight_decay=weight_decay,
                        learning_rate=learning_rate,
                        logging_dir=os.path.join(model_out_dir, "logs"),
                        logging_steps=10,
                        save_steps=100,
                        eval_steps=100,
                        seed=42,
                        remove_unused_columns=False,
                        dataloader_pin_memory=False,
                        dataloader_num_workers=0,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=test_ds,
                        tokenizer=tokenizer,
                        compute_metrics=self.compute_metrics,
                    )

                    # --- Train + Save ---
                    try:
                        LOGGER.info(f"Training started for {excel_file_name}->{sheet_name}->{col_name}")
                        trainer.train()
                        LOGGER.info(f"Training completed for {excel_file_name}->{sheet_name}->{col_name}")
                    except Exception as e:
                        LOGGER.exception(f"Training failed for {excel_file_name}->{sheet_name}->{col_name}: {e}")
                        continue

                    model.save_pretrained(model_out_dir)
                    tokenizer.save_pretrained(model_out_dir)

                    # Save mappings
                    mappings_path = os.path.join(model_out_dir, "label_mapping.json")
                    with open(mappings_path, "w") as f:
                        json.dump(
                            {
                                "label2ids": label2ids,
                                "ids2label": ids2label,
                                "num_labels": len(label2ids),
                            },
                            f,
                            indent=2,
                        )

                    LOGGER.info(f"Saved model and mappings to {model_out_dir}")

        LOGGER.info("COMPLETED TrainingModelFineTuneTask execution.")
        return WfResponses.SUCCESS
