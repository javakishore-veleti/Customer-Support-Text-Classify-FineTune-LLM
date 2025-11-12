from ..interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from packaging import version
import transformers
import numpy as np
import torch
import os
import json
import logging

LOGGER = logging.getLogger(__name__)


class TrainingModelFineTuneTask(TrainingPipelineTask):
    """
    Fine-tunes one transformer model per worksheet.

    - Uses tokenized datasets from TrainingTokenizerTask
    - One model per sheet (or workbook)
    - Saves weights, tokenizer, label mappings, metrics, and inference samples
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_model_fine_tune"

    # ----------------------------------------------------------------------
    # Utility / Helper Methods
    # ----------------------------------------------------------------------

    def _load_env_configs(self):
        """Load model, directory, and training hyperparameter configs."""
        cfg = {}

        cfg["model_output_base_path_folder_name"] = AppConfigs.get_instance().get_str("MODEL_OUTPUT_BASE_FOLDER_NAME_CLASSIFICATION", "aws_service_training_dataset_clustering")
        cfg["model_name"] = AppConfigs.get_instance().get_str("MODEL_NAME", "distilbert-base-uncased")
        cfg["model_version"] = AppConfigs.get_instance().get_str("MODEL_VERSION", "1.0.0")
        cfg["model_dir_name"] = AppConfigs.get_instance().get_str("MODEL_NAME_DIR", "customer-support-distilbert")
        cfg["output_root"] = AppConfigs.get_instance().get_str("MODELS_OUTPUT_DIR_CLASSIFICATION", "../outputs/model_outputs/classification")

        label_cols_csv = AppConfigs.get_instance().get_str("TRAINING_DATASET_CLASSIFICATION_COLUMN_NAMES_CSV", "category")
        cfg["label_cols"] = [c.strip() for c in label_cols_csv.split(",") if c.strip()]

        training_cols_csv = AppConfigs.get_instance().get_str(
            "DATASET_COLUMN_NAMES_FOR_TRAINING", "category,sample_question,sample_question_type"
        )
        cfg["training_cols"] = [c.strip() for c in training_cols_csv.split(",") if c.strip()]

        cfg["num_epochs"] = AppConfigs.get_instance().get_int("TRAIN_NUM_EPOCHS", 3)
        cfg["train_batch_size"] = AppConfigs.get_instance().get_int("TRAIN_BATCH_SIZE", 4)
        cfg["eval_batch_size"] = AppConfigs.get_instance().get_int("EVAL_BATCH_SIZE", 8)
        cfg["warmup_steps"] = AppConfigs.get_instance().get_int("TRAIN_WARMUP_STEPS", 100)
        cfg["learning_rate"] = float(AppConfigs.get_instance().get_str("TRAIN_LEARNING_RATE", "5e-5"))
        cfg["weight_decay"] = float(AppConfigs.get_instance().get_str("TRAIN_WEIGHT_DECAY", "0.01"))

        return cfg

    def _compute_metrics(self, eval_pred):
        """Compute accuracy and weighted F1."""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    def _build_training_args(self, model_output_dir, cfg):
        """Build version-safe TrainingArguments block."""
        transformers_version = version.parse(transformers.__version__)

        common_args = dict(
            output_dir=model_output_dir,
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["train_batch_size"],
            per_device_eval_batch_size=cfg["eval_batch_size"],
            warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"],
            learning_rate=cfg["learning_rate"],
            logging_dir=os.path.join(model_output_dir, "logs"),
            logging_steps=50,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
            disable_tqdm=False,
        )

        try:
            if transformers_version >= version.parse("4.10.0"):
                return TrainingArguments(
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=2,
                    **common_args,
                )
            else:
                return TrainingArguments(
                    do_eval=True,
                    save_steps=500,
                    save_total_limit=2,
                    **common_args,
                )
        except TypeError:
            # For very old versions — absolute fallback
            return TrainingArguments(**common_args)

    def _encode_labels(self, dataset, label_col, label2id):
        """Attach integer label IDs to the dataset."""
        def encode(example):
            return {"labels": label2id.get(example[label_col], -1)}

        return dataset.map(encode)

    def _save_json(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ----------------------------------------------------------------------
    # Main Execution
    # ----------------------------------------------------------------------

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingModelFineTuneTask execution.")

        if not AppConfigs.get_instance().get_bool("TRAINING_ENABLE_CLASSIFICATION", False):
            LOGGER.info("Classification DISABLED in .env — skipping.")
            req_dto.training_clustering_enabled = False
            return WfResponses.SUCCESS

        if not req_dto.training_data_dataframes:
            LOGGER.error("No training_data_dataframes found in req_dto.")
            return WfResponses.FAILURE

        cfg = self._load_env_configs()

        tokenizer = getattr(req_dto, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

        LOGGER.info(f"Model Base: {cfg['model_name']}")
        LOGGER.info(f"Output Root: {cfg['output_root']}")
        LOGGER.info(f"Version: {cfg['model_version']}")
        LOGGER.info(f"Training Columns: {cfg['training_cols']}")
        LOGGER.info(f"Label Columns: {cfg['label_cols']}")

        master_summary = {
            "model_base": cfg["model_name"],
            "model_version": cfg["model_version"],
            "output_root": cfg["output_root"],
            "workbooks": [],
        }

        for workbook in req_dto.training_data_dataframes:
            excel_file_name = workbook.get("file_name")
            workbook_summary = {"workbook_name": excel_file_name, "sheets": []}
            sheets_dict = workbook.get("sheets", {})

            LOGGER.info(f"Starting fine-tuning for workbook: {excel_file_name}")

            for sheet_name, sheet_data in sheets_dict.items():
                tokenized_sets = sheet_data.get("tokenized_datasets", {})
                train_ds = tokenized_sets.get("tokenized_train_dataset") or tokenized_sets.get("train")
                test_ds = tokenized_sets.get("tokenized_test_dataset") or tokenized_sets.get("test")

                if not train_ds or not test_ds:
                    LOGGER.warning(f"Incomplete tokenized datasets for {excel_file_name}->{sheet_name}. Skipping.")
                    continue

                # Detect label column
                label_col = next((lc for lc in cfg["label_cols"] if lc in train_ds.column_names), None)
                if not label_col:
                    LOGGER.warning(f"No label column found in {sheet_name}. Skipping.")
                    continue

                # Map labels to IDs
                unique_labels = sorted(set(train_ds[label_col]))
                label2id = {v: i for i, v in enumerate(unique_labels)}
                id2label = {i: v for v, i in label2id.items()}

                train_ds = self._encode_labels(train_ds, label_col, label2id)
                test_ds = self._encode_labels(test_ds, label_col, label2id)

                model_output_dir = os.path.join(
                    cfg["output_root"],
                    cfg["model_dir_name"],
                    cfg["model_version"],
                    cfg["model_output_base_path_folder_name"],
                    sheet_name,
                )
                os.makedirs(model_output_dir, exist_ok=True)
                LOGGER.info(f"Fine-tuning model for {excel_file_name}->{sheet_name} → {model_output_dir}")

                try:
                    # Load model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        cfg["model_name"],
                        num_labels=len(unique_labels),
                        id2label=id2label,
                        label2id=label2id,
                    )

                    # Build trainer
                    training_args = self._build_training_args(model_output_dir, cfg)
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=test_ds,
                        tokenizer=tokenizer,
                        compute_metrics=self._compute_metrics,
                    )

                    # Train + Save
                    trainer.train()
                    trainer.save_model(model_output_dir)
                    tokenizer.save_pretrained(model_output_dir)

                    # Save label mapping
                    self._save_json(
                        os.path.join(model_output_dir, "label_mapping.json"),
                        {"label2id": label2id, "id2label": id2label, "num_labels": len(unique_labels)},
                    )

                    # Evaluate
                    eval_results = trainer.evaluate()
                    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
                    class_report = classification_report(
                        test_ds["labels"],
                        preds,
                        target_names=[str(x) for x in unique_labels],
                        output_dict=True,
                    )

                    self._save_json(
                        os.path.join(model_output_dir, "metrics_summary.json"),
                        {
                            "workbook": excel_file_name,
                            "sheet": sheet_name,
                            "accuracy": eval_results.get("eval_accuracy"),
                            "f1": eval_results.get("eval_f1"),
                            "per_class": class_report,
                        },
                    )

                    # Inference samples
                    pipe = pipeline("text-classification", model=model_output_dir, tokenizer=model_output_dir)
                    test_texts = test_ds["__combined_text__"][:5] if "__combined_text__" in test_ds.column_names else []
                    inference_samples = [
                        {
                            "text": t[:250],
                            "predicted_label": (pred := pipe(t[:250])[0])["label"],
                            "confidence": round(pred["score"], 4),
                        }
                        for t in test_texts
                    ]
                    self._save_json(os.path.join(model_output_dir, "inference_samples.json"), inference_samples)

                    workbook_summary["sheets"].append({
                        "sheet_name": sheet_name,
                        "model_output_dir": model_output_dir,
                        "num_labels": len(unique_labels),
                        "accuracy": eval_results.get("eval_accuracy"),
                        "f1": eval_results.get("eval_f1"),
                        "num_train_samples": len(train_ds),
                        "num_test_samples": len(test_ds),
                    })

                    LOGGER.info(f"Completed fine-tuning and evaluation for {sheet_name}")

                except Exception as e:
                    LOGGER.exception(f"Training failed for {excel_file_name}->{sheet_name}: {e}")
                    continue

            master_summary["workbooks"].append(workbook_summary)

        # Save master summary
        master_summary_path = os.path.join(
            cfg["output_root"], cfg["model_dir_name"], cfg["model_version"], "training_master_summary.json"
        )
        self._save_json(master_summary_path, master_summary)

        LOGGER.info(f"Master summary saved → {master_summary_path}")
        LOGGER.info("COMPLETED TrainingModelFineTuneTask execution.")
        return WfResponses.SUCCESS
