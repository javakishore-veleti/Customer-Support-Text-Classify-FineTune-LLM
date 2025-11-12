import os
import json
import logging
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
from overrides import overrides

from ..interfaces import TrainingPipelineTask
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_configs_util import AppConfigs
from app_common.app_constants import WfResponses

LOGGER = logging.getLogger(__name__)


class TrainingDataClusteringTask(TrainingPipelineTask):
    """
    Independent unsupervised clustering task.
    - Reads Excel files directly.
    - Embeds configured text columns.
    - Clusters per worksheet (skips sheets from DATASET_WORKSHEETS_TO_SKIP_CSV).
    - Saves results under model/version/<workbook>/<sheet>/clustering.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_data_clustering"

    # ============================================================
    # Configuration and Setup
    # ============================================================
    def _get_env_config(self):
        cfg = AppConfigs.get_instance()
        return {
            "enabled": cfg.get_str("TRAINING_ENABLE_CLUSTERING", "true").lower() == "true",
            "excel_dir": cfg.get_str("TRAINING_DATASET_EXCEL_FILE_PATH"),
            "excel_names_csv": cfg.get_str("TRAINING_DATASET_EXCEL_FILE_NAMES_CSV", "ALL_FILES"),
            "text_columns": [
                c.strip() for c in cfg.get_str("TRAINING_DATASET_CLUSTERING_TEXT_COLUMNS_CSV", "").split(",") if c.strip()
            ],
            "worksheets_to_skip": [
                s.strip() for s in cfg.get_str("DATASET_WORKSHEETS_TO_SKIP_CSV", "").split(",") if s.strip()
            ],
            "algorithm": cfg.get_str("TRAINING_CLUSTERING_ALGORITHM", "kmeans").lower(),
            "num_clusters": int(cfg.get_str("TRAINING_CLUSTERING_NUM_CLUSTERS", "5")),
            "dim_reduction": cfg.get_str("TRAINING_CLUSTERING_DIM_REDUCTION", "umap").lower(),
            "distance_metric": cfg.get_str("TRAINING_CLUSTERING_DISTANCE_METRIC", "cosine").lower(),
            "random_state": int(cfg.get_str("TRAINING_CLUSTERING_RANDOM_STATE", "42")),
            "base_output_dir": cfg.get_str("MODELS_OUTPUT_DIR", "../outputs/model_outputs"),
            "model_dir": cfg.get_str("MODEL_NAME_DIR", "customer-support-distilbert"),
            "model_version": cfg.get_str("MODEL_VERSION", "1.0.0"),
            "model_name": cfg.get_str("MODEL_NAME", "distilbert-base-uncased"),
            "model_output_base_folder_name": cfg.get_str("MODEL_OUTPUT_BASE_FOLDER_NAME_CLUSTERING", "aws_service_training_dataset_clustering"),
        }

    def _get_excel_files(self, config):
        excel_dir = config["excel_dir"]
        if config["excel_names_csv"].strip().upper() == "ALL_FILES":
            return [f for f in os.listdir(excel_dir) if f.endswith(".xlsx")]
        return [f.strip() for f in config["excel_names_csv"].split(",") if f.strip()]

    # ============================================================
    # Embedding and Clustering Utilities
    # ============================================================
    def _load_embedding_model(self, model_name: str):
        LOGGER.info(f"Loading embedding model: {model_name}")
        return SentenceTransformer(model_name)

    def _reduce_dimensions(self, embeddings, method="none"):
        if method == "pca":
            return PCA(n_components=2).fit_transform(embeddings)
        elif method == "umap":
            reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
            return reducer.fit_transform(embeddings)
        return embeddings

    def _cluster_embeddings(self, embeddings, config):
        algo = config["algorithm"]
        if algo == "kmeans":
            model = KMeans(n_clusters=config["num_clusters"], random_state=config["random_state"])
        elif algo == "dbscan":
            model = DBSCAN(metric=config["distance_metric"])
        elif algo == "agglomerative":
            model = AgglomerativeClustering(n_clusters=config["num_clusters"])
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algo}")
        labels = model.fit_predict(embeddings)
        return labels, model

    def _visualize_clusters(self, reduced_embeddings, labels, output_path):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap="tab10",
            s=25
        )
        plt.title("Clustering Visualization")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # ============================================================
    # Core Logic per Worksheet
    # ============================================================
    def _should_skip_sheet(self, sheet_name: str, skip_list: list[str]) -> bool:
        """Return True if this worksheet should be skipped."""
        normalized = sheet_name.strip().lower().replace("amazon_", "").replace("aws_", "")
        skip_normalized = [s.strip().lower().replace("amazon_", "").replace("aws_", "") for s in skip_list]
        return normalized in skip_normalized

    def _process_worksheet(self, workbook_name, sheet_name, df, model, config):
        # --- Skip logic ---
        if self._should_skip_sheet(sheet_name, config["worksheets_to_skip"]):
            LOGGER.warning(f"Skipping sheet per config: {sheet_name}")
            return None

        if df.empty:
            LOGGER.warning(f"Skipping empty sheet: {sheet_name}")
            return None

        text_cols = config["text_columns"]
        if not text_cols or not all(c in df.columns for c in text_cols):
            LOGGER.warning(f"[{sheet_name}] Missing clustering columns: {text_cols}")
            return None

        # Combine text columns
        df["__cluster_text__"] = df[text_cols].astype(str).agg(" | ".join, axis=1)
        texts = df["__cluster_text__"].tolist()

        LOGGER.info(f"[{sheet_name}] Encoding {len(texts)} samples...")
        embeddings = model.encode(texts, show_progress_bar=True)

        # Dimensionality reduction + clustering
        reduced = self._reduce_dimensions(embeddings, method=config["dim_reduction"])
        labels, _ = self._cluster_embeddings(embeddings, config)
        df["cluster_label"] = labels

        # Output structure
        output_dir = os.path.join(
            config["base_output_dir"],
            "clustering",
            config["model_dir"],
            config["model_version"],
            config["model_output_base_folder_name"],
            sheet_name
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save outputs
        df.to_csv(os.path.join(output_dir, "clusters.csv"), index=False)
        if reduced.shape[1] == 2:
            self._visualize_clusters(reduced, labels, os.path.join(output_dir, "clusters.png"))

        summary = {
            "workbook": workbook_name,
            "sheet": sheet_name,
            "records": len(df),
            "clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "algorithm": config["algorithm"],
            "columns_used": text_cols
        }

        with open(os.path.join(output_dir, "clustering_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        LOGGER.info(f"[{sheet_name}] ✅ Completed clustering into {summary['clusters']} clusters.")
        return summary

    # ============================================================
    # Main Execution Entry Point
    # ============================================================
    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingDataClusteringTask execution.")
        config = self._get_env_config()

        if not config["enabled"]:
            LOGGER.info("Clustering disabled in .env — skipping.")
            return WfResponses.SUCCESS

        excel_files = self._get_excel_files(config)
        if not excel_files:
            LOGGER.error("No Excel files found for clustering.")
            return WfResponses.FAILURE

        model = self._load_embedding_model(config["model_name"])
        all_results = []

        for excel_file in excel_files:
            excel_path = os.path.join(config["excel_dir"], excel_file)
            workbook_name = os.path.splitext(excel_file)[0]
            LOGGER.info(f"📘 Processing workbook: {excel_file}")

            sheets_dict = pd.read_excel(excel_path, sheet_name=None)
            for sheet_name, df in sheets_dict.items():
                sheet_summary = self._process_worksheet(workbook_name, sheet_name, df, model, config)
                if sheet_summary:
                    all_results.append(sheet_summary)

        # Save master summary
        global_summary_dir = os.path.join(config["base_output_dir"], "clustering", config["model_dir"], config["model_version"], "clustering_results")
        os.makedirs(global_summary_dir, exist_ok=True)
        with open(os.path.join(global_summary_dir, "clustering_summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        LOGGER.info("COMPLETED TrainingDataClusteringTask execution.")
        return WfResponses.SUCCESS
