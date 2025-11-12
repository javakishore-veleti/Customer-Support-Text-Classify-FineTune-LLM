import os
import logging
import json
import numpy as np
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
    Perform unsupervised clustering on embedded text data.
    Supports KMeans, DBSCAN, HDBSCAN (optional), and Agglomerative clustering.
    Configurable from .env.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_data_clustering"

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _get_env_config(self):
        cfg = AppConfigs.get_instance()

        return {
            "enabled": cfg.get_str("TRAINING_ENABLE_CLUSTERING", "false").lower() == "true",
            "text_columns": [
                c.strip() for c in cfg.get_str("TRAINING_DATASET_CLUSTERING_TEXT_COLUMNS_CSV", "").split(",") if c.strip()
            ],
            "algorithm": cfg.get_str("TRAINING_CLUSTERING_ALGORITHM", "kmeans").lower(),
            "num_clusters": int(cfg.get_str("TRAINING_CLUSTERING_NUM_CLUSTERS", "5")),
            "dim_reduction": cfg.get_str("TRAINING_CLUSTERING_DIM_REDUCTION", "none").lower(),
            "distance_metric": cfg.get_str("TRAINING_CLUSTERING_DISTANCE_METRIC", "cosine").lower(),
            "min_cluster_size": int(cfg.get_str("TRAINING_CLUSTERING_MIN_CLUSTER_SIZE", "5")),
            "random_state": int(cfg.get_str("TRAINING_CLUSTERING_RANDOM_STATE", "42")),
            "output_dir": cfg.get_str("TRAINING_CLUSTERING_OUTPUT_DIR", "../outputs/clustering_results"),
            "model_name": cfg.get_str("MODEL_NAME", "distilbert-base-uncased"),
        }

    def _reduce_dimensions(self, embeddings, method="none"):
        """Optionally reduce embedding dimensions for visualization."""
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
        elif method == "umap":
            reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            reduced = embeddings
        return reduced

    def _cluster_embeddings(self, embeddings, config):
        """Run clustering algorithm on embeddings."""
        algo = config["algorithm"]
        if algo == "kmeans":
            model = KMeans(
                n_clusters=config["num_clusters"],
                random_state=config["random_state"]
            )
        elif algo == "dbscan":
            model = DBSCAN(metric=config["distance_metric"])
        elif algo == "agglomerative":
            model = AgglomerativeClustering(n_clusters=config["num_clusters"])
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algo}")

        labels = model.fit_predict(embeddings)
        return labels, model

    def _visualize_clusters(self, reduced_embeddings, labels, output_path):
        """Save UMAP/PCA visualization."""
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap="tab10",
            s=20
        )
        plt.title("Clustering Visualization")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # ------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("🧩 STARTED TrainingDataClusteringTask execution.")

        config = self._get_env_config()
        if not config["enabled"]:
            LOGGER.info("Clustering disabled by config. Skipping task.")
            return WfResponses.SUCCESS

        os.makedirs(config["output_dir"], exist_ok=True)

        # Load model
        LOGGER.info(f"Loading embedding model: {config['model_name']}")
        model = SentenceTransformer(config["model_name"])

        all_results = []
        for workbook in req_dto.training_data_dataframes:
            excel_name = workbook.get("file_name", "unknown.xlsx")
            sheets_dict = workbook.get("sheets", {})
            LOGGER.info(f"Processing workbook: {excel_name}")

            for sheet_name, sheet_data in sheets_dict.items():
                df = sheet_data.get("full_df")
                if df is None or df.empty:
                    LOGGER.warning(f"[{sheet_name}] Missing full_df. Skipping.")
                    continue

                text_cols = config["text_columns"]
                if not all(col in df.columns for col in text_cols):
                    LOGGER.warning(f"[{sheet_name}] Missing one or more clustering columns: {text_cols}")
                    continue

                # Combine text columns into one
                df["__cluster_text__"] = df[text_cols].astype(str).agg(" | ".join, axis=1)
                texts = df["__cluster_text__"].tolist()

                LOGGER.info(f"[{sheet_name}] Embedding {len(texts)} samples...")
                embeddings = model.encode(texts, show_progress_bar=True)

                # Dimensionality reduction
                reduced = self._reduce_dimensions(embeddings, method=config["dim_reduction"])

                # Clustering
                labels, model_obj = self._cluster_embeddings(embeddings, config)
                df["cluster_label"] = labels

                # Save visualization if reduced
                if reduced.shape[1] == 2:
                    viz_path = os.path.join(
                        config["output_dir"], f"{excel_name}_{sheet_name}_clusters.png"
                    )
                    self._visualize_clusters(reduced, labels, viz_path)
                    LOGGER.info(f"[{sheet_name}] Cluster visualization saved → {viz_path}")

                # Save results per sheet
                csv_path = os.path.join(
                    config["output_dir"], f"{excel_name}_{sheet_name}_clusters.csv"
                )
                df.to_csv(csv_path, index=False)

                sheet_summary = {
                    "workbook": excel_name,
                    "sheet": sheet_name,
                    "num_records": len(df),
                    "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                }
                all_results.append(sheet_summary)
                LOGGER.info(f"[{sheet_name}] Completed clustering ({sheet_summary['num_clusters']} clusters)")

        # Save summary JSON
        summary_path = os.path.join(config["output_dir"], "clustering_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        LOGGER.info(f"Clustering summary saved → {summary_path}")
        LOGGER.info("COMPLETED TrainingDataClusteringTask execution.")
        return WfResponses.SUCCESS
