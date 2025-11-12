from ...interfaces import TrainingPipelineTask
from overrides import overrides
from training.dtos import TrainingReqDTO, TrainingResDTO
from app_common.app_constants import WfResponses
from app_common.app_configs_util import AppConfigs
import logging
import os
import json
from difflib import SequenceMatcher

LOGGER = logging.getLogger(__name__)


class TrainingDataLeakageCheckerTask(TrainingPipelineTask):
    """
    Detects potential data leakage between train/test splits for each sheet.

    Checks for:
      - Exact duplicates (identical combined texts)
      - Near duplicates (textual similarity above threshold)

    Config (.env):
        DATA_LEAKAGE_SIMILARITY_THRESHOLD=0.9
        DATA_LEAKAGE_OUTPUT_DIR=../outputs/data_leakage_reports
        DATA_LEAKAGE_FAIL_THRESHOLD=0.01   # Optional, fail if >1% leakage
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "training_data_leakage_checker"

    # --- Helper methods ---

    def _similarity(self, a: str, b: str) -> float:
        """Compute normalized text similarity between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def _find_exact_duplicates(self, train_texts, test_texts):
        """Find exact text matches between train/test sets."""
        return list(set(train_texts).intersection(set(test_texts)))

    def _find_near_duplicates(self, train_texts, test_texts, threshold):
        """Find near-duplicate text pairs between train/test sets."""
        near_dupes = []
        for t_text in test_texts:
            for tr_text in train_texts:
                sim = self._similarity(tr_text, t_text)
                if sim >= threshold and tr_text != t_text:
                    near_dupes.append({
                        "train_text": tr_text,
                        "test_text": t_text,
                        "similarity": round(sim, 3)
                    })
        return near_dupes

    def _analyze_sheet(self, sheet_name, train_df, test_df, threshold):
        """Analyze one sheet for leakage indicators."""
        if "__combined_text__" not in train_df.columns or "__combined_text__" not in test_df.columns:
            LOGGER.warning(f"[{sheet_name}] Missing '__combined_text__' column. Skipping.")
            return None

        train_texts = train_df["__combined_text__"].astype(str).tolist()
        test_texts = test_df["__combined_text__"].astype(str).tolist()

        exact_dupes = self._find_exact_duplicates(train_texts, test_texts)
        near_dupes = self._find_near_duplicates(train_texts, test_texts, threshold)

        LOGGER.info(
            f"[{sheet_name}] Train={len(train_texts)} Test={len(test_texts)} | "
            f"Exact={len(exact_dupes)} | Near={len(near_dupes)}"
        )

        return {
            "sheet_name": sheet_name,
            "num_train": len(train_texts),
            "num_test": len(test_texts),
            "exact_duplicates": len(exact_dupes),
            "near_duplicates": len(near_dupes),
            "sample_near_duplicates": near_dupes[:5],
        }

    # --- Main execution ---

    @overrides
    def execute(self, req_dto: TrainingReqDTO, res_dto: TrainingResDTO) -> int:
        LOGGER.info("STARTED TrainingDataLeakageCheckerTask execution.")

        if not req_dto.training_data_dataframes:
            LOGGER.error("No training_data_dataframes found in req_dto.")
            return WfResponses.FAILURE

        cfg = AppConfigs.get_instance()
        threshold = float(cfg.get_str("DATA_LEAKAGE_SIMILARITY_THRESHOLD", "0.9"))
        fail_threshold = float(cfg.get_str("DATA_LEAKAGE_FAIL_THRESHOLD", "0.01"))
        output_dir = cfg.get_str("DATA_LEAKAGE_OUTPUT_DIR", "../outputs/data_leakage_reports")

        os.makedirs(output_dir, exist_ok=True)

        master_report = {
            "similarity_threshold": threshold,
            "fail_threshold": fail_threshold,
            "workbooks": []
        }

        total_exact = 0
        total_near = 0
        total_records = 0

        for workbook in req_dto.training_data_dataframes:
            excel_name = workbook.get("file_name", "unknown.xlsx")
            sheets_dict = workbook.get("sheets", {})
            LOGGER.info(f"Checking data leakage for workbook: {excel_name}")

            workbook_report = {"workbook": excel_name, "sheets": []}

            for sheet_name, sheet_data in sheets_dict.items():
                train_df = sheet_data.get("train_df")
                test_df = sheet_data.get("test_df")

                if train_df is None or test_df is None:
                    LOGGER.warning(f"[{sheet_name}] Missing train/test split. Skipping.")
                    continue

                sheet_report = self._analyze_sheet(sheet_name, train_df, test_df, threshold)
                if not sheet_report:
                    continue

                workbook_report["sheets"].append(sheet_report)
                total_exact += sheet_report["exact_duplicates"]
                total_near += sheet_report["near_duplicates"]
                total_records += sheet_report["num_train"] + sheet_report["num_test"]

            master_report["workbooks"].append(workbook_report)

        # --- Compute global leakage ratio ---
        total_dupes = total_exact + total_near
        leakage_ratio = total_dupes / total_records if total_records else 0.0
        master_report["summary"] = {
            "total_records": total_records,
            "exact_duplicates": total_exact,
            "near_duplicates": total_near,
            "leakage_ratio": round(leakage_ratio, 4),
            "status": "PASS" if leakage_ratio <= fail_threshold else "FAIL"
        }

        # --- Save JSON report ---
        report_path = os.path.join(output_dir, "data_leakage_report.json")
        with open(report_path, "w") as f:
            json.dump(master_report, f, indent=2)

        LOGGER.info(f"Leakage report saved → {report_path}")
        LOGGER.info(
            f"Overall Leakage Ratio = {leakage_ratio*100:.2f}% "
            f"(Threshold={fail_threshold*100:.2f}%)"
        )

        if leakage_ratio > fail_threshold:
            LOGGER.error("Data leakage exceeds allowed threshold. Failing pipeline.")
            return WfResponses.FAILURE

        LOGGER.info("COMPLETED TrainingDataLeakageCheckerTask execution.")
        return WfResponses.SUCCESS
