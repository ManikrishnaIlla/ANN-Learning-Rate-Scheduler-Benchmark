# src/model_store/output_io.py
import csv
import os
from utils.exceptions import ModelError


class OutputIO:
    @staticmethod
    def _ensure_dir(filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

    @staticmethod
    def write_predictions(filepath: str, sample_ids: list,
                          probabilities: list, labels: list) -> None:
        OutputIO._ensure_dir(filepath)

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "probability", "predicted_label"])
                for sid, prob, label in zip(sample_ids, probabilities, labels):
                    writer.writerow([sid, round(float(prob), 6), int(label)])
        except Exception as e:
            raise ModelError(f"Failed to write predictions to '{filepath}': {e}")

        print(f"[INFO] Predictions saved → {filepath}")

    @staticmethod
    def write_benchmark(filepath: str, results: list) -> None:
        OutputIO._ensure_dir(filepath)

        if not results:
            raise ModelError("Benchmark results are empty. Nothing to write.")

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "scheduler", "final_loss", "accuracy",
                    "precision", "recall", "f1", "total_time_s"
                ])
                for r in results:
                    writer.writerow([
                        r["scheduler"],
                        round(r["final_loss"],   6),
                        round(r["accuracy"],     4),
                        round(r["precision"],    4),
                        round(r["recall"],       4),
                        round(r["f1"],           4),
                        round(r["total_time_s"], 4)
                    ])
        except Exception as e:
            raise ModelError(
                f"Failed to write benchmark results to '{filepath}': {e}"
            )

        print(f"[INFO] Benchmark report saved → {filepath}")