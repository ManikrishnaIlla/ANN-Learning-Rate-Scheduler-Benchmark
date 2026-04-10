import os
import time
import numpy as np
from model.ann import ANN
from training.trainer import Trainer
from training.loss import BinaryCrossEntropyLoss
from training.metrics import Metrics
from scheduler import get_scheduler
from utils.exceptions import ModelError


class BenchmarkRunner:
    SCHEDULERS = ["constant", "step", "exponential", "cosine"]

    def __init__(self, input_dim: int, hidden_dim: int,
                 initial_lr: float, epochs: int,
                 seed: int = 42, n_jobs: int = -1):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.initial_lr = initial_lr
        self.epochs     = epochs
        self.seed       = seed
        self.n_jobs     = n_jobs

    def run(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> tuple:
 
        from joblib import Parallel, delayed

        # show how many cores are being used
        import multiprocessing
        total_cores    = multiprocessing.cpu_count()
        cores_used     = total_cores if self.n_jobs == -1 else self.n_jobs

        print(f"[INFO] Running benchmark across all schedulers ...")
        print(
            f"[INFO] Epochs: {self.epochs} | "
            f"LR: {self.initial_lr} | "
            f"Hidden: {self.hidden_dim} | "
            f"Seed: {self.seed} | "
            f"Cores: {cores_used}/{total_cores}\n"
        )

        def run_single(name: str) -> dict:
            model = ANN(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim
            )
            model.init_weights(seed=self.seed)

            scheduler = get_scheduler(
                name=name,
                initial_lr=self.initial_lr,
                epochs=self.epochs
            )
            trainer = Trainer(
                model=model,
                scheduler=scheduler,
                loss_fn=BinaryCrossEntropyLoss(),
                epochs=self.epochs,
                verbose=False
            )

            start   = time.time()
            history = trainer.train(X_train, y_train, X_val, y_val)
            elapsed = round(time.time() - start, 4)

            y_pred_val = model.forward(X_val)
            m = Metrics.all_metrics(y_pred_val, y_val)

            return {
                "scheduler":    name,
                "model":        model,
                "final_loss":   history["loss"][-1],
                "accuracy":     m["accuracy"],
                "precision":    m["precision"],
                "recall":       m["recall"],
                "f1":           m["f1"],
                "total_time_s": elapsed
            }

        # run all schedulers in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_single)(name) for name in self.SCHEDULERS
        )

        # identify winners
        best_f1      = max(results, key=lambda r: r["f1"])
        fastest      = min(results, key=lambda r: r["total_time_s"])
        best_name    = best_f1["scheduler"]
        best_model   = best_f1["model"]

        # print aligned results table
        print(f"{'Scheduler':<14} {'Acc':>6} {'F1':>6} "
              f"{'Loss':>7} {'Time':>7}  Note")
        print("─" * 58)

        for r in results:
            note = ""
            if r["scheduler"] == best_f1["scheduler"]:
                note = "← Best F1"
            elif r["scheduler"] == fastest["scheduler"] and \
                 fastest["scheduler"] != best_f1["scheduler"]:
                note = "← Fastest"

            print(
                f"{r['scheduler']:<14} "
                f"{r['accuracy']*100:>5.1f}% "
                f"{r['f1']:>6.3f} "
                f"{r['final_loss']:>7.4f} "
                f"{r['total_time_s']:>6.2f}s  "
                f"{note}"
            )

        print("─" * 58)
        print(f"\n{'Recommended':<14} : "
              f"{best_name} (Best F1: {best_f1['f1']:.3f})")
        print(
            f"\n[INFO] Full metrics for best scheduler ({best_name}):\n"
            f"       Accuracy  : {best_f1['accuracy']*100:.1f}%\n"
            f"       Precision : {best_f1['precision']:.3f}\n"
            f"       Recall    : {best_f1['recall']:.3f}\n"
            f"       F1 Score  : {best_f1['f1']:.3f}\n"
            f"       Loss      : {best_f1['final_loss']:.4f}\n"
            f"       Time      : {best_f1['total_time_s']:.2f}s"
        )

        return results, best_model, best_name