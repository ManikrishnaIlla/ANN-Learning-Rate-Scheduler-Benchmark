import time
import numpy as np
from model.ann import ANN
from training.loss import BinaryCrossEntropyLoss
from training.metrics import Metrics
from scheduler.base import BaseScheduler
from utils.checks import check_not_empty, check_positive_int, check_positive_float

class Trainer:
    def __init__(self, model: ANN, scheduler: BaseScheduler,
                 loss_fn: BinaryCrossEntropyLoss, epochs: int,
                 verbose: bool = True):
        check_positive_int(epochs, "epochs")
        self.model     = model
        self.scheduler = scheduler
        self.loss_fn   = loss_fn
        self.epochs    = epochs
        self.verbose   = verbose

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        check_not_empty(X_train, "training data")
        check_not_empty(X_val,   "validation data")

        history = {
            "epoch":        [],
            "loss":         [],
            "val_loss":     [],
            "acc":          [],
            "val_acc":      [],
            "lr":           [],
            "elapsed_time": []
        }

        start_time = time.time()

        for epoch in range(self.epochs):

            #  step 1: get current learning rate 
            lr = self.scheduler.get_lr(epoch)

            #  step 2: forward pass 
            y_pred_train = self.model.forward(X_train)

            #  step 3: compute training loss 
            loss = self.loss_fn.compute(y_pred_train, y_train)

            #  step 4: backward pass 
            grads = self.model.backward(y_train)

            #  step 5: update weights 
            self.model.update(grads, learning_rate=lr)

            #  step 6: evaluate on validation set 
            y_pred_val = self.model.forward(X_val)
            val_loss   = self.loss_fn.compute(y_pred_val, y_val)
            acc        = Metrics.accuracy(y_pred_train, y_train)
            val_acc    = Metrics.accuracy(y_pred_val, y_val)

            #  step 7: record history 
            elapsed = time.time() - start_time
            history["epoch"].append(epoch + 1)
            history["loss"].append(round(loss, 6))
            history["val_loss"].append(round(val_loss, 6))
            history["acc"].append(round(acc, 6))
            history["val_acc"].append(round(val_acc, 6))
            history["lr"].append(round(lr, 8))
            history["elapsed_time"].append(round(elapsed, 4))

            #  step 8: print progress 
            if self.verbose and (
                (epoch + 1) % 10 == 0 or epoch == 0
            ):
                print(
                    f"Epoch {epoch+1:>4}/{self.epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {acc*100:.1f}% | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc*100:.1f}% | "
                    f"LR: {lr:.6f}"
                )

        total_time = time.time() - start_time
        print(f"\n[DONE] Training complete in {total_time:.2f}s")

        return history