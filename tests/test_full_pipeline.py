import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
from utils.seed import set_seed
from data.preprocess import Preprocessor
from model.ann import ANN
from training.trainer import Trainer
from training.loss import BinaryCrossEntropyLoss
from training.metrics import Metrics
from scheduler import get_scheduler
from model_store.model_io import ModelIO

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../models/pipeline_test.json"
)

def test_full_pipeline():
    set_seed(42)
    np.random.seed(42)

    # generate data
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # preprocess
    preprocessor = Preprocessor(normalize=True)
    X_train, X_val, y_train, y_val = preprocessor.split(X, y, seed=42)
    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)

    # train
    model = ANN(input_dim=4, hidden_dim=8)
    model.init_weights(seed=42)
    scheduler = get_scheduler("cosine", initial_lr=0.01, epochs=50)
    loss_fn   = BinaryCrossEntropyLoss()
    trainer   = Trainer(model, scheduler, loss_fn, epochs=50, verbose=False)
    history   = trainer.train(X_train, y_train, X_val, y_val)

    print(f"[1] Train OK — final loss: {history['loss'][-1]:.4f}")

    # save
    ModelIO.save(
        filepath=MODEL_PATH,
        model=model,
        preprocessor=preprocessor,
        metadata={"scheduler": "cosine", "epochs": 50,
                  "final_loss": history["loss"][-1],
                  "final_val_acc": history["val_acc"][-1]}
    )
    print(f"[2] Save OK — {MODEL_PATH}")

    # load
    loaded_model, loaded_prep, meta = ModelIO.load(MODEL_PATH)
    print(f"[3] Load OK — metadata: {meta}")

    # predict
    X_norm  = loaded_prep.transform(X[:10])
    y_prob  = loaded_model.forward(X_norm)
    y_label = Metrics.predict_labels(y_prob)
    print(f"[4] Predict OK — labels: {y_label}")

    # verify predictions are valid
    assert all(l in [0, 1] for l in y_label), "Labels must be 0 or 1"
    assert all(0 <= p <= 1 for p in y_prob),  "Probabilities must be in [0,1]"

    print("\nPASS — full pipeline works end to end")

    # cleanup
    if os.path.exists(MODEL_PATH):
        os.unlink(MODEL_PATH)


if __name__ == "__main__":
    print("=== test_full_pipeline ===")
    test_full_pipeline()