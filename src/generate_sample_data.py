import numpy as np
import csv
import os

np.random.seed(42)

# Always save to project root/data/ regardless of where script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# training data: 200 samples, 4 features, binary label
n_train = 200
X_train = np.random.randn(n_train, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

train_path = os.path.join(DATA_DIR, "train.csv")
with open(train_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["f1", "f2", "f3", "f4", "label"])
    for i in range(n_train):
        writer.writerow([*X_train[i].round(6), y_train[i]])

# test data: 50 samples, 4 features, no label
n_test = 50
X_test = np.random.randn(n_test, 4)

test_path = os.path.join(DATA_DIR, "test.csv")
with open(test_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["f1", "f2", "f3", "f4"])
    for i in range(n_test):
        writer.writerow([*X_test[i].round(6)])

print("Sample data generated:")
print(f"  {train_path}  — 200 samples, 4 features, binary label")
print(f"  {test_path}   — 50 samples,  4 features, no label")