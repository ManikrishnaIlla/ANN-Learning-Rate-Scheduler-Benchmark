# ANN Learning Rate Scheduler Benchmark

> A CLI-based ANN Training Tool with Modular Learning Rate Schedulers and Benchmarking for Convergence Analysis
 
A CLI-based ANN training tool built from scratch in Python.
Trains a neural network on your data, benchmarks all learning
rate schedulers automatically, identifies the best one, and
saves the best trained model ready for prediction.

No sklearn. No PyTorch. No TensorFlow. Pure Python + NumPy.

---

## Core Goal

Most people choose a learning rate by guesswork or trial and
error. This tool solves that by automatically training your
ANN under 4 different learning rate strategies, comparing
them using full metrics, and telling you which one works best
for your specific dataset.
```
You give data
     ↓
Tool trains ANN with all 4 schedulers in parallel
     ↓
Compares Loss, Accuracy, Precision, Recall, F1
     ↓
Picks best scheduler automatically
     ↓
Saves best model
     ↓
You predict on new data instantly
```

---

## Requirements

- Python 3.8 or above
- Works on Windows, Linux, Mac

---

## Installation
```bash

# step 1 — create virtual environment
python -m venv venv

# step 2 — activate virtual environment

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# step 3 — install dependencies
pip install -r requirements.txt
```

---

## Sample Data (Optional)

If you do not have your own data yet and want to try the tool quickly, generate sample data using the included script:

```bash
python src/generate_sample_data.py
```

This creates two files inside your project:
```
data/
├── train.csv  — 200 samples, 4 features, binary label
└── test.csv   — 50 samples,  4 features, no label
```

Then follow the instructions below using these files.

---

## How To Run

### Option A — using run script
```bash
# Windows
.\run.bat benchmark --data <path/to/your/train.csv>
.\run.bat predict --input <path/to/your/test.csv>

# Linux / Mac
bash run.sh benchmark --data <path/to/your/train.csv>
bash run.sh predict --input <path/to/your/test.csv>
```

### Option B — using Python directly
```bash

# benchmark
python src/main.py benchmark --data <path/to/train.csv>

# predict
python src/main.py predict --input <path/to/test.csv>

# train with specific scheduler
python src/main.py train --data <path/to/train.csv> --scheduler cosine
```

### Get help anytime
```bash
.\run.bat --help
.\run.bat benchmark --help
.\run.bat predict --help
.\run.bat train --help
```

---

## Input Data Format

### Training file
- Any CSV file with a header row
- All columns except last = numeric features
- Last column = binary label (0 or 1 only)
```
age,salary,score,label
23,45000,88,1
45,92000,76,0
31,67000,91,1
```

### Prediction file
- Same columns as training file
- No label column
```
age,salary,score
29,55000,85
41,88000,72
```

---

## CLI — Full Reference

> **Run commands from the project root directory.**
> All paths are relative to the project root.


### Step 1 — benchmark

Runs all 4 schedulers in parallel, compares results, and automatically saves the best model. Start here if you do not know which scheduler works best for your data.
```bash
python src/main.py benchmark --data <file> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| --data | Yes | — | Path to training CSV |
| --output | No | outputs/benchmark.csv | Path to save benchmark report |
| --save-best | No | models/best_model.json | Path to save best model |
| --lr | No | 0.01 | Initial learning rate |
| --epochs | No | 100 | Number of training epochs |
| --hidden-dim | No | 16 | Number of neurons in hidden layer |
| --seed | No | 42 | Random seed for reproducibility |
| --n-jobs | No | -1 | CPU cores to use (-1 = all cores) |

Examples:
```bash
# minimum — benchmark all schedulers on your data
python src/main.py benchmark --data data/train.csv

# run longer with a larger hidden layer
python src/main.py benchmark --data data/train.csv --epochs 200 --hidden-dim 32

# limit CPU cores if your machine is slow
python src/main.py benchmark --data data/train.csv --n-jobs 2

# save benchmark report and best model to custom locations
python src/main.py benchmark --data data/train.csv --output outputs/myreport.csv --save-best models/mymodel.json

# full control over all parameters
python src/main.py benchmark --data data/train.csv --lr 0.005 --epochs 200 --hidden-dim 32 --seed 99 --n-jobs 4
```

**After benchmark:** The best model is saved to `models/best_model.json` and `models/best_model.pkl`. The benchmark report is saved to `outputs/benchmark.csv`. You can now run `predict` directly using the saved best model — no need to run `train` separately unless you want to experiment with a specific scheduler.


---

### Step 2 (optional) — train

Trains with one specific scheduler of your choice and saves the model. Use this if benchmark already told you which scheduler works best and you want to retrain with custom settings, or if you want to save a model under a specific name.
```bash
python src/main.py train --data <file> --scheduler <name> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| --data | Yes | — | Path to training CSV |
| --scheduler | Yes | — | constant / step / exponential / cosine |
| --model | No | models/model.json | Path to save trained model |
| --lr | No | 0.01 | Initial learning rate |
| --epochs | No | 100 | Number of training epochs |
| --hidden-dim | No | 16 | Number of neurons in hidden layer |
| --seed | No | 42 | Random seed for reproducibility |

Examples:
```bash
# train with cosine scheduler using defaults
python src/main.py train --data data/train.csv --scheduler cosine

# train with step decay for more epochs
python src/main.py train --data data/train.csv --scheduler step --epochs 200

# save model to a custom name so you can compare multiple runs
python src/main.py train --data data/train.csv --scheduler cosine --model models/cosine_exp1.json

# full control — recommended after benchmark tells you the best scheduler
python src/main.py train --data data/train.csv --scheduler exponential --lr 0.005 --epochs 150 --hidden-dim 32 --seed 7
```

**After train:** The model is saved to `models/model.json` and `models/model.pkl` (or your custom path if you used `--model`). You are now ready to run `predict`. Pass the saved model path using `--model` in the predict command.

---

### Step 3 — predict

Loads a saved model and runs inference on new unseen data. The input CSV must have the same number of features as the training data but no label column.
```bash
python src/main.py predict --input <file> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| --input | Yes | — | Path to prediction CSV (no label column) |
| --model | No | models/best_model.json | Path to saved model .json file |
| --output | No | outputs/predictions.csv | Path to save predictions |

Examples:
```bash
# minimum — predict using the best model saved by benchmark
python src/main.py predict --input data/test.csv

# use a model saved by the train command instead of the benchmark best model
python src/main.py predict --input data/test.csv --model models/model.json

# use a specific named model from a previous training run
python src/main.py predict --input data/test.csv --model models/cosine_exp1.json

# save predictions to a custom location instead of the default
python src/main.py predict --input data/test.csv --output outputs/myresults.csv
```

**After predict:** Predictions are saved to `outputs/predictions.csv` (or your custom path). Each row contains the sample number, predicted probability (0.0 to 1.0), and predicted class label (0 or 1).
---

## Schedulers

| Scheduler | How LR Changes | Best For |
|---|---|---|
| constant | stays fixed at lr₀ | baseline comparison |
| step | drops by half every 10 epochs | stable step-wise decay |
| exponential | smooth decay every epoch | gradual convergence |
| cosine | follows cosine curve to near zero | smooth convergence |

---

## What This Tool Will NOT Accept

These will raise a clean error — not a crash:

- Non-numeric feature values in CSV
- Labels other than 0 or 1
- Empty CSV file
- CSV with inconsistent column count
- Prediction file with different number of features than training
- Corrupted or missing model file
- Unknown scheduler name
- Negative learning rate

---

## Typical User Journeys

**I don't know which scheduler to use:**
```bash
.\run.bat benchmark --data data/train.csv
.\run.bat predict --input data/test.csv
```

**I want full control:**
```bash
.\run.bat benchmark --data data/train.csv --epochs 200 --hidden-dim 32 --n-jobs 4
.\run.bat predict --input data/test.csv
```

**I already know which scheduler I want:**
```bash
.\run.bat train --data data/train.csv --scheduler cosine --epochs 150
.\run.bat predict --input data/test.csv
```

---

## Running Tests

The test suite validates that all edge cases are handled correctly — bad input, corrupted files, wrong dimensions, and invalid schedulers all raise clean errors without crashing.

Run before using the tool on your own data to confirm everything works in your environment:
```bash
pytest tests/ -v
```

Expected output: 14 passed, 0 failed. If all pass, the tool is ready to use.

---

## Assumptions and Limitations

- Binary classification only — labels must be 0 or 1
- Numeric features only — no categorical or text data
- No missing values (NaN) allowed
- Fixed architecture — one hidden layer (input → hidden → output)
- CSV must have a header row
- Prediction CSV must have same number of features as training CSV
- CPU only — no GPU support