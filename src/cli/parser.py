import argparse

def build_parser():
    parser = argparse.ArgumentParser(
        prog="ann_tool",
        description="ANN Training Tool with Modular LR Schedulers"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── TRAIN ──────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train the ANN model")
    train_parser.add_argument("--data",       required=True,                   help="Path to training CSV file")
    train_parser.add_argument("--model",      default="models/model.json",     help="Path to save trained model (default: models/model.json)")
    train_parser.add_argument("--scheduler",  required=True,
                              choices=["constant", "step", "exponential", "cosine"],
                              help="Learning rate scheduler")
    train_parser.add_argument("--lr",         type=float, default=0.01,        help="Initial learning rate (default: 0.01)")
    train_parser.add_argument("--epochs",     type=int,   default=100,         help="Number of training epochs (default: 100)")
    train_parser.add_argument("--hidden-dim", type=int,   default=16,          help="Hidden layer size (default: 16)")
    train_parser.add_argument("--seed",       type=int,   default=42,          help="Random seed (default: 42)")

    # ── PREDICT ────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Run inference on new data")
    predict_parser.add_argument("--model",  default="models/best_model.json",  help="Path to saved model (default: models/best_model.json)")
    predict_parser.add_argument("--input",  required=True,                     help="Path to input CSV file")
    predict_parser.add_argument("--output", default="outputs/predictions.csv", help="Path to save predictions (default: outputs/predictions.csv)")

    # ── BENCHMARK ──────────────────────────────────────
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark all schedulers")
    benchmark_parser.add_argument("--data",       required=True,                     help="Path to training CSV file")
    benchmark_parser.add_argument("--output",     default="outputs/benchmark.csv",   help="Path to save benchmark CSV (default: outputs/benchmark.csv)")
    benchmark_parser.add_argument("--save-best",  default="models/best_model.json",  help="Path to save best model (default: models/best_model.json)")
    benchmark_parser.add_argument("--lr",         type=float, default=0.01,          help="Initial learning rate (default: 0.01)")
    benchmark_parser.add_argument("--epochs",     type=int,   default=100,           help="Number of epochs (default: 100)")
    benchmark_parser.add_argument("--hidden-dim", type=int,   default=16,            help="Hidden layer size (default: 16)")
    benchmark_parser.add_argument("--seed",       type=int,   default=42,            help="Random seed (default: 42)")
    benchmark_parser.add_argument("--n-jobs",     type=int,   default=-1,            help="CPU cores. -1=all, 1=single (default: -1)")

    return parser