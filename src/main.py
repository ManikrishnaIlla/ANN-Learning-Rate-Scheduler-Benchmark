# src/main.py
import sys
from cli.parser import build_parser


def run_train(args) -> None:
    import numpy as np
    from utils.seed import set_seed
    from utils.exceptions import DataError, ModelError, SchedulerError
    from data.loader import CSVLoader
    from data.validator import DataValidator
    from data.preprocess import Preprocessor
    from model.ann import ANN
    from training.trainer import Trainer
    from training.loss import BinaryCrossEntropyLoss
    from scheduler import get_scheduler
    from model_store.model_io import ModelIO

    try:
        set_seed(args.seed)

        # load and validate data
        print(f"[INFO] Loading data from {args.data} ...")
        loader    = CSVLoader(args.data)
        headers, raw_rows = loader.load()

        validator = DataValidator(is_training=True)
        validator.validate(headers, raw_rows)

        # preprocess
        preprocessor = Preprocessor(normalize=True)
        X, y = preprocessor.parse(raw_rows, is_training=True)
        X_train, X_val, y_train, y_val = preprocessor.split(
            X, y, seed=args.seed
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val   = preprocessor.transform(X_val)

        print(
            f"[INFO] Samples: {len(X_train)} train / "
            f"{len(X_val)} val | Features: {X_train.shape[1]}"
        )

        # build model
        hidden_dim = args.hidden_dim
        model = ANN(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
        model.init_weights(seed=args.seed)

        # build scheduler
        scheduler = get_scheduler(
            name=args.scheduler,
            initial_lr=args.lr,
            epochs=args.epochs
        )
        print(
            f"[INFO] Scheduler: {scheduler.__class__.__name__} "
            f"| LR: {args.lr}"
        )

        # train
        loss_fn = BinaryCrossEntropyLoss()
        trainer = Trainer(
            model=model,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=args.epochs,
            verbose=True
        )
        history = trainer.train(X_train, y_train, X_val, y_val)

        # save model
        metadata = {
            "scheduler":     args.scheduler,
            "epochs":        args.epochs,
            "final_loss":    history["loss"][-1],
            "final_val_acc": history["val_acc"][-1]
        }
        ModelIO.save(
            filepath=args.model,
            model=model,
            preprocessor=preprocessor,
            metadata=metadata
        )

    except (DataError, ModelError, SchedulerError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


def run_predict(args) -> None:
    import numpy as np
    from utils.exceptions import DataError, ModelError, DimensionError
    from data.loader import CSVLoader
    from data.validator import DataValidator
    from data.preprocess import Preprocessor
    from training.metrics import Metrics
    from model_store.model_io import ModelIO
    from model_store.output_io import OutputIO
    from utils.checks import check_dimensions

    try:
        # load model
        model, preprocessor, metadata = ModelIO.load(args.model)

        # load and validate input
        print(f"[INFO] Loading input from {args.input} ...")
        loader = CSVLoader(args.input)
        headers, raw_rows = loader.load()

        validator = DataValidator(is_training=False)
        validator.validate(headers, raw_rows)

        # preprocess
        prep = Preprocessor(normalize=True)
        X, _ = prep.parse(raw_rows, is_training=False)

        # dimension check
        check_dimensions(X, model.input_dim)

        # apply saved normalization
        X_norm = preprocessor.transform(X)

        # inference
        print(f"[INFO] Running inference on {len(X)} samples ...")
        y_prob   = model.forward(X_norm)
        y_labels = Metrics.predict_labels(y_prob)

        #write output
        sample_ids = list(range(1, len(X) + 1))
        OutputIO.write_predictions(
            filepath=args.output,
            sample_ids=sample_ids,
            probabilities=y_prob.tolist(),
            labels=y_labels.tolist()
        )

    except (DataError, ModelError, DimensionError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


def run_benchmark(args) -> None:
    import sys
    from utils.seed import set_seed
    from utils.exceptions import DataError, ModelError
    from data.loader import CSVLoader
    from data.validator import DataValidator
    from data.preprocess import Preprocessor
    from benchmark.runner import BenchmarkRunner
    from model_store.model_io import ModelIO
    from model_store.output_io import OutputIO

    try:
        set_seed(args.seed)

        # load and validate data
        print(f"[INFO] Loading data from {args.data} ...")
        loader = CSVLoader(args.data)
        headers, raw_rows = loader.load()

        validator = DataValidator(is_training=True)
        validator.validate(headers, raw_rows)

        # preprocess
        preprocessor = Preprocessor(normalize=True)
        X, y = preprocessor.parse(raw_rows, is_training=True)
        X_train, X_val, y_train, y_val = preprocessor.split(
            X, y, seed=args.seed
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val   = preprocessor.transform(X_val)

        print(
            f"[INFO] Samples: {len(X_train)} train / "
            f"{len(X_val)} val | Features: {X_train.shape[1]}\n"
        )

        # run benchmark
        runner = BenchmarkRunner(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden_dim,
            initial_lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
        results, best_model, best_name = runner.run(
            X_train, y_train, X_val, y_val
        )

        # save benchmark report
        csv_results = [
            {k: v for k, v in r.items() if k != "model"}
            for r in results
        ]
        OutputIO.write_benchmark(
            filepath=args.output,
            results=csv_results
        )

        # save best model if requested
        if args.save_best:
            best_result = next(
                r for r in results if r["scheduler"] == best_name
            )
            metadata = {
                "scheduler":  best_name,
                "epochs":     args.epochs,
                "final_loss": best_result["final_loss"],
                "accuracy":   best_result["accuracy"],
                "precision":  best_result["precision"],
                "recall":     best_result["recall"],
                "f1":         best_result["f1"]
            }
            ModelIO.save(
                filepath=args.save_best,
                model=best_model,
                preprocessor=preprocessor,
                metadata=metadata
            )
            print(f"[INFO] Best model saved → {args.save_best}")
            print(f"[DONE] Ready for prediction using best model.")

    except (DataError, ModelError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()