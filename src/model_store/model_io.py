import json
import os
import pickle
import numpy as np
from model.ann import ANN
from data.preprocess import Preprocessor
from utils.exceptions import ModelError
from utils.checks import check_file_exists


class ModelIO:
    @staticmethod
    def _get_pkl_path(filepath: str) -> str:
        base = os.path.splitext(filepath)[0]
        return base + ".pkl"

    @staticmethod
    def save(filepath: str, model: ANN, preprocessor: Preprocessor,
             metadata: dict) -> None:
        # ensure output directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

        pkl_path = ModelIO._get_pkl_path(filepath)

        # save metadata to JSON
        json_payload = {
            "architecture": {
                "input_dim":  model.input_dim,
                "hidden_dim": model.hidden_dim,
                "output_dim": model.output_dim
            },
            "weights_file": os.path.basename(pkl_path),
            "metadata":     metadata
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)
        except Exception as e:
            raise ModelError(
                f"Failed to save metadata to '{filepath}': {e}"
            )

        # save weights + norm params to PKL
        pkl_payload = {
            "weights":       model.get_weights(),
            "normalization": preprocessor.get_norm_params()
        }

        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(pkl_payload, f)
        except Exception as e:
            raise ModelError(
                f"Failed to save weights to '{pkl_path}': {e}"
            )

        print(f"[INFO] Metadata saved → {filepath}")
        print(f"[INFO] Weights saved  → {pkl_path}")

    @staticmethod
    def load(filepath: str) -> tuple:
        # check .json exists
        check_file_exists(filepath)

        # load and validate JSON first
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelError(
                f"Metadata file '{filepath}' is corrupted: {e}"
            )
        except Exception as e:
            raise ModelError(
                f"Could not read metadata file '{filepath}': {e}"
            )

        ModelIO._validate_json(json_payload, filepath)

        # now check .pkl exists
        pkl_path = ModelIO._get_pkl_path(filepath)
        check_file_exists(pkl_path)

        # load weights from PKL 
        try:
            with open(pkl_path, "rb") as f:
                pkl_payload = pickle.load(f)
        except Exception as e:
            raise ModelError(
                f"Could not read weights file '{pkl_path}': {e}"
            )

        ModelIO._validate_pkl(pkl_payload, pkl_path)

        # restore model
        arch = json_payload["architecture"]
        try:
            model = ANN(
                input_dim=arch["input_dim"],
                hidden_dim=arch["hidden_dim"]
            )
            model.set_weights(pkl_payload["weights"])
        except Exception as e:
            raise ModelError(
                f"Failed to restore model architecture: {e}"
            )

        # restore preprocessor
        preprocessor = Preprocessor(normalize=True)
        preprocessor.set_norm_params(pkl_payload["normalization"])

        metadata = json_payload.get("metadata", {})

        print(
            f"[INFO] Model loaded. "
            f"Architecture: {arch['input_dim']} → "
            f"{arch['hidden_dim']} → "
            f"{arch['output_dim']}"
        )

        return model, preprocessor, metadata

    @staticmethod
    def _validate_json(payload: dict, filepath: str) -> None:
        required = ["architecture", "weights_file"]
        for key in required:
            if key not in payload:
                raise ModelError(
                    f"Metadata file '{filepath}' missing key: '{key}'. "
                    f"File may be corrupted or incompatible."
                )
        required_arch = ["input_dim", "hidden_dim", "output_dim"]
        for key in required_arch:
            if key not in payload["architecture"]:
                raise ModelError(
                    f"Metadata file '{filepath}' missing "
                    f"architecture key: '{key}'."
                )

    @staticmethod
    def _validate_pkl(payload: dict, filepath: str) -> None:
        required = ["weights", "normalization"]
        for key in required:
            if key not in payload:
                raise ModelError(
                    f"Weights file '{filepath}' missing key: '{key}'. "
                    f"File may be corrupted or incompatible."
                )
        required_weights = ["W1", "b1", "W2", "b2"]
        for key in required_weights:
            if key not in payload["weights"]:
                raise ModelError(
                    f"Weights file '{filepath}' missing "
                    f"weight key: '{key}'."
                )