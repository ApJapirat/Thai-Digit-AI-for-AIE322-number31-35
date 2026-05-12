import gc
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from werkzeug.utils import secure_filename


class ModelManager:
    def __init__(
        self,
        model_path: Path,
        model_name_path: Path,
        model_info_path: Path,
        model_version_dir: Path,
        labels: tuple[str, ...],
    ) -> None:
        self.model_path = model_path
        self.model_name_path = model_name_path
        self.model_info_path = model_info_path
        self.model_version_dir = model_version_dir
        self.labels = labels
        self.expected_input_shape = (784,)
        self._lock = threading.RLock()
        self._model = None
        self._model_mtime = None
        self._active_name = None

    def ensure_dirs(self) -> None:
        self.model_version_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self):
        with self._lock:
            if not self.model_path.exists():
                return None
            mtime = self.model_path.stat().st_mtime
            if self._model is None or self._model_mtime != mtime:
                model, validation = self._load_and_validate(self.model_path)
                self._replace_active_model(model, mtime, self.display_name())
                print(
                    f"[model-manager] model loaded: {self.model_path.name} "
                    f"input={validation['input_shape']} output={validation['output_shape']}",
                    flush=True,
                )
            return self._model

    def predict(self, sample: np.ndarray) -> np.ndarray:
        with self._lock:
            model = self.get_model()
            if model is None:
                raise FileNotFoundError("model.joblib not found")
            print(f"[model-manager] predict using model: {self.display_name()}", flush=True)
            return self._predict_probabilities(model, sample)

    def activate_uploaded_model(self, upload_path: Path, original_filename: str, version_name: str) -> dict[str, str]:
        if upload_path.suffix.lower() != ".joblib":
            raise ValueError("Only .joblib model files are supported")

        uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_version = secure_filename(version_name) if version_name else "model"
        if not safe_version:
            safe_version = "model"
        version_filename = f"{timestamp}_{safe_version}.joblib"
        version_path = self.model_version_dir / version_filename
        replacement_path = self.model_path.with_name(f".{self.model_path.name}.{timestamp}.replace")

        self.ensure_dirs()
        with self._lock:
            candidate_model, validation = self._load_and_validate(upload_path)
            print(
                f"[model-manager] model loaded: {original_filename} "
                f"input={validation['input_shape']} output={validation['output_shape']}",
                flush=True,
            )
            if self.model_path.exists():
                shutil.copy2(self.model_path, self.model_version_dir / f"{timestamp}_previous_active_model.joblib")

            shutil.copy2(upload_path, version_path)
            shutil.copy2(upload_path, replacement_path)
            os.replace(replacement_path, self.model_path)

            self.model_name_path.write_text(original_filename, encoding="utf-8")
            info = {
                "version_name": version_name or "uploaded-model",
                "original_filename": original_filename,
                "stored_version_file": version_filename,
                "active_model_file": self.model_path.name,
                "uploaded_at": uploaded_at,
                "activated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "validation_status": "passed validation and is ACTIVE",
                "input_shape": validation["input_shape"],
                "output_shape": validation["output_shape"],
                "test_output_shape": validation["test_output_shape"],
                "expected_input": "(1, 784)",
                "expected_classes": str(len(self.labels)),
            }
            self.write_info(info)
            self._replace_active_model(candidate_model, self.model_path.stat().st_mtime, original_filename)
            print(f"[model-manager] model replaced: {self.model_path.name}", flush=True)
            print(f"[model-manager] active model updated: {original_filename}", flush=True)
            return info

    def activate_version(self, filename: str) -> dict[str, str]:
        requested = Path(filename)
        if requested.name != filename or requested.is_absolute() or any(part == ".." for part in requested.parts):
            raise ValueError("Invalid model version filename")
        if requested.suffix.lower() != ".joblib":
            raise ValueError("Only .joblib model versions are supported")

        self.ensure_dirs()
        version_path = (self.model_version_dir / requested.name).resolve()
        version_dir = self.model_version_dir.resolve()
        if version_path.parent != version_dir or not version_path.exists() or not version_path.is_file():
            raise FileNotFoundError("Model version not found")

        activated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replacement_path = self.model_path.with_name(f".{self.model_path.name}.{timestamp}.replace")

        with self._lock:
            candidate_model, validation = self._load_and_validate(version_path)
            print(
                f"[model-manager] version loaded: {requested.name} "
                f"input={validation['input_shape']} output={validation['output_shape']}",
                flush=True,
            )

            shutil.copy2(version_path, replacement_path)
            os.replace(replacement_path, self.model_path)

            self.model_name_path.write_text(requested.name, encoding="utf-8")
            previous_info = self.read_info()
            selected_metrics = self._metrics_by_model_file().get(requested.name, {})
            info = {
                "version_name": str(selected_metrics.get("model", requested.stem)),
                "original_filename": requested.name,
                "stored_version_file": requested.name,
                "active_model_file": self.model_path.name,
                "uploaded_at": selected_metrics.get("trained_at", previous_info.get("uploaded_at", activated_at)),
                "activated_at": activated_at,
                "validation_status": "activated from saved model version",
                "input_shape": validation["input_shape"],
                "output_shape": validation["output_shape"],
                "test_output_shape": validation["test_output_shape"],
                "expected_input": "(1, 784)",
                "expected_classes": str(len(self.labels)),
            }
            for key in ("accuracy", "precision", "recall", "f1_score", "f1-score", "confusion_matrix", "labels"):
                if key in selected_metrics:
                    info[key] = selected_metrics[key]
                elif key in previous_info:
                    info[key] = previous_info[key]
            self.write_info(info)
            self._replace_active_model(candidate_model, self.model_path.stat().st_mtime, requested.name)
            print(f"[model-manager] active version switched: {requested.name}", flush=True)
            return info

    def display_name(self) -> str:
        if self._active_name:
            return self._active_name
        if self.model_name_path.exists():
            name = self.model_name_path.read_text(encoding="utf-8").strip()
            if name:
                return name
        return self.model_path.name if self.model_path.exists() else "ยังไม่มี model.joblib"

    def updated_at(self) -> str:
        if not self.model_path.exists():
            return "-"
        return datetime.fromtimestamp(self.model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    def read_info(self) -> dict[str, Any]:
        if not self.model_info_path.exists():
            return {}
        try:
            return json.loads(self.model_info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def write_info(self, info: dict[str, Any]) -> None:
        self.model_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_versions(self, limit: int = 10) -> list[dict[str, str]]:
        self.ensure_dirs()
        active_version = str(self.read_info().get("stored_version_file", ""))
        metrics_by_file = self._metrics_by_model_file()
        versions = []
        for path in sorted(self.model_version_dir.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True):
            metrics = metrics_by_file.get(path.name, {})
            versions.append({
                "filename": path.name,
                "model_name": str(metrics.get("model_name") or path.stem.replace("_", " ").title()),
                "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": f"{path.stat().st_size / 1024:.1f}",
                "is_active": path.name == active_version,
                "accuracy": self._metric_text(metrics.get("accuracy")),
                "precision": self._metric_text(metrics.get("precision")),
                "recall": self._metric_text(metrics.get("recall")),
                "f1_score": self._metric_text(metrics.get("f1_score", metrics.get("f1-score"))),
                "trained_at": str(metrics.get("trained_at", "")),
            })
        return versions[:limit]

    def _metrics_by_model_file(self) -> dict[str, dict[str, Any]]:
        metrics_path = self.model_info_path.with_name("training_metrics.json")
        if not metrics_path.exists():
            return {}
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        results = payload.get("per_model_results") or payload.get("model_comparison") or []
        metrics = {}
        for item in results:
            if isinstance(item, dict) and item.get("model_file"):
                metrics[str(item["model_file"])] = item
        return metrics

    def _metric_text(self, value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _replace_active_model(self, model, mtime: float, active_name: str) -> None:
        previous_model = self._model
        self._model = model
        self._model_mtime = mtime
        self._active_name = active_name
        if previous_model is not None and previous_model is not model:
            del previous_model
            gc.collect()

    def _load_and_validate(self, path: Path):
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise ValueError("Uploaded object must provide a predict method")

        dummy = np.zeros((1, *self.expected_input_shape), dtype=np.float32)
        prediction = np.asarray(model.predict(dummy))
        if prediction.shape[0] != 1:
            raise ValueError(f"Model predict must return one row for input (1, 784), got {prediction.shape}")

        output = self._predict_probabilities(model, dummy)
        if output.shape != (1, len(self.labels)):
            raise ValueError(f"Model must return {len(self.labels)} classes, got shape {output.shape}")
        if not np.all(np.isfinite(output)):
            raise ValueError("Model returned NaN/Infinity during validation")

        classes = tuple(str(item) for item in getattr(model, "classes_", ()))
        if classes and set(classes) != set(self.labels):
            raise ValueError(f"Model classes must be {self.labels}, got {classes}")

        return model, {
            "input_shape": "(None, 784)",
            "output_shape": f"(None, {len(self.labels)})",
            "test_output_shape": str(tuple(output.shape)),
        }

    def _predict_probabilities(self, model, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim != 2 or sample.shape[1] != self.expected_input_shape[0]:
            raise ValueError(f"Expected input shape (n, 784), got {sample.shape}")

        if hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(sample), dtype=np.float32)
            classes = tuple(str(item) for item in getattr(model, "classes_", self.labels))
            probabilities = np.zeros((sample.shape[0], len(self.labels)), dtype=np.float32)
            for source_index, class_label in enumerate(classes):
                if class_label in self.labels and source_index < raw.shape[1]:
                    probabilities[:, self.labels.index(class_label)] = raw[:, source_index]
            return probabilities

        predictions = [str(item) for item in np.asarray(model.predict(sample)).reshape(-1)]
        probabilities = np.zeros((sample.shape[0], len(self.labels)), dtype=np.float32)
        for row_index, label in enumerate(predictions):
            if label not in self.labels:
                raise ValueError(f"Model predicted unsupported label {label!r}")
            probabilities[row_index, self.labels.index(label)] = 1.0
        return probabilities
