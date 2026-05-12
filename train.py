import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_preprocessing import center_digit, image_to_feature_vector, shift_with_white_fill, to_grayscale_28


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model.joblib"
MODEL_NAME_PATH = BASE_DIR / ".model_name"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
MODEL_VERSION_DIR = BASE_DIR / "model_versions"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
TRAINING_METRICS_PATH = BASE_DIR / "training_metrics.json"
LABELS = ("31", "32", "33", "34", "35")
THAI_LABELS = {
    "31": "๓๑",
    "32": "๓๒",
    "33": "๓๓",
    "34": "๓๔",
    "35": "๓๕",
}
RANDOM_SEED = 42
MODEL_DISPLAY_NAMES = {
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM RBF",
}
AUGMENTATIONS = (
    ("original", 0, 0, 0),
    ("rotate_minus_5", -5, 0, 0),
    ("rotate_plus_5", 5, 0, 0),
    ("shift_left_2", 0, -2, 0),
    ("shift_right_2", 0, 2, 0),
    ("shift_up_2", 0, 0, -2),
    ("shift_down_2", 0, 0, 2),
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_source_images() -> tuple[list[Image.Image], np.ndarray]:
    images = []
    labels = []
    skipped = []

    for label in LABELS:
        label_dir = DATASET_DIR / label
        if not label_dir.exists():
            raise RuntimeError(f"Missing dataset folder: {label_dir}")
        for image_path in sorted(label_dir.glob("*.png")):
            try:
                with Image.open(image_path) as image:
                    images.append(to_grayscale_28(image))
                labels.append(label)
            except (OSError, UnidentifiedImageError) as exc:
                skipped.append(f"{image_path}: {exc}")

    if skipped:
        print("Skipped unreadable images:")
        for item in skipped[:10]:
            print(f"- {item}")
    if not images:
        raise RuntimeError("No usable PNG images found in dataset/31 to dataset/35.")

    y = np.asarray(labels, dtype=str)
    for label in LABELS:
        count = int(np.sum(y == label))
        if count < 5:
            raise RuntimeError(f"Class {label} needs at least 5 images for stratified training and CV.")

    return images, y


def augment_image(image: Image.Image, angle: int, shift_x: int, shift_y: int) -> Image.Image:
    augmented = image
    if angle:
        augmented = augmented.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if shift_x or shift_y:
        augmented = shift_with_white_fill(augmented, shift_x, shift_y)
    return center_digit(augmented)


def images_to_features(images: list[Image.Image]) -> np.ndarray:
    return np.asarray([image_to_feature_vector(image) for image in images], dtype=np.float32)


def build_augmented_training_set(images: list[Image.Image], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = []
    augmented_labels = []
    for image, label in zip(images, labels):
        for _name, angle, shift_x, shift_y in AUGMENTATIONS:
            features.append(image_to_feature_vector(augment_image(image, angle, shift_x, shift_y)))
            augmented_labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(augmented_labels, dtype=str)


def build_candidates() -> dict[str, object]:
    return {
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski",
                p=2,
            ),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=50.0,
                gamma="auto",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_SEED,
            ),
        ),
    }


def validate_model(model) -> None:
    dummy = np.zeros((1, 784), dtype=np.float32)
    prediction = np.asarray(model.predict(dummy))
    if prediction.shape[0] != 1:
        raise RuntimeError("Model validation failed: predict((1, 784)) did not return one row.")

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(dummy))
        if probabilities.shape != (1, len(LABELS)):
            raise RuntimeError(
                f"Model validation failed: predict_proba returned {probabilities.shape}, expected (1, {len(LABELS)})."
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Model validation failed: predict_proba returned NaN/Infinity.")


def evaluate_model(model, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, object]:
    y_pred = model.predict(x_val)
    report = classification_report(
        y_val,
        y_pred,
        labels=list(LABELS),
        target_names=[THAI_LABELS[label] for label in LABELS],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_val, y_pred, labels=list(LABELS))
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": report,
    }


def save_confusion_matrix(matrix: list[list[int]]) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=list(LABELS))
    display.plot(cmap="Blues", values_format="d")
    plt.title("Thai Digit Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def model_classes(model) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    return [str(item) for item in classes] if classes is not None else list(LABELS)


def write_metrics_json(
    selected_name: str,
    selected_metrics: dict[str, object],
    per_model_results: list[dict[str, object]],
    selected_model_file: str,
    train_count: int,
    augmented_count: int,
    validation_count: int,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_payload = {
        "version_name": selected_name,
        "original_filename": "generated by train.py",
        "stored_version_file": selected_model_file,
        "active_model_file": MODEL_PATH.name,
        "uploaded_at": now,
        "activated_at": now,
        "validation_status": "trained all candidate models, selected by validation accuracy",
        "input_shape": "(None, 784)",
        "output_shape": f"(None, {len(LABELS)})",
        "test_output_shape": f"(1, {len(LABELS)})",
        "expected_input": "(1, 784)",
        "expected_classes": str(len(LABELS)),
        "preprocessing": "grayscale, center digit, normalize 0-1, flatten to 784",
        "augmentation": [name for name, _angle, _shift_x, _shift_y in AUGMENTATIONS],
        "train_samples_before_augmentation": train_count,
        "train_samples_after_augmentation": augmented_count,
        "validation_samples": validation_count,
        "selected_model": selected_name,
        "selected_model_file": selected_model_file,
        "accuracy": f"{selected_metrics['accuracy']:.4f}",
        "precision": f"{selected_metrics['precision']:.4f}",
        "recall": f"{selected_metrics['recall']:.4f}",
        "f1_score": f"{selected_metrics['f1_score']:.4f}",
        "f1-score": f"{selected_metrics['f1_score']:.4f}",
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "labels": list(LABELS),
        "per_model_results": per_model_results,
        "model_comparison": per_model_results,
    }
    text = json.dumps(metrics_payload, ensure_ascii=False, indent=2)
    MODEL_INFO_PATH.write_text(text, encoding="utf-8")
    TRAINING_METRICS_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    MODEL_VERSION_DIR.mkdir(parents=True, exist_ok=True)
    source_images, y = load_source_images()
    print(f"Loaded {len(source_images)} source images")
    for label in LABELS:
        print(f"- {THAI_LABELS[label]} ({label}): {int(np.sum(y == label))}")

    train_indices, val_indices = train_test_split(
        np.arange(len(source_images)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    train_images = [source_images[index] for index in train_indices]
    val_images = [source_images[index] for index in val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]

    x_train, y_train_augmented = build_augmented_training_set(train_images, y_train)
    x_val = images_to_features(val_images)

    print(f"Training samples: {len(train_images)} source -> {len(x_train)} augmented")
    print(f"Validation samples: {len(x_val)}")

    candidates = build_candidates()
    per_model_results = []
    fitted_models = {}

    print("\nTraining lightweight scikit-learn models")
    for name, model in candidates.items():
        model.fit(x_train, y_train_augmented)
        validate_model(model)
        metrics = evaluate_model(model, x_val, y_val)
        fitted_models[name] = model
        model_file = f"{name}.joblib"
        model_path = MODEL_VERSION_DIR / model_file
        joblib.dump(model, model_path, compress=3)
        row = {
            "model": name,
            "model_name": MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
            "model_file": model_file,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "f1-score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"],
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        per_model_results.append(row)
        print(
            f"- {name}: acc={row['accuracy']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1_score']:.4f}, saved={model_path}"
        )

    per_model_results.sort(key=lambda item: (item["accuracy"], item["f1_score"]), reverse=True)
    selected_name = str(per_model_results[0]["model"])
    selected_model_file = str(per_model_results[0]["model_file"])
    selected_model = fitted_models[selected_name]
    selected_metrics = evaluate_model(selected_model, x_val, y_val)

    shutil.copy2(MODEL_VERSION_DIR / selected_model_file, MODEL_PATH)
    MODEL_NAME_PATH.write_text(selected_model_file, encoding="utf-8")
    save_confusion_matrix(selected_metrics["confusion_matrix"])
    write_metrics_json(
        selected_name,
        selected_metrics,
        per_model_results,
        selected_model_file,
        train_count=len(train_images),
        augmented_count=len(x_train),
        validation_count=len(x_val),
    )

    print(f"\nSelected model: {selected_name}")
    print(f"Classes: {model_classes(selected_model)}")
    print(f"Accuracy: {selected_metrics['accuracy']:.4f}")
    print(f"Precision: {selected_metrics['precision']:.4f}")
    print(f"Recall: {selected_metrics['recall']:.4f}")
    print(f"F1-score: {selected_metrics['f1_score']:.4f}")
    print(f"Confusion matrix: {selected_metrics['confusion_matrix']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {TRAINING_METRICS_PATH}")
    print(f"Saved model versions: {MODEL_VERSION_DIR}")
    print(f"Saved confusion matrix image: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
