# Thai Digit AI 31-35

Flask web app for collecting 28x28 handwriting samples and predicting Thai digit labels `๓๑` to `๓๕`.

This project is PythonAnywhere free-tier friendly. It uses Flask, scikit-learn, Pillow, NumPy, and joblib only. There is no TensorFlow or Keras dependency.

## Features

- Draw and save dataset samples for labels `31`, `32`, `33`, `34`, `35`
- Train lightweight scikit-learn models from `dataset/31` through `dataset/35`
- Save all trained models into `model_versions/`
- Pick the active model from the Admin page
- Predict with the active `model.joblib`
- Show backend debug preview and per-label probabilities on the Predict page

## Routes

- `GET /` - draw and save dataset samples
- `GET /predict-page` - draw and test prediction
- `GET /admin` - admin login, model upload, model metrics, and version switching
- `POST /predict` - predict from a base64 image
- `POST /upload-model` - upload and validate a `.joblib` model
- `POST /activate-model-version` - activate a saved model from `model_versions/`
- `GET /download-dataset` - download the dataset ZIP

## Model Contract

- Active model file: `model.joblib`
- Model format: scikit-learn estimator saved with joblib
- Input shape: `(1, 784)`
- Input data: shared preprocessing output flattened to 784 normalized features
- Classes: `31`, `32`, `33`, `34`, `35`
- Returned Thai labels: `๓๑`, `๓๒`, `๓๓`, `๓๔`, `๓๕`

The app prefers classifiers with `predict_proba`. If a classifier only has `predict`, the API returns confidence `1.0` for the predicted class.

## Preprocessing

Training and web prediction both use [image_preprocessing.py](image_preprocessing.py).

The shared preprocessing does this:

- composite image on a white background
- convert to grayscale
- resize to `28x28`
- auto-invert only when the background is dark
- crop bounding box around ink
- resize digit to fit inside `20x20`
- paste centered on a `28x28` white canvas
- normalize pixels to `0..1`
- flatten to `784` features

This is important because the web canvas and dataset images must follow the same black-ink-on-white convention.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Dataset folders must exist here:

```text
dataset/
  31/
  32/
  33/
  34/
  35/
```

Each folder should contain PNG images for that label.

## Train

Run:

```powershell
python train.py
```

`train.py` trains these models:

- KNN
- Decision Tree
- Random Forest
- SVM RBF

Each trained model is saved:

```text
model_versions/knn.joblib
model_versions/decision_tree.joblib
model_versions/random_forest.joblib
model_versions/svm_rbf.joblib
```

The best model by validation accuracy is also copied to:

```text
model.joblib
```

Training also writes:

- `training_metrics.json` - selected model and per-model metrics
- `model_info.json` - active model metadata
- `confusion_matrix.png` - confusion matrix for the selected best model

Metrics include accuracy, precision, recall, f1-score, confusion matrix, model file, and trained timestamp.

## Admin

Open:

```text
http://127.0.0.1:5000/admin
```

Demo login:

```text
username: admin
password: 911
```

The Admin page can:

- upload a `.joblib` model
- validate model compatibility
- show model versions from `model_versions/`
- show accuracy, precision, recall, and f1-score for trained models
- activate any saved model version

Activation validates the selected model before replacing `model.joblib`. If validation fails, the current active model stays unchanged.

## Prediction Debug

Open:

```text
http://127.0.0.1:5000/predict-page
```

After prediction, the page shows:

- predicted Thai label
- confidence
- exact backend-processed `28x28` image sent to the model
- probabilities for labels `31`, `32`, `33`, `34`, `35`

The `/predict` JSON response includes:

- `prediction`
- `label`
- `confidence`
- `processed_image`
- `probabilities`

## Run Locally

```powershell
python main.py
```

Open:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/predict-page
http://127.0.0.1:5000/admin
```

## PythonAnywhere Notes

Install requirements inside your PythonAnywhere virtualenv:

```bash
pip install -r requirements.txt
```

No large deep-learning package is required. The active model is `model.joblib`, and all switchable models live in `model_versions/`.
