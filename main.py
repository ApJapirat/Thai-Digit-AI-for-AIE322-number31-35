import base64
import csv
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from PIL import Image, UnidentifiedImageError

from image_preprocessing import image_to_preview_data_url, image_to_feature_vector, preprocess_image, to_grayscale_28
from model_manager import ModelManager


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
METADATA_PATH = DATASET_DIR / "metadata.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
MODEL_NAME_PATH = BASE_DIR / ".model_name"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
MODEL_VERSION_DIR = BASE_DIR / "model_versions"
LABELS = ("31", "32", "33", "34", "35")
THAI_LABELS = {
    "31": "๓๑",
    "32": "๓๒",
    "33": "๓๓",
    "34": "๓๔",
    "35": "๓๕",
}
METADATA_COLUMNS = ("filename", "label", "contributor", "saved_at")
MIN_INK_PIXELS_28 = 8
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "911"

app = Flask(__name__)
app.config["SECRET_KEY"] = "thai-digit-assignment-demo"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

model_manager = ModelManager(
    model_path=MODEL_PATH,
    model_name_path=MODEL_NAME_PATH,
    model_info_path=MODEL_INFO_PATH,
    model_version_dir=MODEL_VERSION_DIR,
    labels=LABELS,
)


def ensure_model_dirs() -> None:
    model_manager.ensure_dirs()


def ensure_dataset_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)
    if not METADATA_PATH.exists() or METADATA_PATH.stat().st_size == 0:
        with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
            writer.writeheader()


def counts_by_label() -> dict[str, int]:
    ensure_dataset_dirs()
    return {label: len(list((DATASET_DIR / label).glob("*.png"))) for label in LABELS}


def dataset_audit(limit: int = 10) -> dict[str, object]:
    ensure_dataset_dirs()
    issues = []
    total = 0
    for label in LABELS:
        for path in sorted((DATASET_DIR / label).glob("*.png")):
            total += 1
            try:
                with Image.open(path) as image:
                    if image.size != (28, 28):
                        issues.append(f"{label}/{path.name}: size {image.size}, expected 28x28")
                    image.convert("L")
            except (OSError, UnidentifiedImageError) as exc:
                issues.append(f"{label}/{path.name}: cannot read image ({exc})")
            if len(issues) >= limit:
                break
    return {
        "total_images": total,
        "issue_count": len(issues),
        "issues": issues,
        "ok": len(issues) == 0,
    }


def read_metadata() -> list[dict[str, str]]:
    ensure_dataset_dirs()
    with METADATA_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_metadata(row: dict[str, str]) -> None:
    ensure_dataset_dirs()
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writerow(row)


def write_metadata(rows: list[dict[str, str]]) -> None:
    ensure_dataset_dirs()
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def next_filename(label: str) -> Path:
    label_dir = DATASET_DIR / label
    existing_numbers = []
    for path in label_dir.glob(f"{label}_*.png"):
        match = re.fullmatch(rf"{re.escape(label)}_(\d+)\.png", path.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    return label_dir / f"{label}_{next_number:03d}.png"


def error_response(message: str, status_code: int):
    response = jsonify({"detail": message})
    response.status_code = status_code
    return response


def decode_image(image_data: str) -> Image.Image:
    if not image_data:
        abort(error_response("Image is required", 400))
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        abort(error_response("รูปภาพไม่ถูกต้อง กรุณาส่ง base64 PNG/JPEG ที่เปิดอ่านได้", 400))
    return image


def image_to_grayscale_28(image: Image.Image, invert: bool) -> Image.Image:
    grayscale = to_grayscale_28(image)
    if invert:
        pixels = 255 - np.asarray(grayscale, dtype=np.uint8)
        return Image.fromarray(pixels, mode="L")
    return grayscale


def normalize_sample_image(image: Image.Image) -> Image.Image:
    return image_to_grayscale_28(image, invert=False)


def preprocess_for_model(image: Image.Image) -> np.ndarray:
    return image_to_feature_vector(image).reshape(1, 784)


def preprocess_for_model_debug(image: Image.Image) -> tuple[np.ndarray, str]:
    features, processed_image = preprocess_image(image)
    return features.reshape(1, 784), image_to_preview_data_url(processed_image)


def ink_pixel_count(image: Image.Image) -> int:
    grayscale = image.convert("L")
    return sum(1 for value in grayscale.getdata() if value < 245)


@app.route("/", methods=["GET"])
def index():
    ensure_dataset_dirs()
    return render_template("index.html", labels=LABELS, thai_labels=THAI_LABELS)


@app.route("/predict-page", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/admin", methods=["GET"])
def admin():
    if not session.get("admin_logged_in"):
        return render_template("admin.html", logged_in=False)
    return render_template(
        "admin.html",
        logged_in=True,
        model_exists=MODEL_PATH.exists(),
        model_filename=model_manager.display_name(),
        model_updated=model_manager.updated_at(),
        model_info=model_manager.read_info(),
        model_versions=model_manager.list_versions(),
        counts=counts_by_label(),
        dataset_audit=dataset_audit(),
        thai_labels=THAI_LABELS,
    )


@app.route("/admin-login", methods=["POST"])
def admin_login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        flash("เข้าสู่ระบบสำเร็จ")
        return redirect(url_for("admin"))
    flash("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    return redirect(url_for("admin"))


@app.route("/admin-logout", methods=["GET"])
def admin_logout():
    session.pop("admin_logged_in", None)
    flash("ออกจากระบบแล้ว")
    return redirect(url_for("admin"))


@app.route("/upload-model", methods=["POST"])
def upload_model():
    if not session.get("admin_logged_in"):
        flash("กรุณาเข้าสู่ระบบก่อน")
        return redirect(url_for("admin"))

    upload = request.files.get("model")
    version_name = request.form.get("version_name", "").strip()
    if upload is None or upload.filename == "":
        flash("กรุณาเลือกไฟล์โมเดล .joblib")
        return redirect(url_for("admin"))
    if not upload.filename.lower().endswith(".joblib"):
        flash("รองรับเฉพาะไฟล์ .joblib เท่านั้น")
        return redirect(url_for("admin"))

    ensure_model_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = BASE_DIR / f".upload_check_{timestamp}.joblib"

    try:
        upload.save(temp_path)
        model_manager.activate_uploaded_model(temp_path, upload.filename, version_name)
        flash("อัปโหลดสำเร็จ ตรวจโมเดลผ่าน เปลี่ยน model.joblib และ reload เข้า memory แล้ว")
    except Exception as exc:
        flash(f"อัปโหลดไม่สำเร็จ: โมเดลนี้ใช้ไม่ได้ ({exc})")
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return redirect(url_for("admin"))


@app.route("/activate-model-version", methods=["POST"])
def activate_model_version():
    if not session.get("admin_logged_in"):
        flash("กรุณาเข้าสู่ระบบก่อน")
        return redirect(url_for("admin"))

    filename = request.form.get("filename", "").strip()
    if not filename:
        flash("กรุณาเลือก model version")
        return redirect(url_for("admin"))

    try:
        model_manager.activate_version(filename)
        flash(f"Activated model version: {filename}")
    except Exception as exc:
        flash(f"Activate ไม่สำเร็จ: {exc}")
    return redirect(url_for("admin"))


@app.route("/save-sample", methods=["POST"])
def save_sample():
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", ""))
    image_data = str(payload.get("image", ""))
    contributor = str(payload.get("contributor", "")).strip() or "anonymous"

    if label not in LABELS:
        return error_response("Label must be one of 31, 32, 33, 34, 35", 400)
    if not image_data:
        return error_response("Image is required", 400)

    output_path = next_filename(label)
    image = normalize_sample_image(decode_image(image_data))
    if ink_pixel_count(image) < MIN_INK_PIXELS_28:
        return error_response("กรุณาเขียนตัวเลขให้ชัดเจนก่อนบันทึก", 400)

    image.save(output_path, format="PNG")
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_metadata({
        "filename": output_path.name,
        "label": label,
        "contributor": contributor,
        "saved_at": saved_at,
    })
    return jsonify({
        "ok": True,
        "filename": output_path.name,
        "label": label,
        "saved_at": saved_at,
        "counts": counts_by_label(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = str(payload.get("image", ""))
    if not image_data:
        return error_response("Image is required", 400)

    image = decode_image(image_data)
    sample, processed_image = preprocess_for_model_debug(image)
    try:
        probabilities = model_manager.predict(sample)[0]
    except FileNotFoundError:
        return error_response("ยังไม่พบ model.joblib กรุณาเทรนหรืออัปโหลดโมเดลก่อน", 503)
    except Exception as exc:
        return error_response(f"ทำนายไม่สำเร็จ ({exc})", 500)
    if probabilities.shape[0] != len(LABELS):
        return error_response("โมเดลมีจำนวน class ไม่ตรงกับระบบ", 500)

    best_index = int(np.argmax(probabilities))
    label = LABELS[best_index]
    confidence = float(probabilities[best_index])
    return jsonify({
        "prediction": THAI_LABELS[label],
        "label": label,
        "confidence": round(confidence, 4),
        "processed_image": processed_image,
        "probabilities": {
            item_label: {
                "label": item_label,
                "thai_label": THAI_LABELS[item_label],
                "probability": round(float(probabilities[index]), 4),
            }
            for index, item_label in enumerate(LABELS)
        },
    })


@app.route("/delete-sample", methods=["POST"])
def delete_sample():
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", ""))
    filename = str(payload.get("filename", ""))
    if label not in LABELS:
        return error_response("Label must be one of 31, 32, 33, 34, 35", 400)
    if not re.fullmatch(rf"{re.escape(label)}_\d+\.png", filename):
        return error_response("Invalid filename", 400)

    path = DATASET_DIR / label / filename
    if not path.exists():
        return error_response("File not found", 404)

    path.unlink()
    rows = read_metadata()
    filtered_rows = [
        row for row in rows
        if not (row.get("label") == label and row.get("filename") == filename)
    ]
    write_metadata(filtered_rows)
    return jsonify({"ok": True, "deleted": filename, "counts": counts_by_label()})


@app.route("/counts", methods=["GET"])
def get_counts():
    return jsonify(counts_by_label())


@app.route("/stats", methods=["GET"])
def get_stats():
    rows = read_metadata()
    label_counts = counts_by_label()
    recent = []
    for row in rows[-5:][::-1]:
        label = row.get("label", "")
        filename = row.get("filename", "")
        if label in LABELS and filename:
            recent.append({
                "filename": filename,
                "label": label,
                "saved_at": row.get("saved_at", ""),
                "image_url": f"/sample/{label}/{filename}",
            })
    return jsonify({
        "total_images": sum(label_counts.values()),
        "counts_per_label": label_counts,
        "recent_samples": recent,
    })


@app.route("/sample/<label>/<filename>", methods=["GET"])
def get_sample(label: str, filename: str):
    if label not in LABELS:
        abort(404)
    if not re.fullmatch(rf"{re.escape(label)}_\d+\.png", filename):
        abort(404)
    path = DATASET_DIR / label / filename
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/download-dataset", methods=["GET"])
def download_dataset():
    ensure_dataset_dirs()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for label in LABELS:
            zip_file.writestr(f"dataset/{label}/", "")
        for path in DATASET_DIR.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(BASE_DIR))
    archive.seek(0)
    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name="thai_digits_dataset.zip",
    )


@app.errorhandler(413)
def file_too_large(_error):
    return error_response("ไฟล์ใหญ่เกินไป จำกัด 50 MB", 413)


ensure_dataset_dirs()
ensure_model_dirs()


if __name__ == "__main__":
    app.run(debug=True)
