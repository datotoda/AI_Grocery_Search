import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from utils import (
    DATASET_DIR,
    UPLOAD_DIR,
    detect_objects,
    get_dataset_image_path,
    load_resources,
    crop_object,
    search_similar_images,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Load the models, metadata, and FAISS index once when the application starts.
print("Loading AI image search resources...")
resources = load_resources()
print("Resources loaded.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(image, extension):
    """Save a PIL image under a UUID filename and return that filename."""
    filename = f"{uuid.uuid4().hex}.{extension}"
    image.save(UPLOAD_DIR / filename)
    return filename


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/upload")
def upload():
    """Save an uploaded image and return its UUID filename and URL."""
    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({"error": "Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Supported formats: PNG, JPG, JPEG, WEBP."}), 400

    extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()

    try:
        image = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    filename = save_image(image, extension)

    return jsonify({
        "image_uuid": filename,
        "filename": filename,
        "image_url": url_for("uploaded_file", filename=filename),
    })


@app.post("/detect_objects")
def detect_objects_endpoint():
    """
    Detect every YOLO class in the uploaded image, crop each detection,
    save the crops, and return their metadata.
    """
    data = request.get_json(silent=True) or {}
    image_uuid = data.get("image_uuid")

    if not image_uuid:
        return jsonify({"error": "image_uuid is required."}), 400

    image_path = UPLOAD_DIR / Path(image_uuid).name
    if not image_path.is_file():
        return jsonify({"error": "Uploaded image was not found."}), 404

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Could not read the uploaded image."}), 400

    detections = detect_objects(
        resources["yolo_model"],
        image,
        conf=0.5,
        iou=0.7,
    )

    objects = []
    for detection in detections:
        cropped = crop_object(image, detection["bbox"])

        # Use the same image format as the original where possible.
        extension = image_path.suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            extension = "jpg"

        crop_uuid = save_image(cropped, extension)

        objects.append({
            "image_uuid": crop_uuid,
            "filename": crop_uuid,
            "image_url": url_for("uploaded_file", filename=crop_uuid),
            "confidence": round(detection["confidence"], 4),
            "category": detection["category"],
            "class_id": detection["class_id"],
            "bbox": list(detection["bbox"]),
        })

    return jsonify({
        "image_uuid": image_uuid,
        "count": len(objects),
        "objects": objects,
    })


@app.post("/search")
def search():
    """Search the FAISS image index using an uploaded image or detected crop UUID."""
    data = request.get_json(silent=True) or {}
    image_uuid = data.get("image_uuid")

    if not image_uuid:
        return jsonify({"error": "image_uuid is required."}), 400

    image_path = UPLOAD_DIR / Path(image_uuid).name
    if not image_path.is_file():
        return jsonify({"error": "Image was not found in uploads."}), 404

    try:
        results = search_similar_images(
            resources=resources,
            image_path=image_path,
            top_k=20,
        )
    except Exception as error:
        app.logger.exception("Image search failed")
        return jsonify({"error": f"Image search failed: {error}"}), 500

    for result in results:
        result["image_url"] = url_for(
            "dataset_file",
            filename=result.pop("dataset_filename"),
        )

    return jsonify({
        "query_image_uuid": image_uuid,
        "count": len(results),
        "results": results,
    })


@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/dataset/<path:filename>")
def dataset_file(filename):
    return send_from_directory(DATASET_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
