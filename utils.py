import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "media"
UPLOAD_DIR = BASE_DIR / "uploads"

PRODUCTS_FILE = DATASET_DIR / "products.json"
PRODUCT_IMAGES_FILE = DATASET_DIR / "product_images.json"
FAISS_INDEX_FILE = BASE_DIR / "image.index"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
YOLO_MODEL_NAME = "yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def crop_object(image: Image, bbox: tuple) -> Image:
    x1, y1, x2, y2 = bbox

    cropped_image = image.crop(
        (x1, y1, x2, y2)
    )

    return cropped_image


def detect_objects(model: YOLO, image: Image, conf: float = 0.5, iou: float = 0.7) -> list:
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False
    )

    boxes_list = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            boxes_list.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'category': model.names[int(box.cls[0])]
            })

    return boxes_list


# ---------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}")

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_models():
    print(f"Loading CLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) on {DEVICE}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    clip_model.to(DEVICE)
    clip_model.eval()

    print(f"Loading YOLO model: {YOLO_MODEL_NAME}...")
    yolo_model = YOLO(YOLO_MODEL_NAME)

    return clip_model, preprocess, yolo_model


def calculate_image_embedding(model, preprocess, image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype("float32")

    return embedding


def normalize_embeddings(embeddings):
    """Return a contiguous float32 matrix with every vector L2-normalized."""
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    return np.ascontiguousarray(embeddings)


def build_or_load_image_index(product_images):
    """
    Load image.index if it exists. Otherwise build it from the embeddings
    stored in product_images.json and save it.

    FAISS row positions correspond to the insertion order of product_images.json.
    Keep that order unchanged when using an existing index.
    """
    image_paths = list(product_images.keys())

    if FAISS_INDEX_FILE.exists():
        print(f"Loading FAISS index: {FAISS_INDEX_FILE}")
        index = faiss.read_index(str(FAISS_INDEX_FILE))

        if index.ntotal != len(image_paths):
            raise ValueError(
                "FAISS index and product_images.json contain different numbers "
                f"of rows (index: {index.ntotal}, metadata: {len(image_paths)}). "
                "Rebuild the index using the same product_images.json ordering."
            )

        return index, image_paths

    print("FAISS index not found. Building it from product_images.json...")
    vectors = []

    for image_path in image_paths:
        embedding = product_images[image_path].get("image_embbeding")
        if embedding is None:
            raise ValueError(
                f"Missing image_embbeding for {image_path}. "
                "Run your dataset preparation script first."
            )
        vectors.append(embedding)

    if not vectors:
        raise ValueError("product_images.json does not contain any image embeddings.")

    matrix = normalize_embeddings(np.asarray(vectors, dtype="float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"Saved FAISS index to {FAISS_INDEX_FILE}")

    return index, image_paths


def load_resources():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}. "
            "Set DATASET_DIR to the directory containing products.json "
            "and product_images.json."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    products = load_json(PRODUCTS_FILE)
    product_images = load_json(PRODUCT_IMAGES_FILE)

    clip_model, preprocess, yolo_model = load_models()
    image_index, image_paths = build_or_load_image_index(product_images)

    return {
        "clip_model": clip_model,
        "preprocess": preprocess,
        "yolo_model": yolo_model,
        "products": products,
        "product_images": product_images,
        "image_index": image_index,
        "image_paths": image_paths,
    }


# ---------------------------------------------------------------------
# Dataset paths and similarity search
# ---------------------------------------------------------------------


def search_similar_images(resources, image_path, top_k=20):
    query = calculate_image_embedding(
        resources["clip_model"],
        resources["preprocess"],
        image_path,
    )
    query = normalize_embeddings(query)

    index = resources["image_index"]
    scores, indices = index.search(query, top_k)

    results = []
    for score, row_index in zip(scores[0], indices[0]):
        if row_index < 0:
            continue

        dataset_path = resources["image_paths"][int(row_index)]
        image_metadata = resources["product_images"].get(dataset_path, {})

        class_id = image_metadata.get("class_id")
        product = resources["products"].get(str(class_id), {})

        full_path = get_dataset_image_path(dataset_path)
        if not full_path.is_file():
            # Keep the result visible in the response, but the image URL
            # will not be useful if the underlying dataset file is missing.
            continue

        results.append({
            "image_path": dataset_path,
            "dataset_filename": full_path.relative_to(DATASET_DIR).as_posix(),
            "similarity": round(float(score), 4),
            "class_id": class_id,
            "class_name": product.get("class_name", "Unknown"),
            "coarse_class_name": product.get("coarse_class_name", "Unknown"),
            "description": product.get("description", ""),
            "split": image_metadata.get("split"),
        })

    return results
