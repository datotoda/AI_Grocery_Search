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
    pass

    # @TODO

    # return cropped_image


def detect_objects(model: YOLO, image: Image, conf: float = 0.5, iou: float = 0.7) -> list:
    """
    Example Output:
    [
        {
            'bbox': (337, 119, 471, 270), 
            'confidence': 0.9207653403282166, 
            'class_id': 46, 
            'category': 'banana'
        }, 
        {
            'bbox': (460, 556, 560, 706), 
            'confidence': 0.7962380051612854, 
            'class_id': 39, 
            'category': 'bottle'
        }, 
        {
            'bbox': (290, 502, 341, 621), 
            'confidence': 0.4161379635334015, 
            'class_id': 39, 
            'category': 'bottle'
        }
    ]
    """

    # @TODO

    # return boxes_list


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

    # @TODO

    # return clip_model, preprocess, yolo_model


def calculate_image_embedding(model, preprocess, image_path):
    pass

    # @TODO

    # return embedding


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
    
    # @TODO scores, indices = index.search(query, top_k)

    results = []
    # for score, row_index in zip(scores[0], indices[0]):
    #     if row_index < 0:
    #         continue
    
    #     # @TODO

    #     results.append({
    #         "image_path": dataset_path,
    #         "dataset_filename": dataset_filename,
    #         "similarity": round(float(score), 4),
    #         "class_id": class_id,
    #         "class_name": class_name,
    #         "coarse_class_name": coarse_class_name,
    #         "description": description,
    #         "split": split,
    #     })

    return results
