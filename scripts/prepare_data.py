import csv
import json
import os
from pathlib import Path

import open_clip
import torch
from PIL import Image


# Configuration

MODEL_NAME = "ViT-B-32"
MODEL_PRETRAINED = "openai"
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "media"
PRODUCTS_FILE = DATASET_DIR / "products.json"
PRODUCT_IMAGES_FILE = DATASET_DIR / "product_images.json"


def load_models():
    print(f"Loading {MODEL_NAME}...")
    print(f"Device: {device}")

    # @TODO

    # return model, preprocess


def calculate_image_embedding(model, processor, image_path):
    print(f'Calculate embeding for {image_path}')

    # @TODO

    # return embedding


def load_json(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    model, processor = load_models()
    print(f"Dataset directory: {DATASET_DIR}")

    # Load existing data to support upserts
    products = load_json(PRODUCTS_FILE)
    product_images = load_json(PRODUCT_IMAGES_FILE)

    # Import product classes
    with open(DATASET_DIR / "classes.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            class_id = int(row["Class ID (int)"])

            description_path = DATASET_DIR / row["Product Description Path (str)"].lstrip("/")

            with open(description_path, encoding="utf-8") as desc_file:
                description = desc_file.read().strip()

            products[str(class_id)] = {
                "class_id": class_id,
                "class_name": row["Class Name (str)"],
                "coarse_class_id": int(row["Coarse Class ID (int)"]),
                "coarse_class_name": row["Coarse Class Name (str)"],
                "icon_path": "media" + row["Iconic Image Path (str)"],
                "description": description
            }

    processed = 0
    failed = 0

    # Import product images
    for file_name in ["train.txt", "val.txt", "test.txt"]:
        split = file_name.split(".")[0]

        with open(DATASET_DIR / file_name, encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                image_path = "media/" + row[0]
                
                try:
                    product_images[image_path] = {
                        "class_id": int(row[1]),
                        "coarse_class_id": int(row[2]),
                        "image_path": image_path,
                        "split": split,
                        "image_embbeding": calculate_image_embedding(model, processor, DATASET_DIR / row[0]).flatten().tolist()
                    }
                    processed += 1

                    print(
                        f"[{processed + failed}/{split}] "
                        f"Processed: {image_path}"
                    )
                    
                except Exception as error:
                    failed += 1

                    print(
                        f"[ERROR] {image_path}: {error}"
                    )

    # Save data
    save_json(PRODUCTS_FILE, products)
    save_json(PRODUCT_IMAGES_FILE, product_images)
    

    print("\nSeeding complete.")
    print(f"  Products:      {len(products)}")
    print(f"  ProductImages: {len(product_images)}")


if __name__ == "__main__":
    main()
