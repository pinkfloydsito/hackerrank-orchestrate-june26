"""Dataset preprocessor: convert Roboflow COCO to unified format."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from hackerrank_orchestrate.config import ISSUE_TYPES, OBJECT_PARTS, OBJECT_TYPES, PROCESSED_DATA_DIR
from hackerrank_orchestrate.data.roboflow_downloader import verify_downloads
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "bonnet-dent": ("dent", "hood"),
    "boot-dent": ("dent", "rear_bumper"),
    "doorouter-dent": ("dent", "door"),
    "scratch": ("scratch", "body"),
    "broken": ("broken_part", "screen"),
    "bezel_broken": ("broken_part", "screen"),
    "faded_key": ("stain", "keyboard"),
}


def heuristic_map(cat_name: str, object_type: str) -> Tuple[str, str]:
    """Infer (issue_type, object_part) from a Roboflow category name."""
    cat_name = cat_name.lower()

    # Issue type detection
    if any(w in cat_name for w in ["dent", "dented"]):
        issue = "dent"
    elif any(w in cat_name for w in ["scratch", "scratched"]):
        issue = "scratch"
    elif any(w in cat_name for w in ["crack", "cracked", "shatter"]):
        issue = "crack"
    elif any(w in cat_name for w in ["broken", "break", "damage"]):
        issue = "broken_part"
    elif any(w in cat_name for w in ["torn", "tear", "rip"]):
        issue = "torn_packaging"
    elif any(w in cat_name for w in ["crush", "crushed"]):
        issue = "crushed_packaging"
    elif any(w in cat_name for w in ["water", "wet"]):
        issue = "water_damage"
    elif any(w in cat_name for w in ["stain"]):
        issue = "stain"
    else:
        issue = "unknown"

    # Object part detection
    part = _infer_part(object_type, cat_name)
    return issue, part


def _infer_part(object_type: str, cat_name: str) -> str:
    """Infer object part from category name and object type."""
    if object_type == "car":
        if any(w in cat_name for w in ["bonnet", "hood"]):
            return "hood"
        elif any(w in cat_name for w in ["boot", "trunk", "rear"]):
            return "rear_bumper"
        elif any(w in cat_name for w in ["door", "side"]):
            return "door"
        elif any(w in cat_name for w in ["bumper", "front"]):
            return "front_bumper"
        elif any(w in cat_name for w in ["windshield", "glass"]):
            return "windshield"
        elif any(w in cat_name for w in ["mirror"]):
            return "side_mirror"
        elif any(w in cat_name for w in ["headlight"]):
            return "headlight"
        elif any(w in cat_name for w in ["taillight"]):
            return "taillight"
        else:
            return "body"
    elif object_type == "laptop":
        if any(w in cat_name for w in ["screen", "display", "bezel"]):
            return "screen"
        elif any(w in cat_name for w in ["keyboard", "key"]):
            return "keyboard"
        elif any(w in cat_name for w in ["trackpad", "touchpad"]):
            return "trackpad"
        elif any(w in cat_name for w in ["hinge"]):
            return "hinge"
        elif any(w in cat_name for w in ["lid", "corner"]):
            return "corner" if "corner" in cat_name else "lid"
        else:
            return "body"
    elif object_type == "package":
        if any(w in cat_name for w in ["corner"]):
            return "package_corner"
        elif any(w in cat_name for w in ["side"]):
            return "package_side"
        elif any(w in cat_name for w in ["seal", "tape"]):
            return "seal"
        elif any(w in cat_name for w in ["label"]):
            return "label"
        elif any(w in cat_name for w in ["content", "item"]):
            return "contents"
        else:
            return "box"
    else:
        return "unknown"


def load_coco_annotations(coco_path: Path) -> Dict[str, Any]:
    """Load COCO format annotations from JSON."""
    with open(coco_path, "r") as f:
        return json.load(f)


def create_label_mapping(coco_data: Dict[str, Any], object_type: str) -> Dict[int, Tuple[str, str]]:
    """Map COCO category IDs to (issue_type, object_part)."""
    mapping = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat["id"]
        cat_name = cat["name"].lower().strip()

        if cat_name in LABEL_MAP:
            issue, part = LABEL_MAP[cat_name]
        else:
            issue, part = heuristic_map(cat_name, object_type)

        mapping[cat_id] = (issue, part)
    return mapping


def process_dataset(raw_dir: Path, object_type: str) -> List[Dict[str, Any]]:
    """Process a single downloaded dataset into unified records."""
    logger.info(f"Processing {object_type} from {raw_dir}")

    coco_files = list(raw_dir.rglob("*_annotations.coco.json"))
    if not coco_files:
        logger.warning(f"No COCO annotations found in {raw_dir}")
        return []

    coco_path = coco_files[0]
    logger.info(f"Found annotations: {coco_path}")

    coco_data = load_coco_annotations(coco_path)
    label_mapping = create_label_mapping(coco_data, object_type)

    logger.info(
        f"Categories: {len(coco_data.get('categories', []))}, "
        f"Images: {len(coco_data.get('images', []))}, "
        f"Annotations: {len(coco_data.get('annotations', []))}"
    )

    images_by_id = {img["id"]: img for img in coco_data.get("images", [])}
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    records = []
    output_img_dir = PROCESSED_DATA_DIR / "images" / object_type
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_data in tqdm(images_by_id.items(), desc=f"Processing {object_type}"):
        img_path = raw_dir / img_data["file_name"]
        if not img_path.exists():
            img_path = coco_path.parent / img_data["file_name"]

        if not img_path.exists():
            logger.warning(f"Image not found: {img_data['file_name']}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
        except Exception as e:
            logger.warning(f"Cannot read {img_path}: {e}")
            continue

        anns = annotations_by_image.get(img_id, [])
        if not anns:
            issue_type = "none"
            object_part = "unknown"
            has_damage = False
        else:
            ann = anns[0]
            cat_id = ann["category_id"]
            issue_type, object_part = label_mapping.get(cat_id, ("unknown", "unknown"))
            has_damage = issue_type != "none"

        # Save processed image
        output_img_path = output_img_dir / f"{img_id:06d}.jpg"
        img.save(output_img_path, quality=90)

        # Bounding box
        bbox = None
        if anns and "bbox" in anns[0]:
            x, y, w, h = anns[0]["bbox"]
            bbox = [x, y, x + w, y + h]

        record = {
            "image_id": img_id,
            "image_path": str(output_img_path),
            "original_path": str(img_path),
            "object_type": object_type,
            "issue_type": issue_type,
            "object_part": object_part,
            "has_damage": has_damage,
            "width": width,
            "height": height,
            "bbox": bbox,
            "num_annotations": len(anns),
        }
        records.append(record)

    logger.info(f"Processed {len(records)} images for {object_type}")
    return records


def split_dataset(
    records: List[Dict[str, Any]],
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test with stratification."""
    labels = [r["object_type"] for r in records]
    train_val, test = train_test_split(records, test_size=test_size, random_state=42, stratify=labels)
    val_ratio = val_size / (1 - test_size)
    train_labels = [r["object_type"] for r in train_val]
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42, stratify=train_labels)
    return train, val, test


def main() -> None:
    """CLI entry point for preprocessing."""
    from hackerrank_orchestrate.config import RAW_DATA_DIR

    verify_downloads()
    all_records = []

    for object_type in OBJECT_TYPES:
        dataset_dir = RAW_DATA_DIR / object_type
        if not dataset_dir.exists():
            logger.warning(f"Directory not found: {dataset_dir}")
            continue

        records = process_dataset(dataset_dir, object_type)
        all_records.extend(records)

    if not all_records:
        logger.error("No records processed. Check data/raw/.")
        return

    logger.info(f"Total records: {len(all_records)}")
    logger.info(f"Object types: {set(r['object_type'] for r in all_records)}")
    logger.info(f"Issue types: {set(r['issue_type'] for r in all_records)}")

    train, val, test = split_dataset(all_records)
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_path = PROCESSED_DATA_DIR / f"{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} to {split_path}")

    metadata = {
        "total_records": len(all_records),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "object_types": list(set(r["object_type"] for r in all_records)),
        "issue_types": list(set(r["issue_type"] for r in all_records)),
        "object_parts": {
            obj_type: list(set(r["object_part"] for r in all_records if r["object_type"] == obj_type))
            for obj_type in OBJECT_TYPES
        },
    }
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
