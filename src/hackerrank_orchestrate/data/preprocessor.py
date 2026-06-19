"""Dataset preprocessor with embedding-based label normalization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from hackerrank_orchestrate.config import ISSUE_TYPES, OBJECT_PARTS, OBJECT_TYPES, PROCESSED_DATA_DIR
from hackerrank_orchestrate.data.roboflow_downloader import verify_downloads
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class LabelNormalizer:
    """Normalize arbitrary labels to canonical schema using embeddings."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Build canonical label sets
        self.canonical_issues = ISSUE_TYPES
        self.canonical_parts = {}
        for obj_type in OBJECT_TYPES:
            self.canonical_parts[obj_type] = OBJECT_PARTS[obj_type]
        
        # Pre-compute embeddings for canonical labels
        logger.info("Computing canonical label embeddings...")
        self.issue_embeddings = self.encoder.encode(
            self.canonical_issues, 
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        self.part_embeddings = {}
        for obj_type in OBJECT_TYPES:
            self.part_embeddings[obj_type] = self.encoder.encode(
                self.canonical_parts[obj_type],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        
        logger.info("Label normalizer ready")
    
    def normalize(self, raw_label: str, object_type: str) -> Tuple[str, str]:
        """Map raw label to canonical (issue_type, object_part) via embeddings.
        
        Args:
            raw_label: The raw category name from the dataset
            object_type: One of car, laptop, package
            
        Returns:
            Tuple of (issue_type, object_part)
        """
        # Encode the raw label
        label_embedding = self.encoder.encode(
            [raw_label],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        # Find nearest issue type
        issue_sims = cosine_similarity(label_embedding, self.issue_embeddings)[0]
        best_issue_idx = int(np.argmax(issue_sims))
        issue_type = self.canonical_issues[best_issue_idx]
        issue_confidence = float(issue_sims[best_issue_idx])
        
        # Find nearest object part for this object type
        part_sims = cosine_similarity(label_embedding, self.part_embeddings[object_type])[0]
        best_part_idx = int(np.argmax(part_sims))
        object_part = self.canonical_parts[object_type][best_part_idx]
        part_confidence = float(part_sims[best_part_idx])
        
        # Log low confidence matches
        if issue_confidence < 0.3 or part_confidence < 0.3:
            logger.warning(
                f"Low confidence for '{raw_label}' -> "
                f"issue={issue_type} ({issue_confidence:.3f}), "
                f"part={object_part} ({part_confidence:.3f})"
            )
        
        return issue_type, object_part


def load_coco_annotations(coco_path: Path) -> Dict[str, Any]:
    """Load COCO format annotations from JSON."""
    with open(coco_path, "r") as f:
        return json.load(f)


def create_label_mapping(
    coco_data: Dict[str, Any], 
    object_type: str,
    normalizer: LabelNormalizer,
) -> Dict[int, Tuple[str, str]]:
    """Map COCO category IDs to (issue_type, object_part) using embeddings."""
    mapping = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat["id"]
        cat_name = cat["name"].strip()
        
        issue, part = normalizer.normalize(cat_name, object_type)
        mapping[cat_id] = (issue, part)
        
        logger.info(f"  {cat_name} -> issue={issue}, part={part}")
    
    return mapping


def process_dataset(
    raw_dir: Path, 
    object_type: str,
    normalizer: LabelNormalizer,
) -> List[Dict[str, Any]]:
    """Process a single downloaded dataset into unified records."""
    logger.info(f"Processing {object_type} from {raw_dir}")
    
    # Find all COCO annotation files (train/val/test)
    coco_files = list(raw_dir.rglob("*_annotations.coco.json"))
    if not coco_files:
        logger.warning(f"No COCO annotations found in {raw_dir}")
        return []
    
    logger.info(f"Found {len(coco_files)} annotation files")
    
    # Use train annotations for label mapping (all splits should have same categories)
    train_coco = load_coco_annotations(coco_files[0])
    label_mapping = create_label_mapping(train_coco, object_type, normalizer)
    
    records = []
    output_img_dir = PROCESSED_DATA_DIR / "images" / object_type
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    img_id_counter = 0
    
    # Process all splits
    for coco_path in coco_files:
        coco_data = load_coco_annotations(coco_path)
        
        images_by_id = {img["id"]: img for img in coco_data.get("images", [])}
        annotations_by_image: Dict[int, List[Dict]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            annotations_by_image.setdefault(img_id, []).append(ann)
        
        logger.info(f"  {coco_path.name}: {len(images_by_id)} images, {len(coco_data.get('annotations', []))} annotations")
        
        for img_id, img_data in tqdm(images_by_id.items(), desc=f"Processing {object_type} {coco_path.parent.name}"):
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
            
            # Save processed image with unique ID
            output_img_path = output_img_dir / f"{img_id_counter:06d}.jpg"
            img.save(output_img_path, quality=90)
            img_id_counter += 1
            
            # Get annotations
            anns = annotations_by_image.get(img_id, [])
            if not anns:
                # No annotations - this might be a background image or no damage
                issue_type = "none"
                object_part = "unknown"
                has_damage = False
                bbox = None
            else:
                # Use the first annotation's category (primary damage)
                ann = anns[0]
                cat_id = ann["category_id"]
                issue_type, object_part = label_mapping.get(cat_id, ("unknown", "unknown"))
                has_damage = issue_type != "none"
                
                # Get bounding box if available
                bbox = None
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    bbox = [x, y, x + w, y + h]
            
            record = {
                "image_id": img_id_counter - 1,
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
                "split": coco_path.parent.name,  # train/val/test
            }
            records.append(record)
    
    logger.info(f"Processed {len(records)} images for {object_type}")
    return records


def split_dataset(
    records: List[Dict[str, Any]],
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test with stratification."""
    # Create stratification labels combining object_type and issue_type
    labels = [f"{r['object_type']}_{r['issue_type']}" for r in records]
    
    # First split: train vs (val+test)
    train_val_test, test = train_test_split(
        records, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_labels = [f"{r['object_type']}_{r['issue_type']}" for r in train_val_test]
    train, val = train_test_split(
        train_val_test, test_size=val_ratio, random_state=42, stratify=train_labels
    )
    
    return train, val, test


def main() -> None:
    """CLI entry point for preprocessing."""
    from hackerrank_orchestrate.config import RAW_DATA_DIR
    
    verify_downloads()
    
    # Initialize label normalizer
    normalizer = LabelNormalizer()
    
    all_records = []
    
    for object_type in OBJECT_TYPES:
        dataset_dir = RAW_DATA_DIR / object_type
        if not dataset_dir.exists():
            logger.warning(f"Directory not found: {dataset_dir}")
            continue
        
        records = process_dataset(dataset_dir, object_type, normalizer)
        all_records.extend(records)
    
    if not all_records:
        logger.error("No records processed. Check data/raw/.")
        return
    
    logger.info(f"Total records: {len(all_records)}")
    
    # Show distribution
    issue_counts = {}
    part_counts = {}
    for r in all_records:
        issue_counts[r["issue_type"]] = issue_counts.get(r["issue_type"], 0) + 1
        part_counts[r["object_part"]] = part_counts.get(r["object_part"], 0) + 1
    
    logger.info(f"Issue type distribution: {issue_counts}")
    logger.info(f"Object part distribution: {part_counts}")
    
    # Split dataset
    train, val, test = split_dataset(all_records)
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save splits
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_path = PROCESSED_DATA_DIR / f"{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} to {split_path}")
    
    # Save metadata
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
        "issue_distribution": issue_counts,
    }
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
