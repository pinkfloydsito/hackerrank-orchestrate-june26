"""Configuration and constants for the project."""

from pathlib import Path
from typing import Dict, List, Tuple


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Roboflow dataset configurations
ROBOFLOW_DATASETS: List[Tuple[str, str, int, str]] = [
    ("gharavi", "laptop-defects-all-sides", 1, "laptop"),
    ("bhagyashri-biradar", "damage-package-detection", 5, "package"),
    ("sindhu", "car_dent_scratch_detection-1", 9, "car"),
]

# Unified schema
OBJECT_TYPES = ["car", "laptop", "package"]

ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]

OBJECT_PARTS: Dict[str, List[str]] = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    ],
}

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]
SEVERITY = ["none", "low", "medium", "high", "unknown"]

# Image processing
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

# Qwen model
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_FALLBACK_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 512

# Evidence requirements
EVIDENCE_REQUIREMENTS_PATH = PROJECT_ROOT / "dataset" / "evidence_requirements.csv"
USER_HISTORY_PATH = PROJECT_ROOT / "dataset" / "user_history.csv"
SAMPLE_CLAIMS_PATH = PROJECT_ROOT / "dataset" / "sample_claims.csv"
TEST_CLAIMS_PATH = PROJECT_ROOT / "dataset" / "claims.csv"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "sample"
TEST_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "test"
