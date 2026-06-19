#!/usr/bin/env python3
"""Train improved MobileNetV3-Large multi-task classifier with augmentation and class balancing."""

import torch
from hackerrank_orchestrate.data.dataset_loader import DamageDataset, create_dataloader
from hackerrank_orchestrate.models.mobilenet_classifier import MobileNetMultiTask, Trainer
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    logger.info("Loading datasets with augmentation...")
    train_dataset = DamageDataset.from_split("train", training=True)
    val_dataset = DamageDataset.from_split("val", training=False)

    # Compute class weights for imbalanced classes
    logger.info("Computing class weights...")
    issue_weights = train_dataset.compute_class_weights("issue_type")
    part_weights = train_dataset.compute_class_weights("object_part")
    
    logger.info(f"Issue type weights: {issue_weights.numpy().round(3)}")
    logger.info(f"Object part weights: {part_weights.numpy().round(3)}")
    
    class_weights = {
        "issue_type": issue_weights,
        "object_part": part_weights,
    }

    # Use balanced sampling for training to oversample minority classes
    logger.info("Creating dataloaders with balanced sampling...")
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=32, 
        shuffle=False,  # Balanced sampler handles shuffling
        num_workers=4,
        use_balanced_sampling=True,
    )
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        use_balanced_sampling=False,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Use label encoder dimensions to ensure model matches dataset
    from hackerrank_orchestrate.data.dataset_loader import LABEL_ENCODERS

    num_object_types = len(LABEL_ENCODERS["object_type"])
    num_issue_types = len(LABEL_ENCODERS["issue_type"])
    num_object_parts = len(LABEL_ENCODERS["object_part"])

    logger.info(
        f"Model classes: object_types={num_object_types}, "
        f"issue_types={num_issue_types}, object_parts={num_object_parts}"
    )

    model = MobileNetMultiTask(
        num_object_types=num_object_types,
        num_issue_types=num_issue_types,
        num_object_parts=num_object_parts,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Improved training with SGD, cosine annealing, weighted loss, gradient clipping
    trainer = Trainer(
        model, 
        device=device, 
        lr=0.01,  # Higher initial LR for SGD
        weight_decay=5e-4,  # Stronger regularization
        class_weights=class_weights,
        warmup_epochs=5,
    )
    trainer.train(train_loader, val_loader, epochs=50)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
