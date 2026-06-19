#!/usr/bin/env python3
"""Train MobileNetV3-Large multi-task classifier on processed datasets."""

import torch
from hackerrank_orchestrate.data.dataset_loader import DamageDataset, create_dataloader
from hackerrank_orchestrate.models.mobilenet_classifier import MobileNetMultiTask, Trainer
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    logger.info("Loading datasets...")
    train_dataset = DamageDataset.from_split("train")
    val_dataset = DamageDataset.from_split("val")

    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

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

    trainer = Trainer(model, device=device, lr=1e-3, weight_decay=1e-4)
    trainer.train(train_loader, val_loader, epochs=30)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
