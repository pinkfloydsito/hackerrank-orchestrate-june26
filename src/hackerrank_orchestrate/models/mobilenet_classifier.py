import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import logging
from pathlib import Path
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
)

logger = setup_logger(__name__)


class MobileNetMultiTask(nn.Module):
    """Multi-task classifier using MobileNetV3-Large backbone."""

    def __init__(
        self,
        num_object_types: int = 3,
        num_issue_types: int = 12,
        num_object_parts: int = 27,
    ):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = 960

        self.object_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_object_types),
        )
        self.issue_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_issue_types),
        )
        self.object_part_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_object_parts),
        )
        self.has_damage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return {
            "object_type": self.object_type_head(x),
            "issue_type": self.issue_type_head(x),
            "object_part": self.object_part_head(x),
            "has_damage": self.has_damage_head(x).squeeze(1),
        }


class WeightedMultiTaskLoss(nn.Module):
    def __init__(
        self, class_weights: Optional[Dict[str, torch.Tensor]] = None, label_smoothing: float = 0.1
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()

        # Store class weights for potential use
        self.class_weights = class_weights or {}

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Use weighted cross entropy if weights available
        if "issue_type" in self.class_weights:
            issue_ce = nn.CrossEntropyLoss(
                weight=self.class_weights["issue_type"].to(predictions["issue_type"].device),
                label_smoothing=self.label_smoothing,
            )
        else:
            issue_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        if "object_part" in self.class_weights:
            part_ce = nn.CrossEntropyLoss(
                weight=self.class_weights["object_part"].to(predictions["object_part"].device),
                label_smoothing=self.label_smoothing,
            )
        else:
            part_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        loss = (
            self.ce(predictions["object_type"], targets["object_type"])
            + issue_ce(predictions["issue_type"], targets["issue_type"])
            + part_ce(predictions["object_part"], targets["object_part"])
            + self.bce(predictions["has_damage"], targets["has_damage"])
        )
        return loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        warmup_epochs: int = 5,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = WeightedMultiTaskLoss(class_weights=class_weights, label_smoothing=0.1)

        # Use SGD with momentum instead of AdamW for CNN training
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )

        self.warmup_epochs = warmup_epochs
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0

    def _get_lr(self, epoch: int, total_epochs: int, base_lr: float) -> float:
        """Cosine annealing with warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def _run_epoch(
        self, dataloader: DataLoader, training: bool, epoch: int, total_epochs: int
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train(training)

        # Update learning rate for this epoch
        if training:
            lr = self._get_lr(epoch, total_epochs, self.optimizer.defaults["lr"])
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            logger.info(f"Epoch {epoch+1}/{total_epochs} - LR: {lr:.6f}")

        total_loss = 0.0
        correct = {"object_type": 0, "issue_type": 0, "object_part": 0, "has_damage": 0}
        total = {"object_type": 0, "issue_type": 0, "object_part": 0, "has_damage": 0}

        for images, labels in tqdm(dataloader, desc="Train" if training else "Val"):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            if training:
                self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if training:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            for key in ["object_type", "issue_type", "object_part"]:
                preds = outputs[key].argmax(dim=1)
                correct[key] += (preds == labels[key]).sum().item()
                total[key] += labels[key].size(0)

            damage_preds = (torch.sigmoid(outputs["has_damage"]) > 0.5).float()
            correct["has_damage"] += (damage_preds == labels["has_damage"]).sum().item()
            total["has_damage"] += labels["has_damage"].size(0)

        avg_loss = total_loss / len(dataloader)
        accuracies = {k: correct[k] / total[k] for k in total}
        return avg_loss, accuracies

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = EPOCHS) -> None:
        best_val_loss = float("inf")
        patience_counter = 0
        best_path = CHECKPOINTS_DIR / "best_mobilenet_v2.pt"
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save initial checkpoint
        torch.save(self.model.state_dict(), CHECKPOINTS_DIR / "initial_mobilenet.pt")

        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(
                train_loader, training=True, epoch=epoch, total_epochs=epochs
            )
            val_loss, val_acc = self._run_epoch(
                val_loader, training=False, epoch=epoch, total_epochs=epochs
            )

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            logger.info(
                f"  Train -> Object: {train_acc['object_type']:.3f}, Issue: {train_acc['issue_type']:.3f}, Part: {train_acc['object_part']:.3f}, Damage: {train_acc['has_damage']:.3f}"
            )
            logger.info(
                f"  Val   -> Object: {val_acc['object_type']:.3f}, Issue: {val_acc['issue_type']:.3f}, Part: {val_acc['object_part']:.3f}, Damage: {val_acc['has_damage']:.3f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"  Saved best model to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(
                    self.model.state_dict(), CHECKPOINTS_DIR / f"mobilenet_epoch_{epoch+1}.pt"
                )

        logger.info("Training complete.")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
