import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import (
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOPPING_PATIENCE, CHECKPOINTS_DIR
)

logger = setup_logger(__name__)

class MobileNetMultiTask(nn.Module):
    """Multi-task classifier using MobileNetV3-Large backbone."""

    def __init__(
        self,
        num_object_types: int = 3,
        num_issue_types: int = 12,
        num_object_parts: int = 24,
    ):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = 960
        
        self.object_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_object_types)
        )
        self.issue_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_issue_types)
        )
        self.object_part_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_object_parts)
        )
        self.has_damage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
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

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = (
            self.ce(predictions["object_type"], targets["object_type"]) +
            self.ce(predictions["issue_type"], targets["issue_type"]) +
            self.ce(predictions["object_part"], targets["object_part"]) +
            self.bce(predictions["has_damage"], targets["has_damage"])
        )
        return loss

class Trainer:
    def __init__(self, model: nn.Module, device: str, lr: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY):
        self.model = model.to(device)
        self.device = device
        self.criterion = MultiTaskLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> Tuple[float, Dict[str, float]]:
        self.model.train(training)
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
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
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
        best_val_loss = float('inf')
        patience_counter = 0
        best_path = CHECKPOINTS_DIR / "best_mobilenet.pt"
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)
            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"  Accs -> Object: {val_acc['object_type']:.3f}, Issue: {val_acc['issue_type']:.3f}, Part: {val_acc['object_part']:.3f}, Damage: {val_acc['has_damage']:.3f}")

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

        logger.info("Training complete.")
