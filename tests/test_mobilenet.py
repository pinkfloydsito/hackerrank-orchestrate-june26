"""Tests for MobileNet multi-task classifier."""

import pytest
import torch

from hackerrank_orchestrate.models.mobilenet_classifier import MobileNetMultiTask, WeightedMultiTaskLoss, Trainer


class TestMobileNetMultiTask:
    def test_forward_pass(self):
        model = MobileNetMultiTask(num_object_types=3, num_issue_types=12, num_object_parts=24)
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        
        outputs = model(images)
        
        assert isinstance(outputs, dict)
        assert "object_type" in outputs
        assert "issue_type" in outputs
        assert "object_part" in outputs
        assert "has_damage" in outputs
        
        assert outputs["object_type"].shape == (batch_size, 3)
        assert outputs["issue_type"].shape == (batch_size, 12)
        assert outputs["object_part"].shape == (batch_size, 24)
        assert outputs["has_damage"].shape == (batch_size,)

    def test_model_device(self):
        model = MobileNetMultiTask()
        # Test CPU by default
        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)
        assert outputs["object_type"].device == torch.device("cpu")

    def test_save_load(self, tmp_path):
        model = MobileNetMultiTask()
        model.eval()  # Set to eval mode for deterministic comparison
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        loaded_model = MobileNetMultiTask()
        loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
        loaded_model.eval()
        
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(images)
            out2 = loaded_model(images)
        
        assert torch.allclose(out1["object_type"], out2["object_type"], atol=1e-5)
        assert torch.allclose(out1["issue_type"], out2["issue_type"], atol=1e-5)
        assert torch.allclose(out1["object_part"], out2["object_part"], atol=1e-5)
        assert torch.allclose(out1["has_damage"], out2["has_damage"], atol=1e-5)


class TestWeightedMultiTaskLoss:
    def test_loss_computation(self):
        model = MobileNetMultiTask()
        criterion = WeightedMultiTaskLoss()
        
        images = torch.randn(2, 3, 224, 224)
        labels = {
            "object_type": torch.tensor([0, 1]),
            "issue_type": torch.tensor([0, 1]),
            "object_part": torch.tensor([0, 1]),
            "has_damage": torch.tensor([1.0, 0.0]),
        }
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0


class TestTrainer:
    def test_trainer_init(self):
        model = MobileNetMultiTask()
        trainer = Trainer(model, device="cpu", lr=1e-3)
        assert trainer.device == "cpu"
        assert trainer.model is model
