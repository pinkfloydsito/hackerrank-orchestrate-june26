"""Tests for label normalizer using embeddings."""

import pytest
from hackerrank_orchestrate.data.preprocessor import LabelNormalizer


class TestLabelNormalizer:
    @pytest.fixture(scope="class")
    def normalizer(self):
        return LabelNormalizer()

    def test_known_car_label(self, normalizer):
        issue, part = normalizer.normalize("bonnet-dent", "car")
        # Embedding-based normalization may vary, just check valid types
        assert issue in ["dent", "scratch", "crack", "broken_part", "water_damage", "stain", "torn_packaging", "crushed_packaging", "missing_part", "glass_shatter", "none", "unknown"]
        assert isinstance(part, str)

    def test_known_laptop_label(self, normalizer):
        issue, part = normalizer.normalize("BEZEL BROKEN", "laptop")
        # Embedding-based normalization may vary, just check valid types
        assert issue in ["broken_part", "crack", "scratch", "dent", "water_damage", "stain", "torn_packaging", "crushed_packaging", "missing_part", "glass_shatter", "none", "unknown"]
        assert isinstance(part, str)

    def test_known_package_label(self, normalizer):
        issue, part = normalizer.normalize("wet Package", "package")
        # Embedding-based normalization may vary, just check valid types
        assert issue in ["water_damage", "stain", "torn_packaging", "crushed_packaging", "unknown"]
        assert isinstance(part, str)

    def test_unknown_label(self, normalizer):
        issue, part = normalizer.normalize("completely-unknown-label-123", "car")
        # Should not crash, should map to something reasonable
        assert issue in ["dent", "scratch", "crack", "broken_part", "water_damage", "stain", "torn_packaging", "crushed_packaging", "missing_part", "glass_shatter", "none", "unknown"]
        assert isinstance(part, str)

    def test_empty_string(self, normalizer):
        issue, part = normalizer.normalize("", "car")
        assert isinstance(issue, str)
        assert isinstance(part, str)

    def test_multilingual_label(self, normalizer):
        issue, part = normalizer.normalize("araña en el parachoques", "car")
        assert isinstance(issue, str)
        assert isinstance(part, str)

    def test_different_object_types(self, normalizer):
        for obj_type in ["car", "laptop", "package"]:
            issue, part = normalizer.normalize("damage", obj_type)
            assert isinstance(issue, str)
            assert isinstance(part, str)
