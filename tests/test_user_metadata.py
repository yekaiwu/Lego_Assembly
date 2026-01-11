"""
Unit tests for user-provided metadata system.
Tests metadata_models.py and user_metadata_collector.py
"""

import pytest
from src.vision_processing.metadata_models import UserProvidedMetadata
from src.vision_processing.user_metadata_collector import UserMetadataCollector


class TestUserProvidedMetadata:
    """Test UserProvidedMetadata data model."""

    def test_basic_creation(self):
        """Test creating metadata with basic required fields."""
        metadata = UserProvidedMetadata(
            main_build="Fire Truck Set #6454922",
            total_pages=50,
            instruction_pages=[5, 6, 7, 8, 9, 10]
        )

        assert metadata.main_build == "Fire Truck Set #6454922"
        assert metadata.total_pages == 50
        assert metadata.instruction_pages == [5, 6, 7, 8, 9, 10]
        assert metadata.total_instruction_pages == 6

    def test_page_ranges(self):
        """Test conversion of page lists to ranges."""
        metadata = UserProvidedMetadata(
            main_build="Test Build",
            total_pages=50,
            instruction_pages=[5, 6, 7, 10, 11, 12, 15]
        )

        ranges = metadata.instruction_page_ranges
        assert ranges == [(5, 7), (10, 12), (15, 15)]

    def test_estimated_steps(self):
        """Test step estimation."""
        metadata = UserProvidedMetadata(
            main_build="Test Build",
            total_pages=50,
            instruction_pages=list(range(5, 48))  # Pages 5-47
        )

        assert metadata.estimated_steps == 43  # 47 - 5 + 1

    def test_page_classification(self):
        """Test page classification mapping."""
        metadata = UserProvidedMetadata(
            main_build="Test Build",
            total_pages=10,
            instruction_pages=[5, 6, 7],
            final_product_pages=[8],
            parts_list_pages=[4]
        )

        classification = metadata.get_page_classification()

        assert classification[5] == "instruction"
        assert classification[6] == "instruction"
        assert classification[7] == "instruction"
        assert classification[8] == "final_product"
        assert classification[4] == "parts_list"
        assert classification[1] == "other"

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        metadata = UserProvidedMetadata(
            main_build="Fire Truck",
            total_pages=50,
            instruction_pages=[5, 6, 7],
            final_product_pages=[48],
            set_number="6454922",
            notes="Test manual"
        )

        # Convert to dict and back
        data = metadata.to_dict()
        restored = UserProvidedMetadata.from_dict(data)

        assert restored.main_build == metadata.main_build
        assert restored.total_pages == metadata.total_pages
        assert restored.instruction_pages == metadata.instruction_pages
        assert restored.final_product_pages == metadata.final_product_pages
        assert restored.set_number == metadata.set_number
        assert restored.notes == metadata.notes

    def test_invalid_page_numbers(self):
        """Test validation of page numbers."""
        with pytest.raises(ValueError):
            UserProvidedMetadata(
                main_build="Test",
                total_pages=10,
                instruction_pages=[1, 2, 15]  # Page 15 out of range
            )

    def test_page_sorting(self):
        """Test that instruction pages are automatically sorted."""
        metadata = UserProvidedMetadata(
            main_build="Test",
            total_pages=20,
            instruction_pages=[10, 5, 15, 7]  # Unsorted
        )

        assert metadata.instruction_pages == [5, 7, 10, 15]

    def test_display_summary(self):
        """Test summary display formatting."""
        metadata = UserProvidedMetadata(
            main_build="Fire Truck Set #6454922",
            total_pages=50,
            instruction_pages=list(range(5, 48)),
            final_product_pages=[48],
            parts_list_pages=[4]
        )

        summary = metadata.display_summary()

        assert "Fire Truck Set #6454922" in summary
        assert "Total Pages: 50" in summary
        assert "5-47" in summary


class TestUserMetadataCollector:
    """Test UserMetadataCollector functionality."""

    def test_parse_page_range(self):
        """Test parsing page range strings."""
        collector = UserMetadataCollector(total_pages=100)

        # Simple range
        pages = collector.parse_page_input("5-10")
        assert pages == [5, 6, 7, 8, 9, 10]

        # List
        pages = collector.parse_page_input("1, 3, 5")
        assert pages == [1, 3, 5]

        # Combined
        pages = collector.parse_page_input("1-3, 7, 10-12")
        assert pages == [1, 2, 3, 7, 10, 11, 12]

    def test_parse_page_input_with_spaces(self):
        """Test parsing with various spacing."""
        collector = UserMetadataCollector(total_pages=100)

        pages = collector.parse_page_input(" 1 - 5 ,  10  , 15 - 20 ")
        assert pages == [1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19, 20]

    def test_parse_page_input_removes_duplicates(self):
        """Test that duplicates are removed."""
        collector = UserMetadataCollector(total_pages=100)

        pages = collector.parse_page_input("1-5, 3-7")
        # Should be [1,2,3,4,5,3,4,5,6,7] → [1,2,3,4,5,6,7]
        assert pages == [1, 2, 3, 4, 5, 6, 7]

    def test_invalid_range(self):
        """Test invalid range handling."""
        collector = UserMetadataCollector(total_pages=100)

        with pytest.raises(ValueError):
            collector.parse_page_input("10-5")  # Start > end

    def test_invalid_format(self):
        """Test invalid format handling."""
        collector = UserMetadataCollector(total_pages=100)

        with pytest.raises(ValueError):
            collector.parse_page_input("abc")

        with pytest.raises(ValueError):
            collector.parse_page_input("1, 2, xyz")

    def test_collect_from_dict(self):
        """Test programmatic metadata collection from dictionary."""
        collector = UserMetadataCollector(total_pages=50)

        data = {
            "main_build": "Fire Truck Set #6454922",
            "instruction_pages": "5-47",
            "final_product_pages": "48",
            "parts_list_pages": "4",
            "notes": "Test manual"
        }

        metadata = collector.collect_metadata_from_dict(data)

        assert metadata.main_build == "Fire Truck Set #6454922"
        assert metadata.instruction_pages == list(range(5, 48))
        assert metadata.final_product_pages == [48]
        assert metadata.parts_list_pages == [4]
        assert metadata.notes == "Test manual"

    def test_collect_from_dict_with_list_input(self):
        """Test dict input with lists instead of strings."""
        collector = UserMetadataCollector(total_pages=50)

        data = {
            "main_build": "Test Build",
            "instruction_pages": [5, 6, 7, 8],  # Already a list
            "final_product_pages": [48, 49]
        }

        metadata = collector.collect_metadata_from_dict(data)

        assert metadata.instruction_pages == [5, 6, 7, 8]
        assert metadata.final_product_pages == [48, 49]

    def test_extract_set_number(self):
        """Test set number extraction from build name."""
        collector = UserMetadataCollector(total_pages=50)

        # Pattern: Set #XXXXX
        set_num = collector._extract_set_number("Fire Truck Set #6454922")
        assert set_num == "6454922"

        # Pattern: Set XXXXX
        set_num = collector._extract_set_number("Race Car Set 42123")
        assert set_num == "42123"

        # Pattern: #XXXXX
        set_num = collector._extract_set_number("Build #75301")
        assert set_num == "75301"

        # No set number
        set_num = collector._extract_set_number("Custom Build")
        assert set_num is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
