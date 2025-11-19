"""
Unit tests for LoRA metadata extraction module.

Tests the extraction of metadata from .safetensors LoRA files.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.lora_metadata import (
    extract_metadata,
    get_metadata_summary,
    is_valid_lora_file,
    _normalize_base_model,
    _extract_trigger_words,
    _extract_characters,
    _extract_styles,
)


class TestBaseModelNormalization(unittest.TestCase):
    """Test base model name normalization."""

    def test_sdxl_variations(self):
        """Test SDXL name variations are normalized."""
        variations = ['sdxl', 'SDXL', 'sdxl1.0', 'SDXL 1.0', 'stable-diffusion-xl']
        for variant in variations:
            result = _normalize_base_model(variant)
            self.assertEqual(result, 'SDXL 1.0', f"Failed for variant: {variant}")

    def test_sd15_variations(self):
        """Test SD 1.5 name variations are normalized."""
        variations = ['sd1.5', 'SD 1.5', 'sd-1.5', 'stable-diffusion-1.5']
        for variant in variations:
            result = _normalize_base_model(variant)
            self.assertEqual(result, 'SD 1.5', f"Failed for variant: {variant}")

    def test_pony_variations(self):
        """Test Pony model variations are normalized."""
        variations = ['pony', 'Pony', 'PONY', 'pdxl']
        for variant in variations:
            result = _normalize_base_model(variant)
            self.assertEqual(result, 'Pony', f"Failed for variant: {variant}")

    def test_unknown_model(self):
        """Test unknown model names are returned as-is."""
        result = _normalize_base_model('custom_model_v1')
        self.assertEqual(result, 'custom_model_v1')


class TestTriggerWordExtraction(unittest.TestCase):
    """Test trigger word extraction from metadata."""

    def test_ss_tag_frequency(self):
        """Test extraction from ss_tag_frequency format."""
        metadata = {
            'ss_tag_frequency': json.dumps({
                'dataset1': {'character_name': 50, 'style_tag': 30, 'rare_tag': 5}
            })
        }
        result = _extract_trigger_words(metadata)
        self.assertIn('character_name', result)
        self.assertIn('style_tag', result)

    def test_ss_dataset_dirs(self):
        """Test extraction from ss_dataset_dirs format."""
        metadata = {
            'ss_dataset_dirs': json.dumps({
                '10_anime_girl': {'n_repeats': 10},
                '5_cyberpunk_style': {'n_repeats': 5}
            })
        }
        result = _extract_trigger_words(metadata)
        self.assertIn('anime girl', result)
        self.assertIn('cyberpunk style', result)

    def test_plain_trigger_words(self):
        """Test extraction from plain trigger_words field."""
        metadata = {'trigger_words': 'word1, word2, word3'}
        result = _extract_trigger_words(metadata)
        self.assertEqual(result, ['word1', 'word2', 'word3'])

    def test_empty_metadata(self):
        """Test empty metadata returns empty list."""
        result = _extract_trigger_words({})
        self.assertEqual(result, [])

    def test_deduplication(self):
        """Test duplicate triggers are removed."""
        metadata = {
            'trigger_words': 'tag1, tag2, TAG1, tag2'
        }
        result = _extract_trigger_words(metadata)
        # Should have unique tags (case-insensitive dedup)
        self.assertEqual(len(result), 2)


class TestCharacterExtraction(unittest.TestCase):
    """Test character name extraction."""

    def test_character_metadata_key(self):
        """Test extraction from character metadata key."""
        metadata = {'character': 'Hatsune Miku, Luka Megurine'}
        result = _extract_characters('', metadata)
        self.assertIn('Hatsune Miku', result)
        self.assertIn('Luka Megurine', result)

    def test_character_pattern_in_text(self):
        """Test extraction from text patterns."""
        text = 'character: Alice from Wonderland'
        result = _extract_characters(text, {})
        self.assertIn('Alice', result)


class TestStyleExtraction(unittest.TestCase):
    """Test style keyword extraction."""

    def test_common_styles(self):
        """Test common style keywords are detected."""
        text = 'anime style illustration with watercolor effects'
        result = _extract_styles(text, {})
        self.assertIn('Anime', result)
        self.assertIn('Illustration', result)
        self.assertIn('Watercolor', result)

    def test_style_metadata_key(self):
        """Test extraction from style metadata key."""
        metadata = {'style': 'fantasy, sci-fi'}
        result = _extract_styles('', metadata)
        self.assertIn('fantasy', result)
        self.assertIn('sci-fi', result)


class TestExtractMetadata(unittest.TestCase):
    """Test the main extract_metadata function."""

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        result = extract_metadata('/nonexistent/path/file.safetensors')
        self.assertEqual(result['filename'], 'file.safetensors')
        self.assertTrue(len(result['extraction_errors']) > 0)

    def test_invalid_file_extension(self):
        """Test file with wrong extension still processes."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not a safetensors file')
            temp_path = f.name

        try:
            result = extract_metadata(temp_path)
            self.assertTrue(len(result['extraction_errors']) > 0)
        finally:
            os.unlink(temp_path)

    @patch('modules.lora_metadata.safe_open')
    def test_successful_extraction(self, mock_safe_open):
        """Test successful metadata extraction with mocked safetensors."""
        # Mock the safetensors file
        mock_metadata = {
            'ss_base_model_version': 'sdxl_1.0',
            'ss_training_comment': 'Test LoRA for anime style',
            'ss_tag_frequency': json.dumps({'dataset': {'anime': 100, 'girl': 50}}),
            'ss_network_dim': '32',
            'ss_network_alpha': '16.0',
        }

        mock_file = MagicMock()
        mock_file.metadata.return_value = mock_metadata
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_safe_open.return_value = mock_file

        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
            temp_path = f.name

        try:
            result = extract_metadata(temp_path)

            self.assertEqual(result['base_model'], 'SDXL 1.0')
            self.assertEqual(result['description'], 'Test LoRA for anime style')
            self.assertIn('anime', result['trigger_words'])
            self.assertEqual(result['network_dim'], 32)
            self.assertEqual(result['network_alpha'], 16.0)
            self.assertIn('Anime', result['styles'])
        finally:
            os.unlink(temp_path)


class TestGetMetadataSummary(unittest.TestCase):
    """Test the metadata summary generation."""

    def test_full_summary(self):
        """Test summary with all fields populated."""
        metadata = {
            'filename': 'test_lora.safetensors',
            'file_path': '/path/to/test_lora.safetensors',
            'file_size': 50 * 1024 * 1024,  # 50 MB
            'base_model': 'SDXL 1.0',
            'trigger_words': ['tag1', 'tag2', 'tag3'],
            'description': 'Test description',
            'characters': ['Alice'],
            'styles': ['Anime'],
            'training_epochs': None,
            'training_steps': None,
            'resolution': None,
            'network_dim': 32,
            'network_alpha': 16.0,
            'raw_metadata': {},
            'extraction_errors': [],
        }

        summary = get_metadata_summary(metadata)

        self.assertIn('test_lora.safetensors', summary)
        self.assertIn('SDXL 1.0', summary)
        self.assertIn('tag1', summary)
        self.assertIn('50.00 MB', summary)
        self.assertIn('Network Dim: 32', summary)

    def test_minimal_summary(self):
        """Test summary with minimal data."""
        metadata = {
            'filename': 'minimal.safetensors',
            'file_path': '/path/to/minimal.safetensors',
            'file_size': 1024,
            'base_model': None,
            'trigger_words': [],
            'description': None,
            'characters': [],
            'styles': [],
            'training_epochs': None,
            'training_steps': None,
            'resolution': None,
            'network_dim': None,
            'network_alpha': None,
            'raw_metadata': {},
            'extraction_errors': [],
        }

        summary = get_metadata_summary(metadata)

        self.assertIn('Unknown', summary)
        self.assertIn('No trigger words', summary)
        self.assertIn('No description', summary)


class TestIsValidLoraFile(unittest.TestCase):
    """Test LoRA file validation."""

    def test_invalid_extension(self):
        """Test file with wrong extension."""
        result = is_valid_lora_file('/path/to/file.pt')
        self.assertFalse(result)

    def test_nonexistent_file(self):
        """Test nonexistent file."""
        result = is_valid_lora_file('/nonexistent/lora.safetensors')
        self.assertFalse(result)


class TestRealLoraFile(unittest.TestCase):
    """Test with real LoRA file if available."""

    def test_real_lora_extraction(self):
        """Test extraction from a real LoRA file in the models directory."""
        lora_path = Path(__file__).parent.parent / 'models' / 'loras'

        if not lora_path.exists():
            self.skipTest("LoRA directory not found")

        lora_files = list(lora_path.glob('*.safetensors'))

        if not lora_files:
            self.skipTest("No LoRA files found in models/loras")

        # Test with the first available LoRA file
        test_file = str(lora_files[0])
        result = extract_metadata(test_file)

        # Basic validation
        self.assertEqual(result['filename'], lora_files[0].name)
        self.assertGreater(result['file_size'], 0)
        self.assertIsInstance(result['trigger_words'], list)
        self.assertIsInstance(result['raw_metadata'], dict)

        # Print summary for manual inspection
        print("\n" + "="*50)
        print("Real LoRA File Test:")
        print(get_metadata_summary(result))
        print("="*50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
