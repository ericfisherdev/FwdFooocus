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
    LoraMetadataScanner,
    get_scanner,
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


class TestLoraMetadataScanner(unittest.TestCase):
    """Test the LoraMetadataScanner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scanner_initialization(self):
        """Test scanner initializes with correct defaults."""
        scanner = LoraMetadataScanner()
        self.assertFalse(scanner.is_scanning)
        self.assertFalse(scanner.scan_complete)
        self.assertEqual(len(scanner.metadata_index), 0)

    def test_scanner_with_custom_paths(self):
        """Test scanner accepts custom paths."""
        paths = ['/path1', '/path2']
        scanner = LoraMetadataScanner(lora_paths=paths)
        self.assertEqual(scanner._lora_paths, paths)

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        self.scanner.start_scan(blocking=True)
        self.assertTrue(self.scanner.scan_complete)
        self.assertEqual(len(self.scanner.metadata_index), 0)

    def test_scan_stats(self):
        """Test scan statistics are tracked."""
        self.scanner.start_scan(blocking=True)
        stats = self.scanner.scan_stats

        self.assertIn('is_scanning', stats)
        self.assertIn('scan_complete', stats)
        self.assertIn('files_scanned', stats)
        self.assertIn('files_failed', stats)
        self.assertIn('total_indexed', stats)
        self.assertIn('elapsed_time', stats)

    def test_get_metadata_not_found(self):
        """Test get_metadata returns None for unknown file."""
        result = self.scanner.get_metadata('/nonexistent/file.safetensors')
        self.assertIsNone(result)

    def test_get_metadata_by_filename_no_match(self):
        """Test get_metadata_by_filename returns empty list for no match."""
        result = self.scanner.get_metadata_by_filename('nonexistent.safetensors')
        self.assertEqual(result, [])

    def test_search_by_base_model_empty(self):
        """Test search_by_base_model returns empty list when no matches."""
        result = self.scanner.search_by_base_model('SDXL')
        self.assertEqual(result, [])

    def test_search_by_trigger_word_empty(self):
        """Test search_by_trigger_word returns empty list when no matches."""
        result = self.scanner.search_by_trigger_word('anime')
        self.assertEqual(result, [])

    def test_clear_index(self):
        """Test clearing the metadata index."""
        # Manually add an entry
        self.scanner._metadata_index['test'] = {'filename': 'test'}
        self.scanner._scan_complete = True

        self.scanner.clear_index()

        self.assertEqual(len(self.scanner.metadata_index), 0)
        self.assertFalse(self.scanner.scan_complete)

    def test_remove_file(self):
        """Test removing a file from the index."""
        # Add a file
        test_path = '/test/file.safetensors'
        self.scanner._metadata_index[test_path] = {'filename': 'file.safetensors'}

        # Remove it
        result = self.scanner.remove_file(test_path)
        self.assertTrue(result)
        self.assertIsNone(self.scanner.get_metadata(test_path))

    def test_remove_file_not_found(self):
        """Test removing a file that doesn't exist in index."""
        result = self.scanner.remove_file('/nonexistent/file.safetensors')
        self.assertFalse(result)

    def test_scan_already_in_progress(self):
        """Test that starting a scan when one is in progress is ignored."""
        self.scanner._is_scanning = True

        # This should not start a new scan
        self.scanner.start_scan(blocking=False)

        # Scanner should still be in the original scanning state
        self.assertTrue(self.scanner._is_scanning)

    def test_nonexistent_path_warning(self):
        """Test that nonexistent paths are handled gracefully."""
        scanner = LoraMetadataScanner(lora_paths=['/nonexistent/path'])
        scanner.start_scan(blocking=True)

        self.assertTrue(scanner.scan_complete)
        self.assertEqual(len(scanner.metadata_index), 0)


class TestLoraMetadataScannerWithMockedFiles(unittest.TestCase):
    """Test scanner with mocked safetensors files."""

    def setUp(self):
        """Set up test fixtures with mocked files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('modules.lora_metadata.extract_metadata')
    def test_scan_with_mock_files(self, mock_extract):
        """Test scanning with mocked metadata extraction."""
        # Create fake safetensors files
        for i in range(3):
            Path(self.temp_dir, f'lora_{i}.safetensors').touch()

        # Mock the extraction function
        def mock_extract_fn(path):
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': 'SDXL 1.0',
                'trigger_words': ['test'],
                'description': 'Test LoRA',
                'characters': [],
                'styles': [],
                'training_epochs': None,
                'training_steps': None,
                'resolution': None,
                'network_dim': 32,
                'network_alpha': 16.0,
                'raw_metadata': {},
                'extraction_errors': [],
            }

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        self.assertEqual(len(scanner.metadata_index), 3)
        self.assertEqual(scanner.scan_stats['files_scanned'], 3)
        self.assertEqual(scanner.scan_stats['files_failed'], 0)

    @patch('modules.lora_metadata.extract_metadata')
    def test_scan_handles_extraction_errors(self, mock_extract):
        """Test that individual file failures don't stop the scan."""
        # Create fake safetensors files
        for i in range(3):
            Path(self.temp_dir, f'lora_{i}.safetensors').touch()

        # Make the second file fail
        def mock_extract_fn(path):
            if 'lora_1' in path:
                raise RuntimeError("Simulated extraction failure")
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': 'SDXL 1.0',
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

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        # Should have processed 2 files successfully, 1 failed
        self.assertEqual(scanner.scan_stats['files_scanned'], 2)
        self.assertEqual(scanner.scan_stats['files_failed'], 1)

    @patch('modules.lora_metadata.extract_metadata')
    def test_search_by_base_model(self, mock_extract):
        """Test searching by base model."""
        # Create fake safetensors files
        Path(self.temp_dir, 'sdxl_lora.safetensors').touch()
        Path(self.temp_dir, 'sd15_lora.safetensors').touch()

        def mock_extract_fn(path):
            base_model = 'SDXL 1.0' if 'sdxl' in path else 'SD 1.5'
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': base_model,
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

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        sdxl_results = scanner.search_by_base_model('SDXL')
        sd15_results = scanner.search_by_base_model('SD 1.5')

        self.assertEqual(len(sdxl_results), 1)
        self.assertEqual(len(sd15_results), 1)
        self.assertEqual(sdxl_results[0]['base_model'], 'SDXL 1.0')
        self.assertEqual(sd15_results[0]['base_model'], 'SD 1.5')

    @patch('modules.lora_metadata.extract_metadata')
    def test_search_by_trigger_word(self, mock_extract):
        """Test searching by trigger word."""
        Path(self.temp_dir, 'anime_lora.safetensors').touch()
        Path(self.temp_dir, 'realistic_lora.safetensors').touch()

        def mock_extract_fn(path):
            triggers = ['anime', 'girl'] if 'anime' in path else ['realistic', 'photo']
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': 'SDXL 1.0',
                'trigger_words': triggers,
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

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        anime_results = scanner.search_by_trigger_word('anime')
        photo_results = scanner.search_by_trigger_word('photo')

        self.assertEqual(len(anime_results), 1)
        self.assertEqual(len(photo_results), 1)

    @patch('modules.lora_metadata.extract_metadata')
    def test_get_metadata_by_filename(self, mock_extract):
        """Test getting metadata by filename."""
        Path(self.temp_dir, 'test_lora.safetensors').touch()

        def mock_extract_fn(path):
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': 'SDXL 1.0',
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

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        results = scanner.get_metadata_by_filename('test_lora.safetensors')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['filename'], 'test_lora.safetensors')

    @patch('modules.lora_metadata.extract_metadata')
    def test_recursive_scanning(self, mock_extract):
        """Test that scanner finds files in subdirectories."""
        # Create nested directories with files
        subdir = Path(self.temp_dir) / 'subdir' / 'nested'
        subdir.mkdir(parents=True)

        Path(self.temp_dir, 'root_lora.safetensors').touch()
        Path(subdir, 'nested_lora.safetensors').touch()

        def mock_extract_fn(path):
            return {
                'filename': os.path.basename(path),
                'file_path': path,
                'file_size': 1024,
                'base_model': 'SDXL 1.0',
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

        mock_extract.side_effect = mock_extract_fn

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        scanner.start_scan(blocking=True)

        # Should find both files
        self.assertEqual(len(scanner.metadata_index), 2)

    @patch('modules.lora_metadata.extract_metadata')
    def test_refresh_file(self, mock_extract):
        """Test refreshing metadata for a specific file."""
        file_path = str(Path(self.temp_dir) / 'test.safetensors')
        Path(file_path).touch()

        mock_extract.return_value = {
            'filename': 'test.safetensors',
            'file_path': file_path,
            'file_size': 2048,
            'base_model': 'SD 1.5',
            'trigger_words': ['refreshed'],
            'description': 'Refreshed',
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

        scanner = LoraMetadataScanner(lora_paths=[self.temp_dir])
        result = scanner.refresh_file(file_path)

        self.assertIsNotNone(result)
        self.assertEqual(result['description'], 'Refreshed')
        self.assertEqual(scanner.get_metadata(file_path)['description'], 'Refreshed')


class TestBackgroundScanFunction(unittest.TestCase):
    """Test the start_background_scan convenience function."""

    @patch('modules.lora_metadata._scanner', None)
    def test_get_scanner_creates_instance(self):
        """Test that get_scanner creates a new instance if none exists."""
        # Reset global scanner
        import modules.lora_metadata as lm
        lm._scanner = None

        scanner = get_scanner()
        self.assertIsInstance(scanner, LoraMetadataScanner)

        # Second call should return same instance
        scanner2 = get_scanner()
        self.assertIs(scanner, scanner2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
