"""
LoRA Metadata Extraction Module

Extracts metadata from .safetensors LoRA files including base model version,
trigger words, descriptions, character names, and style information.

This module provides the foundation for the LoRA Library feature (Epic 2).
"""

import copy
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open

logger = logging.getLogger(__name__)


# Common metadata key variations from different LoRA sources
METADATA_KEY_MAPPINGS = {
    'base_model': [
        'ss_base_model_version',
        'base_model',
        'ss_sd_model_name',
        'modelspec.architecture',
        'sd_model_name',
    ],
    'trigger_words': [
        'ss_tag_frequency',
        'trigger_words',
        'activation_text',
        'ss_dataset_dirs',
    ],
    'description': [
        'ss_training_comment',
        'description',
        'modelspec.description',
        'ss_output_name',
    ],
    'training_epochs': [
        'ss_epoch',
        'ss_num_epochs',
        'epochs',
    ],
    'training_steps': [
        'ss_steps',
        'ss_max_train_steps',
        'steps',
    ],
    'resolution': [
        'ss_resolution',
        'ss_bucket_info',
        'resolution',
    ],
    'network_dim': [
        'ss_network_dim',
        'network_dim',
        'lora_network_dim',
    ],
    'network_alpha': [
        'ss_network_alpha',
        'network_alpha',
        'lora_network_alpha',
    ],
}

# Base model name normalization patterns
BASE_MODEL_PATTERNS = {
    r'sd[-_]?1\.?5|stable[-_]?diffusion[-_]?1\.?5': 'SD 1.5',
    r'sd[-_]?2\.?1|stable[-_]?diffusion[-_]?2\.?1': 'SD 2.1',
    r'sdxl(?:[-_ ]?1\.?0)?|stable[-_]?diffusion[-_]?xl': 'SDXL 1.0',
    r'pony|pdxl': 'Pony',
    r'sd[-_]?3|stable[-_]?diffusion[-_]?3': 'SD 3',
    r'flux': 'Flux',
}


def extract_metadata(file_path: str) -> dict[str, Any]:
    """
    Extract metadata from a .safetensors LoRA file.

    Args:
        file_path: Path to the .safetensors file

    Returns:
        Dictionary containing extracted metadata with the following structure:
        {
            'filename': str,
            'file_path': str,
            'file_size': int,
            'base_model': str or None,
            'trigger_words': list[str],
            'description': str or None,
            'characters': list[str],
            'styles': list[str],
            'training_epochs': int or None,
            'training_steps': int or None,
            'resolution': str or None,
            'network_dim': int or None,
            'network_alpha': float or None,
            'raw_metadata': dict,
            'extraction_errors': list[str],
        }
    """
    result = {
        'filename': os.path.basename(file_path),
        'file_path': file_path,
        'file_size': 0,
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

    try:
        # Get file size
        result['file_size'] = os.path.getsize(file_path)
    except OSError as e:
        result['extraction_errors'].append(f"Failed to get file size: {e}")

    try:
        # Open safetensors file and extract metadata from header
        with safe_open(file_path, framework="pt") as f:
            raw_metadata = f.metadata()

            if raw_metadata is None:
                result['extraction_errors'].append("No metadata found in file")
                return result

            result['raw_metadata'] = dict(raw_metadata)

            # Extract base model
            result['base_model'] = _extract_base_model(raw_metadata)

            # Extract trigger words
            result['trigger_words'] = _extract_trigger_words(raw_metadata)

            # Extract description
            result['description'] = _extract_description(raw_metadata)

            # Extract training info
            result['training_epochs'] = _extract_numeric_field(
                raw_metadata, METADATA_KEY_MAPPINGS['training_epochs']
            )
            result['training_steps'] = _extract_numeric_field(
                raw_metadata, METADATA_KEY_MAPPINGS['training_steps']
            )

            # Extract network parameters
            result['network_dim'] = _extract_numeric_field(
                raw_metadata, METADATA_KEY_MAPPINGS['network_dim']
            )
            result['network_alpha'] = _extract_numeric_field(
                raw_metadata, METADATA_KEY_MAPPINGS['network_alpha'], as_float=True
            )

            # Extract resolution
            result['resolution'] = _extract_resolution(raw_metadata)

            # Parse characters and styles from description and trigger words
            all_text = ' '.join([
                result['description'] or '',
                ' '.join(result['trigger_words']),
            ])
            result['characters'] = _extract_characters(all_text, raw_metadata)
            result['styles'] = _extract_styles(all_text, raw_metadata)

    except Exception as e:
        error_msg = f"Failed to extract metadata: {type(e).__name__}: {e}"
        result['extraction_errors'].append(error_msg)
        logger.warning(f"Error extracting metadata from {file_path}: {error_msg}")

    return result


def _extract_base_model(metadata: dict) -> str | None:
    """Extract and normalize base model name from metadata."""
    for key in METADATA_KEY_MAPPINGS['base_model']:
        if key in metadata:
            value = metadata[key]
            if value:
                return _normalize_base_model(str(value))
    return None


def _normalize_base_model(model_string: str) -> str:
    """Normalize base model string to standard format."""
    model_lower = model_string.lower()

    for pattern, normalized_name in BASE_MODEL_PATTERNS.items():
        if re.search(pattern, model_lower):
            return normalized_name

    # Return original if no pattern matches, but clean it up
    return model_string.strip()


def _extract_trigger_words(metadata: dict) -> list[str]:
    """Extract trigger words from metadata."""
    trigger_words = []

    for key in METADATA_KEY_MAPPINGS['trigger_words']:
        if key not in metadata:
            continue

        value = metadata[key]
        if not value:
            continue

        # ss_tag_frequency is a JSON string with tag frequencies
        if key == 'ss_tag_frequency':
            try:
                tag_freq = json.loads(value) if isinstance(value, str) else value
                if isinstance(tag_freq, dict):
                    # Format: {"dataset_name": {"tag1": count, "tag2": count}}
                    for dataset_tags in tag_freq.values():
                        if isinstance(dataset_tags, dict):
                            # Sort by frequency and take top tags
                            sorted_tags = sorted(
                                dataset_tags.items(),
                                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                                reverse=True
                            )
                            trigger_words.extend([tag for tag, _ in sorted_tags[:20]])
            except (json.JSONDecodeError, TypeError):
                pass

        # ss_dataset_dirs contains directory names which often have trigger words
        elif key == 'ss_dataset_dirs':
            try:
                dirs = json.loads(value) if isinstance(value, str) else value
                if isinstance(dirs, dict):
                    for dir_name in dirs.keys():
                        # Directory names often contain trigger words
                        # Format: "1_character_name" or "10_style_name"
                        parts = dir_name.split('_', 1)
                        if len(parts) > 1:
                            trigger_words.append(parts[1].replace('_', ' '))
            except (json.JSONDecodeError, TypeError):
                pass

        # Plain trigger words or activation text
        else:
            if isinstance(value, str):
                # Split by common delimiters
                words = re.split(r'[,;\n]', value)
                trigger_words.extend([w.strip() for w in words if w.strip()])
            elif isinstance(value, list):
                trigger_words.extend([str(w).strip() for w in value if w])

    # Remove duplicates while preserving order
    seen = set()
    unique_triggers = []
    for word in trigger_words:
        word_lower = word.lower()
        if word_lower not in seen and word:
            seen.add(word_lower)
            unique_triggers.append(word)

    return unique_triggers


def _extract_description(metadata: dict) -> str | None:
    """Extract description from metadata."""
    for key in METADATA_KEY_MAPPINGS['description']:
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return None


def _extract_numeric_field(
    metadata: dict,
    keys: list[str],
    as_float: bool = False
) -> int | float | None:
    """Extract a numeric field from metadata."""
    for key in keys:
        if key in metadata:
            value = metadata[key]
            try:
                if as_float:
                    return float(value)
                return int(float(value))
            except (ValueError, TypeError):
                continue
    return None


def _extract_resolution(metadata: dict) -> str | None:
    """Extract training resolution from metadata."""
    for key in METADATA_KEY_MAPPINGS['resolution']:
        if key not in metadata:
            continue

        value = metadata[key]
        if not value:
            continue

        # ss_bucket_info contains bucket resolution info
        if key == 'ss_bucket_info':
            try:
                bucket_info = json.loads(value) if isinstance(value, str) else value
                if isinstance(bucket_info, dict):
                    # Get the max resolution from buckets
                    buckets = bucket_info.get('buckets', {})
                    if buckets:
                        resolutions = []
                        for res_key in buckets.keys():
                            # Format: "[512, 768]" or "(512, 768)"
                            try:
                                res = json.loads(res_key.replace('(', '[').replace(')', ']'))
                                if isinstance(res, list) and len(res) == 2:
                                    resolutions.append(f"{res[0]}x{res[1]}")
                            except (json.JSONDecodeError, ValueError, TypeError):
                                continue
                        if resolutions:
                            return ', '.join(sorted(set(resolutions)))
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            # Plain resolution string
            return str(value).strip()

    return None


def _extract_characters(text: str, metadata: dict) -> list[str]:
    """
    Extract character names from text and metadata.

    Looks for common patterns indicating character names in LoRA descriptions.
    """
    characters = []

    # Check for character-related metadata keys
    character_keys = ['ss_character', 'character', 'characters']
    for key in character_keys:
        value = metadata.get(key)
        if value:
            if isinstance(value, str):
                characters.extend([c.strip() for c in value.split(',') if c.strip()])
            elif isinstance(value, list):
                characters.extend([str(c).strip() for c in value if c])

    # Look for character patterns in text
    # Common patterns: "character: Name", "for character Name", etc.
    patterns = [
        r'character[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:^|\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:from|character)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        characters.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_chars = []
    for char in characters:
        char_lower = char.lower()
        if char_lower not in seen and char:
            seen.add(char_lower)
            unique_chars.append(char)

    return unique_chars


def _extract_styles(text: str, metadata: dict) -> list[str]:
    """
    Extract style keywords from text and metadata.

    Identifies common artistic style indicators.
    """
    styles = []

    # Common style keywords to look for
    style_keywords = [
        'anime', 'realistic', 'photorealistic', 'cartoon', 'manga',
        'watercolor', 'oil painting', 'digital art', 'concept art',
        'illustration', 'sketch', 'line art', 'cel shaded',
        '3d render', 'pixel art', 'fantasy', 'sci-fi', 'cyberpunk',
        'steampunk', 'art nouveau', 'art deco', 'impressionist',
        'surreal', 'abstract', 'minimalist', 'vintage', 'retro',
    ]

    text_lower = text.lower()
    for keyword in style_keywords:
        if keyword in text_lower:
            styles.append(keyword.title())

    # Check for style-related metadata keys
    style_keys = ['ss_style', 'style', 'styles', 'art_style']
    for key in style_keys:
        value = metadata.get(key)
        if value:
            if isinstance(value, str):
                styles.extend([s.strip() for s in value.split(',') if s.strip()])
            elif isinstance(value, list):
                styles.extend([str(s).strip() for s in value if s])

    # Remove duplicates while preserving order
    seen = set()
    unique_styles = []
    for style in styles:
        style_lower = style.lower()
        if style_lower not in seen and style:
            seen.add(style_lower)
            unique_styles.append(style)

    return unique_styles


def get_metadata_summary(metadata: dict) -> str:
    """
    Get a human-readable summary of the extracted metadata.

    Args:
        metadata: Dictionary returned by extract_metadata()

    Returns:
        Formatted string summary of the metadata
    """
    lines = [f"LoRA: {metadata['filename']}"]

    if metadata['base_model']:
        lines.append(f"  Base Model: {metadata['base_model']}")
    else:
        lines.append("  Base Model: Unknown")

    if metadata['trigger_words']:
        triggers = ', '.join(metadata['trigger_words'][:5])
        if len(metadata['trigger_words']) > 5:
            triggers += f" (+{len(metadata['trigger_words']) - 5} more)"
        lines.append(f"  Trigger Words: {triggers}")
    else:
        lines.append("  Trigger Words: No trigger words available")

    if metadata['description']:
        desc = metadata['description'][:100]
        if len(metadata['description']) > 100:
            desc += "..."
        lines.append(f"  Description: {desc}")
    else:
        lines.append("  Description: No description")

    if metadata['characters']:
        lines.append(f"  Characters: {', '.join(metadata['characters'])}")

    if metadata['styles']:
        lines.append(f"  Styles: {', '.join(metadata['styles'])}")

    # File info
    size_mb = metadata['file_size'] / (1024 * 1024)
    lines.append(f"  File Size: {size_mb:.2f} MB")

    # Network info
    if metadata['network_dim']:
        lines.append(f"  Network Dim: {metadata['network_dim']}")
    if metadata['network_alpha']:
        lines.append(f"  Network Alpha: {metadata['network_alpha']}")

    if metadata['extraction_errors']:
        lines.append(f"  Warnings: {len(metadata['extraction_errors'])} extraction issues")

    return '\n'.join(lines)


def is_valid_lora_file(file_path: str) -> bool:
    """
    Check if a file is a valid LoRA safetensors file.

    Args:
        file_path: Path to check

    Returns:
        True if the file appears to be a valid LoRA file
    """
    if not file_path.lower().endswith('.safetensors'):
        return False

    if not os.path.isfile(file_path):
        return False

    try:
        # Try to open and check for LoRA-specific keys
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            # LoRA files typically have keys with 'lora' in them
            has_lora_keys = any('lora' in key.lower() for key in keys)
            # Also check for common LoRA patterns like 'down' and 'up' blocks
            has_lora_structure = any(
                'lora_down' in key.lower() or 'lora_up' in key.lower()
                for key in keys
            )
            return has_lora_keys or has_lora_structure
    except (OSError, RuntimeError) as e:
        logger.debug(f"File validation failed for {file_path}: {e}")
        return False


class LoraMetadataScanner:
    """
    Background scanner for LoRA metadata extraction.

    Scans configured directories for .safetensors files and builds
    an in-memory index of extracted metadata. Runs in a background
    thread to avoid blocking UI startup.
    """

    def __init__(self, lora_paths: list[str] | None = None):
        """
        Initialize the scanner.

        Args:
            lora_paths: List of directory paths to scan. If None, will be
                       loaded from config when scan starts.
        """
        self._lora_paths: list[str] = lora_paths or []
        self._metadata_index: dict[str, dict[str, Any]] = {}
        self._scan_thread: threading.Thread | None = None
        self._is_scanning: bool = False
        self._scan_complete: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._stop_requested: bool = False
        self._scan_start_time: float = 0.0
        self._files_scanned: int = 0
        self._files_failed: int = 0

    @property
    def is_scanning(self) -> bool:
        """Check if a scan is currently in progress."""
        return self._is_scanning

    @property
    def scan_complete(self) -> bool:
        """Check if the initial scan has completed."""
        return self._scan_complete

    @property
    def metadata_index(self) -> dict[str, dict[str, Any]]:
        """
        Get the current metadata index.

        Returns:
            Dictionary mapping file paths to their extracted metadata.
        """
        with self._lock:
            return dict(self._metadata_index)

    @property
    def scan_stats(self) -> dict[str, Any]:
        """
        Get statistics about the last/current scan.

        Returns:
            Dictionary with scan statistics.
        """
        with self._lock:
            return {
                'is_scanning': self._is_scanning,
                'scan_complete': self._scan_complete,
                'files_scanned': self._files_scanned,
                'files_failed': self._files_failed,
                'total_indexed': len(self._metadata_index),
                'elapsed_time': time.time() - self._scan_start_time if self._scan_start_time else 0,
            }

    def start_scan(self, blocking: bool = False) -> None:
        """
        Start scanning for LoRA files.

        Args:
            blocking: If True, wait for scan to complete. If False,
                     run in background thread.
        """
        with self._lock:
            if self._is_scanning:
                logger.warning("Scan already in progress, ignoring start request")
                return

            # Initialize scan state atomically while holding the lock
            self._is_scanning = True
            self._scan_complete = False
            self._stop_requested = False
            self._files_scanned = 0
            self._files_failed = 0
            self._scan_start_time = time.time()

        if blocking:
            self._run_scan()
        else:
            self._scan_thread = threading.Thread(
                target=self._run_scan,
                name="LoraMetadataScanner",
                daemon=True
            )
            self._scan_thread.start()
            logger.info("Started background LoRA metadata scan")

    def stop_scan(self) -> None:
        """Request the current scan to stop."""
        self._stop_requested = True
        if self._scan_thread and self._scan_thread.is_alive():
            logger.info("Stopping LoRA metadata scan...")
            self._scan_thread.join(timeout=5.0)

    def get_metadata(self, file_path: str) -> dict[str, Any] | None:
        """
        Get metadata for a specific file.

        Args:
            file_path: Path to the LoRA file.

        Returns:
            Metadata dictionary (deep copy) or None if not indexed.
        """
        with self._lock:
            metadata = self._metadata_index.get(file_path)
            if metadata is None:
                return None
            return copy.deepcopy(metadata)

    def get_metadata_by_filename(self, filename: str) -> list[dict[str, Any]]:
        """
        Get metadata for files matching a filename.

        Args:
            filename: Filename to search for (without path).

        Returns:
            List of metadata dictionaries (deep copies) for matching files.
        """
        results = []
        with self._lock:
            for path, metadata in self._metadata_index.items():
                if os.path.basename(path) == filename:
                    results.append(copy.deepcopy(metadata))
        return results

    def search_by_base_model(self, base_model: str) -> list[dict[str, Any]]:
        """
        Find all LoRAs for a specific base model.

        Args:
            base_model: Base model name to search for.

        Returns:
            List of metadata dictionaries (deep copies) for matching LoRAs.
        """
        results = []
        base_model_lower = base_model.lower()
        with self._lock:
            for metadata in self._metadata_index.values():
                if metadata.get('base_model'):
                    if base_model_lower in metadata['base_model'].lower():
                        results.append(copy.deepcopy(metadata))
        return results

    def search_by_trigger_word(self, trigger_word: str) -> list[dict[str, Any]]:
        """
        Find all LoRAs containing a specific trigger word.

        Args:
            trigger_word: Trigger word to search for.

        Returns:
            List of metadata dictionaries (deep copies) for matching LoRAs.
        """
        results = []
        trigger_lower = trigger_word.lower()
        with self._lock:
            for metadata in self._metadata_index.values():
                for word in metadata.get('trigger_words', []):
                    if trigger_lower in word.lower():
                        results.append(copy.deepcopy(metadata))
                        break
        return results

    def _run_scan(self) -> None:
        """Execute the scan operation."""
        try:
            # Load paths from config if not provided
            if not self._lora_paths:
                self._lora_paths = self._load_lora_paths_from_config()

            if not self._lora_paths:
                logger.warning("No LoRA paths configured, scan aborted")
                return

            logger.info(f"Starting LoRA scan in {len(self._lora_paths)} directories")

            # Find all .safetensors files
            files_to_scan = self._discover_lora_files()
            total_files = len(files_to_scan)

            if total_files == 0:
                logger.info("No LoRA files found to scan")
                return

            logger.info(f"Found {total_files} LoRA files to scan")

            # Process each file
            for index, file_path in enumerate(files_to_scan):
                if self._stop_requested:
                    logger.info("Scan stopped by request")
                    break

                try:
                    metadata = extract_metadata(file_path)
                    with self._lock:
                        self._metadata_index[file_path] = metadata
                        self._files_scanned += 1

                    # Log progress every 10 files or at milestones
                    if (index + 1) % 10 == 0 or (index + 1) == total_files:
                        progress = ((index + 1) / total_files) * 100
                        logger.info(
                            f"Scan progress: {index + 1}/{total_files} "
                            f"({progress:.1f}%) - {metadata['filename']}"
                        )

                except Exception as e:
                    with self._lock:
                        self._files_failed += 1
                    logger.error(f"Failed to extract metadata from {file_path}: {e}")

            elapsed = time.time() - self._scan_start_time
            logger.info(
                f"LoRA scan complete: {self._files_scanned} files indexed, "
                f"{self._files_failed} failures, {elapsed:.2f}s elapsed"
            )

        except Exception as e:
            logger.error(f"Scan failed with error: {e}")

        finally:
            with self._lock:
                self._is_scanning = False
                self._scan_complete = True

    def _load_lora_paths_from_config(self) -> list[str]:
        """Load LoRA directory paths from config."""
        try:
            # Import here to avoid circular imports
            from modules.config import paths_loras

            if isinstance(paths_loras, list):
                return [str(p) for p in paths_loras]
            elif paths_loras:
                return [str(paths_loras)]
            return []
        except ImportError as e:
            logger.error(f"Failed to import config: {e}")
            return []

    def _discover_lora_files(self) -> list[str]:
        """
        Recursively discover all .safetensors files in configured paths.

        Returns:
            List of file paths to scan.
        """
        files = []

        for lora_path in self._lora_paths:
            path = Path(lora_path)

            if not path.exists():
                logger.warning(f"LoRA path does not exist: {lora_path}")
                continue

            if not path.is_dir():
                logger.warning(f"LoRA path is not a directory: {lora_path}")
                continue

            # Recursively find all .safetensors files
            for file_path in path.rglob('*.safetensors'):
                files.append(str(file_path))

        return sorted(files)

    def refresh_file(self, file_path: str) -> dict[str, Any] | None:
        """
        Refresh metadata for a specific file.

        Args:
            file_path: Path to the file to refresh.

        Returns:
            Updated metadata or None if extraction failed.
        """
        try:
            metadata = extract_metadata(file_path)
            with self._lock:
                self._metadata_index[file_path] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Failed to refresh metadata for {file_path}: {e}")
            return None

    def remove_file(self, file_path: str) -> bool:
        """
        Remove a file from the index.

        Args:
            file_path: Path to remove from index.

        Returns:
            True if file was in index and removed.
        """
        with self._lock:
            if file_path in self._metadata_index:
                del self._metadata_index[file_path]
                return True
        return False

    def clear_index(self) -> None:
        """Clear all indexed metadata."""
        with self._lock:
            self._metadata_index.clear()
            self._scan_complete = False


# Global scanner instance and lock for thread-safe singleton
_scanner: LoraMetadataScanner | None = None
_scanner_lock: threading.Lock = threading.Lock()


def get_scanner() -> LoraMetadataScanner:
    """
    Get the global scanner instance.

    Uses double-checked locking to ensure thread-safe singleton creation.

    Returns:
        The global LoraMetadataScanner instance.
    """
    global _scanner
    if _scanner is None:
        with _scanner_lock:
            # Double-check after acquiring lock
            if _scanner is None:
                _scanner = LoraMetadataScanner()
    return _scanner


def start_background_scan() -> None:
    """
    Start the background LoRA metadata scan.

    This is the main entry point for initiating the scan at application startup.
    """
    scanner = get_scanner()
    scanner.start_scan(blocking=False)


def get_all_library_data() -> list[dict[str, Any]]:
    """
    Get all LoRA metadata for the library page.

    Returns:
        List of metadata dictionaries sorted by filename with fallback values applied.
    """
    scanner = get_scanner()
    index = scanner.metadata_index

    # Convert to list, apply fallbacks, and sort by filename
    library_data = []
    for metadata in index.values():
        # Apply fallback values for missing metadata
        processed = dict(metadata)
        if not processed.get('base_model'):
            processed['base_model'] = 'Unknown'
        if not processed.get('trigger_words'):
            processed['trigger_words'] = []
        if not processed.get('description'):
            processed['description'] = ''

        library_data.append(processed)

    library_data.sort(key=lambda x: x.get('filename', '').lower())

    return library_data


def get_distinct_base_models() -> list[str]:
    """
    Get all distinct base model types from the library.

    Returns:
        Sorted list of unique base model names.
    """
    scanner = get_scanner()
    index = scanner.metadata_index

    base_models = set()
    for metadata in index.values():
        model = metadata.get('base_model')
        if model:
            base_models.add(model)

    # Return sorted list with 'Unknown' at the end if present
    models = sorted(base_models)
    if 'Unknown' in models:
        models.remove('Unknown')
        models.append('Unknown')

    return models


def get_trigger_words_for_filename(filename: str) -> list[str]:
    """
    Get trigger words for a specific LoRA by filename.

    Args:
        filename: The LoRA filename (without path).

    Returns:
        List of trigger words, or empty list if not found.
    """
    scanner = get_scanner()
    results = scanner.get_metadata_by_filename(filename)

    if results:
        return results[0].get('trigger_words', [])

    return []


def search_library(
    query: str = '',
    base_model_filter: str = ''
) -> list[dict[str, Any]]:
    """
    Search and filter the LoRA library.

    Args:
        query: Text search query (searches all metadata fields).
        base_model_filter: Filter by specific base model (empty for all).

    Returns:
        List of matching metadata dictionaries.
    """
    library_data = get_all_library_data()
    results = []

    query_lower = query.lower().strip()

    for metadata in library_data:
        # Apply base model filter
        if base_model_filter:
            if metadata.get('base_model') != base_model_filter:
                continue

        # Apply text search
        if query_lower:
            # Build searchable text from all fields
            searchable_parts = [
                metadata.get('filename', ''),
                metadata.get('base_model', ''),
                metadata.get('description', ''),
                ' '.join(metadata.get('trigger_words', [])),
                ' '.join(metadata.get('characters', [])),
                ' '.join(metadata.get('styles', [])),
            ]
            searchable_text = ' '.join(str(p) for p in searchable_parts).lower()

            if query_lower not in searchable_text:
                continue

        results.append(metadata)

    return results
