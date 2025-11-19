"""
LoRA Metadata Extraction Module

Extracts metadata from .safetensors LoRA files including base model version,
trigger words, descriptions, character names, and style information.

This module provides the foundation for the LoRA Library feature (Epic 2).
"""

import json
import logging
import os
import re
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
    r'sdxl[-_]?1\.?0|stable[-_]?diffusion[-_]?xl': 'SDXL 1.0',
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
        if key in metadata and metadata[key]:
            return str(metadata[key]).strip()
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
        if key in metadata and metadata[key]:
            value = metadata[key]
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
        if key in metadata and metadata[key]:
            value = metadata[key]
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
            return has_lora_keys or len(keys) > 0
    except Exception:
        return False
