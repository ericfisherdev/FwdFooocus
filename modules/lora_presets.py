"""
LoRA Preset Management Module

This module provides functionality to save, load, list, and delete LoRA presets.
LoRA presets store the current LoRA selection (files and weights) for quick restoration.

Preset Format (JSON):
{
    "preset_name": "My Preset",
    "created_date": "2025-11-17",
    "loras": [
        [true, "lora_file1.safetensors", 0.8],
        [true, "lora_file2.safetensors", 0.6],
        [true, "None", 1.0],
        [true, "None", 1.0],
        [true, "None", 1.0]
    ]
}

Each LoRA entry: [enabled (bool), filename (str), weight (float)]
"""

import os
import json
import re
from datetime import datetime
from typing import Optional, Any


def get_preset_directory() -> str:
    """
    Get the LoRA presets directory path.
    Creates the directory if it doesn't exist.

    Returns:
        str: Absolute path to the lora_presets directory
    """
    # Get project root (parent of modules directory)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    preset_dir = os.path.join(project_root, 'lora_presets')

    # Create directory if it doesn't exist
    os.makedirs(preset_dir, exist_ok=True)

    return preset_dir


def sanitize_preset_name(name: str) -> str:
    """
    Sanitize preset name to be safe for filesystem.
    Removes or replaces invalid filename characters.

    Args:
        name: Raw preset name from user

    Returns:
        str: Sanitized filename-safe preset name
    """
    # Replace invalid characters with underscores
    # Invalid chars for most filesystems: \ / : * ? " < > |
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', name)

    # Remove leading/trailing whitespace and dots (can cause issues)
    sanitized = sanitized.strip('. ')

    # Ensure name is not empty after sanitization
    if not sanitized:
        sanitized = "unnamed_preset"

    return sanitized


def save_preset(
    preset_name: str,
    loras: list[tuple[bool, str, float]]
) -> tuple[bool, str]:
    """
    Save a LoRA preset to disk.

    Args:
        preset_name: Name for the preset
        loras: List of LoRA configurations, each as (enabled, filename, weight)
               Example: [(True, "lora1.safetensors", 0.8), (True, "None", 1.0), ...]

    Returns:
        tuple[bool, str]: (success, message)
            success: True if saved successfully, False otherwise
            message: Success message or error description
    """
    try:
        # Sanitize the preset name
        safe_name = sanitize_preset_name(preset_name)

        # Get preset directory
        preset_dir = get_preset_directory()

        # Build preset file path
        preset_file = os.path.join(preset_dir, f"{safe_name}.json")

        # Build preset data structure
        preset_data = {
            "preset_name": preset_name,  # Store original name
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loras": loras
        }

        # Write to file with proper formatting
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

        return True, f"Preset '{preset_name}' saved successfully to: {preset_file}"

    except PermissionError:
        return False, f"Permission denied: Cannot write to {preset_dir}"
    except OSError as e:
        return False, f"File system error: {str(e)}"
    except Exception as e:
        return False, f"Failed to save preset: {str(e)}"


def load_preset(preset_name: str) -> tuple[bool, Optional[list[tuple[bool, str, float]]], str]:
    """
    Load a LoRA preset from disk.

    Args:
        preset_name: Name of the preset to load

    Returns:
        tuple[bool, Optional[list], str]: (success, loras, message)
            success: True if loaded successfully, False otherwise
            loras: List of LoRA configurations if successful, None otherwise
            message: Success message or error description
    """
    try:
        # Sanitize the preset name
        safe_name = sanitize_preset_name(preset_name)

        # Get preset directory
        preset_dir = get_preset_directory()

        # Build preset file path
        preset_file = os.path.join(preset_dir, f"{safe_name}.json")

        # Check if file exists
        if not os.path.exists(preset_file):
            return False, None, f"Preset '{preset_name}' not found"

        # Load and parse JSON
        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)

        # Extract LoRAs list
        loras = preset_data.get('loras', [])

        # Validate structure (should be list of [bool, str, float])
        if not isinstance(loras, list):
            return False, None, f"Invalid preset format: 'loras' must be a list"

        # Convert to tuples and validate each entry
        loras_tuples = []
        for idx, lora in enumerate(loras):
            # Convert to tuple if it's a list
            if isinstance(lora, list):
                lora_tuple = tuple(lora)
            else:
                lora_tuple = lora

            # Validate that it's a tuple/sequence with exactly 3 elements
            if not isinstance(lora_tuple, (tuple, list)) or len(lora_tuple) != 3:
                raise ValueError(
                    f"Invalid LoRA entry at index {idx} in preset '{preset_name}': "
                    f"Expected tuple of 3 elements, got {type(lora_tuple).__name__} with {len(lora_tuple) if hasattr(lora_tuple, '__len__') else 'unknown'} elements. "
                    f"Entry: {lora_tuple}"
                )

            # Validate element types: (bool/int/float, str, float)
            enabled, filename, weight = lora_tuple

            # First element should be numeric (bool, int, or float for enabled state)
            if not isinstance(enabled, (bool, int, float)):
                raise ValueError(
                    f"Invalid LoRA entry at index {idx} in preset '{preset_name}': "
                    f"First element (enabled) must be bool/int/float, got {type(enabled).__name__}. "
                    f"Entry: {lora_tuple}"
                )

            # Second element should be string (filename)
            if not isinstance(filename, str):
                raise ValueError(
                    f"Invalid LoRA entry at index {idx} in preset '{preset_name}': "
                    f"Second element (filename) must be string, got {type(filename).__name__}. "
                    f"Entry: {lora_tuple}"
                )

            # Third element should be numeric (weight)
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Invalid LoRA entry at index {idx} in preset '{preset_name}': "
                    f"Third element (weight) must be int/float, got {type(weight).__name__}. "
                    f"Entry: {lora_tuple}"
                )

            loras_tuples.append(lora_tuple)

        return True, loras_tuples, f"Preset '{preset_name}' loaded successfully"

    except json.JSONDecodeError:
        return False, None, f"Invalid JSON in preset file: {preset_file}"
    except Exception as e:
        return False, None, f"Failed to load preset: {str(e)}"


def list_presets() -> list[str]:
    """
    List all available LoRA presets.

    Returns:
        list[str]: List of preset names (without .json extension), sorted alphabetically
    """
    try:
        # Get preset directory
        preset_dir = get_preset_directory()

        # List all .json files
        preset_files = [
            f[:-5]  # Remove .json extension
            for f in os.listdir(preset_dir)
            if f.endswith('.json') and os.path.isfile(os.path.join(preset_dir, f))
        ]

        # Sort alphabetically
        preset_files.sort()

        return preset_files

    except Exception as e:
        print(f"Error listing presets: {str(e)}")
        return []


def delete_preset(preset_name: str) -> tuple[bool, str]:
    """
    Delete a LoRA preset from disk.

    Args:
        preset_name: Name of the preset to delete

    Returns:
        tuple[bool, str]: (success, message)
            success: True if deleted successfully, False otherwise
            message: Success message or error description
    """
    try:
        # Sanitize the preset name
        safe_name = sanitize_preset_name(preset_name)

        # Get preset directory
        preset_dir = get_preset_directory()

        # Build preset file path
        preset_file = os.path.join(preset_dir, f"{safe_name}.json")

        # Check if file exists
        if not os.path.exists(preset_file):
            return False, f"Preset '{preset_name}' not found"

        # Delete the file
        os.remove(preset_file)

        return True, f"Preset '{preset_name}' deleted successfully"

    except PermissionError:
        return False, f"Permission denied: Cannot delete {preset_file}"
    except OSError as e:
        return False, f"File system error: {str(e)}"
    except Exception as e:
        return False, f"Failed to delete preset: {str(e)}"


def preset_exists(preset_name: str) -> bool:
    """
    Check if a preset already exists.

    Args:
        preset_name: Name of the preset to check

    Returns:
        bool: True if preset exists, False otherwise
    """
    try:
        safe_name = sanitize_preset_name(preset_name)
        preset_dir = get_preset_directory()
        preset_file = os.path.join(preset_dir, f"{safe_name}.json")
        return os.path.exists(preset_file)
    except Exception:
        return False


def get_preset_info(preset_name: str) -> Optional[dict[str, Any]]:
    """
    Get metadata about a preset without loading the full LoRA list.

    Args:
        preset_name: Name of the preset

    Returns:
        Optional[dict]: Preset metadata (name, created_date) or None if not found
    """
    try:
        safe_name = sanitize_preset_name(preset_name)
        preset_dir = get_preset_directory()
        preset_file = os.path.join(preset_dir, f"{safe_name}.json")

        if not os.path.exists(preset_file):
            return None

        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)

        # Count valid LoRA entries (filter out malformed entries and "None" entries)
        lora_count = 0
        for lora in preset_data.get('loras', []):
            # Check if entry is a valid sequence with at least 2 elements
            if isinstance(lora, (list, tuple)) and len(lora) >= 2:
                # Check if the second element (filename) is not "None"
                if lora[1] != 'None':
                    lora_count += 1
            # Silently ignore malformed entries (they will be caught by load_preset validation)

        return {
            'preset_name': preset_data.get('preset_name', preset_name),
            'created_date': preset_data.get('created_date', 'Unknown'),
            'lora_count': lora_count
        }

    except Exception as e:
        print(f"Error getting preset info: {str(e)}")
        return None
