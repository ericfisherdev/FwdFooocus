"""
Test script for LoRA Preset functionality
Run this to verify Story 1.1 (FWDF-5) implementation
"""

import sys
import os

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import lora_presets

def test_preset_directory():
    """Test that preset directory is created"""
    print("=" * 60)
    print("TEST 1: Preset Directory Creation")
    print("=" * 60)

    preset_dir = lora_presets.get_preset_directory()
    print(f"✓ Preset directory: {preset_dir}")
    print(f"✓ Directory exists: {os.path.exists(preset_dir)}")
    print(f"✓ Directory is writable: {os.access(preset_dir, os.W_OK)}")
    print()

def test_sanitize_name():
    """Test preset name sanitization"""
    print("=" * 60)
    print("TEST 2: Preset Name Sanitization")
    print("=" * 60)

    test_cases = [
        ("Normal Name", "Normal Name"),
        ("Name/With\\Slashes", "Name_With_Slashes"),
        ("Name:With*Special?Chars", "Name_With_Special_Chars"),
        ("  Leading Spaces  ", "Leading Spaces"),
        (".hidden.preset.", "hidden.preset"),
        ("", "unnamed_preset")
    ]

    for input_name, expected in test_cases:
        result = lora_presets.sanitize_preset_name(input_name)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_name}' → '{result}' (expected: '{expected}')")
    print()

def test_save_load_preset():
    """Test saving and loading a preset"""
    print("=" * 60)
    print("TEST 3: Save and Load Preset")
    print("=" * 60)

    # Test data - typical LoRA configuration
    test_loras = [
        (True, "character_lora_v2.safetensors", 0.8),
        (True, "anime_style.safetensors", 0.6),
        (True, "detail_enhancer.safetensors", 0.4),
        (True, "None", 1.0),
        (True, "None", 1.0)
    ]

    preset_name = "Test_Anime_Style"

    # Save preset
    print(f"Saving preset '{preset_name}'...")
    success, message = lora_presets.save_preset(preset_name, test_loras)
    print(f"{'✓' if success else '✗'} {message}")

    # Load preset
    print(f"\nLoading preset '{preset_name}'...")
    success, loaded_loras, message = lora_presets.load_preset(preset_name)
    print(f"{'✓' if success else '✗'} {message}")

    if success:
        print(f"\nLoaded {len(loaded_loras)} LoRA slots:")
        for i, (enabled, filename, weight) in enumerate(loaded_loras, 1):
            print(f"  Slot {i}: {filename} @ {weight} (enabled: {enabled})")

        # Verify data matches
        if loaded_loras == test_loras:
            print("\n✓ Loaded data matches saved data!")
        else:
            print("\n✗ Data mismatch!")
    print()

def test_list_presets():
    """Test listing presets"""
    print("=" * 60)
    print("TEST 4: List Presets")
    print("=" * 60)

    presets = lora_presets.list_presets()
    print(f"Found {len(presets)} preset(s):")
    for preset in presets:
        info = lora_presets.get_preset_info(preset)
        if info:
            print(f"  • {preset}")
            print(f"      Created: {info['created_date']}")
            print(f"      LoRAs: {info['lora_count']}")
        else:
            print(f"  • {preset}")
    print()

def test_preset_exists():
    """Test preset existence check"""
    print("=" * 60)
    print("TEST 5: Preset Existence Check")
    print("=" * 60)

    exists = lora_presets.preset_exists("Test_Anime_Style")
    print(f"✓ Preset 'Test_Anime_Style' exists: {exists}")

    exists = lora_presets.preset_exists("Nonexistent_Preset")
    print(f"✓ Preset 'Nonexistent_Preset' exists: {exists}")
    print()

def test_delete_preset():
    """Test deleting a preset"""
    print("=" * 60)
    print("TEST 6: Delete Preset")
    print("=" * 60)

    preset_name = "Test_Anime_Style"

    print(f"Deleting preset '{preset_name}'...")
    success, message = lora_presets.delete_preset(preset_name)
    print(f"{'✓' if success else '✗'} {message}")

    # Verify deletion
    exists = lora_presets.preset_exists(preset_name)
    print(f"✓ Preset exists after deletion: {exists} (should be False)")
    print()

def test_error_handling():
    """Test error handling"""
    print("=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)

    # Try to load non-existent preset
    print("Loading non-existent preset...")
    success, loras, message = lora_presets.load_preset("Nonexistent_Preset")
    print(f"{'✓' if not success else '✗'} {message}")

    # Try to delete non-existent preset
    print("\nDeleting non-existent preset...")
    success, message = lora_presets.delete_preset("Nonexistent_Preset")
    print(f"{'✓' if not success else '✗'} {message}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LoRA PRESET MODULE TEST SUITE")
    print("Story 1.1 (FWDF-5) Acceptance Criteria Verification")
    print("=" * 60 + "\n")

    try:
        test_preset_directory()
        test_sanitize_name()
        test_save_load_preset()
        test_list_presets()
        test_preset_exists()
        test_delete_preset()
        test_error_handling()

        print("=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
