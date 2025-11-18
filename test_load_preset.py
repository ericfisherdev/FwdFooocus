"""
Test script for Load Preset functionality (Story 1.3 - FWDF-7)
Verifies acceptance criteria for loading presets
"""

import sys
import os

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import lora_presets

def test_load_preset_basic():
    """Test basic load preset functionality"""
    print("=" * 60)
    print("TEST 1: Load Preset - Basic Functionality")
    print("=" * 60)

    # First, create a test preset
    test_loras = [
        (True, "character_lora_v2.safetensors", 0.8),
        (True, "anime_style.safetensors", 0.6),
        (False, "detail_enhancer.safetensors", 0.4),
        (True, "None", 1.0),
        (True, "None", 1.0)
    ]

    preset_name = "Test_Load_Preset"

    # Save preset
    success, message = lora_presets.save_preset(preset_name, test_loras)
    print(f"{'✓' if success else '✗'} Created test preset: {message}")

    # Load preset
    success, loaded_loras, message = lora_presets.load_preset(preset_name)
    print(f"{'✓' if success else '✗'} Loaded preset: {message}")

    if success:
        print(f"\n✓ Loaded {len(loaded_loras)} LoRA slots")

        # Verify data matches
        if loaded_loras == test_loras:
            print("✓ All LoRA slots match saved data")
            for i, (enabled, filename, weight) in enumerate(loaded_loras, 1):
                status = "enabled" if enabled else "disabled"
                print(f"  Slot {i}: {filename} @ {weight} ({status})")
        else:
            print("✗ Data mismatch!")
            print(f"  Expected: {test_loras}")
            print(f"  Got: {loaded_loras}")

    print()

def test_load_preset_alphabetical():
    """Test that presets are listed alphabetically"""
    print("=" * 60)
    print("TEST 2: Preset List Alphabetical Order")
    print("=" * 60)

    # Create multiple presets with different names
    test_presets = ["Zebra_Preset", "Alpha_Preset", "Beta_Preset"]

    for name in test_presets:
        lora_presets.save_preset(name, [(True, "None", 1.0)] * 5)

    # Get preset list
    presets = lora_presets.list_presets()

    # Check if alphabetically sorted
    sorted_presets = sorted(presets)
    if presets == sorted_presets:
        print("✓ Presets are listed alphabetically")
        print(f"  Order: {', '.join(presets)}")
    else:
        print("✗ Presets are NOT alphabetically sorted")
        print(f"  Current: {presets}")
        print(f"  Expected: {sorted_presets}")

    # Cleanup
    for name in test_presets:
        lora_presets.delete_preset(name)

    print()

def test_missing_lora_files():
    """Test handling of missing LoRA files"""
    print("=" * 60)
    print("TEST 3: Missing LoRA File Handling")
    print("=" * 60)

    # Create preset with non-existent LoRA files
    test_loras = [
        (True, "nonexistent_lora_1.safetensors", 0.7),
        (True, "nonexistent_lora_2.safetensors", 0.5),
        (True, "None", 1.0),
        (True, "None", 1.0),
        (True, "None", 1.0)
    ]

    preset_name = "Test_Missing_Files"

    # Save preset
    success, message = lora_presets.save_preset(preset_name, test_loras)
    print(f"{'✓' if success else '✗'} Created preset with non-existent files: {message}")

    # Load preset
    success, loaded_loras, message = lora_presets.load_preset(preset_name)
    print(f"{'✓' if success else '✗'} Loaded preset: {message}")

    if success:
        print("✓ Preset loaded despite missing files (graceful handling)")
        print(f"  Note: In the UI, missing files would be set to 'None' with warning")
        print(f"  Loaded LoRAs: {loaded_loras}")

    print()

def test_empty_slots():
    """Test that empty slots (None, 1.0) are preserved"""
    print("=" * 60)
    print("TEST 4: Empty Slot Preservation")
    print("=" * 60)

    # Create preset with mix of filled and empty slots
    test_loras = [
        (True, "lora1.safetensors", 0.8),
        (True, "None", 1.0),
        (False, "lora2.safetensors", 0.5),
        (True, "None", 1.0),
        (True, "None", 1.0)
    ]

    preset_name = "Test_Empty_Slots"

    # Save and load
    lora_presets.save_preset(preset_name, test_loras)
    success, loaded_loras, message = lora_presets.load_preset(preset_name)

    if success and loaded_loras == test_loras:
        print("✓ Empty slots (None, 1.0) preserved correctly")

        # Count empty vs filled slots
        empty_count = sum(1 for _, filename, _ in loaded_loras if filename == "None")
        filled_count = sum(1 for _, filename, _ in loaded_loras if filename != "None")

        print(f"  Empty slots: {empty_count}")
        print(f"  Filled slots: {filled_count}")
    else:
        print("✗ Empty slots not preserved correctly")

    print()

def test_load_nonexistent():
    """Test loading non-existent preset"""
    print("=" * 60)
    print("TEST 5: Load Non-Existent Preset")
    print("=" * 60)

    success, loras, message = lora_presets.load_preset("Nonexistent_Preset_XYZ")

    if not success and loras is None:
        print(f"✓ Non-existent preset handled gracefully: {message}")
    else:
        print("✗ Should have returned error for non-existent preset")

    print()

def cleanup_test_presets():
    """Clean up all test presets"""
    print("=" * 60)
    print("CLEANUP: Removing Test Presets")
    print("=" * 60)

    test_names = ["Test_Load_Preset", "Test_Missing_Files", "Test_Empty_Slots"]

    for name in test_names:
        if lora_presets.preset_exists(name):
            success, message = lora_presets.delete_preset(name)
            print(f"{'✓' if success else '✗'} Deleted: {name}")

    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LOAD PRESET FUNCTIONALITY TEST SUITE")
    print("Story 1.3 (FWDF-7) Acceptance Criteria Verification")
    print("=" * 60 + "\n")

    try:
        test_load_preset_basic()
        test_load_preset_alphabetical()
        test_missing_lora_files()
        test_empty_slots()
        test_load_nonexistent()
        cleanup_test_presets()

        print("=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
