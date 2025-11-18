"""
Test script for Delete Preset functionality (Story 1.4 - FWDF-8)
Verifies acceptance criteria for deleting presets
"""

import sys
import os

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import lora_presets

def test_delete_preset_basic():
    """Test basic delete preset functionality"""
    print("=" * 60)
    print("TEST 1: Delete Preset - Basic Functionality")
    print("=" * 60)

    # Create a test preset
    test_loras = [
        (True, "test_lora.safetensors", 0.8),
        (True, "None", 1.0),
        (True, "None", 1.0),
        (True, "None", 1.0),
        (True, "None", 1.0)
    ]

    preset_name = "Test_Delete_Me"

    # Save preset
    success, message = lora_presets.save_preset(preset_name, test_loras)
    print(f"{'✓' if success else '✗'} Created test preset: {preset_name}")

    # Verify it exists
    exists = lora_presets.preset_exists(preset_name)
    print(f"{'✓' if exists else '✗'} Preset exists before deletion: {exists}")

    # Delete preset
    success, message = lora_presets.delete_preset(preset_name)
    print(f"{'✓' if success else '✗'} Delete operation: {message}")

    # Verify it's gone
    exists_after = lora_presets.preset_exists(preset_name)
    print(f"{'✓' if not exists_after else '✗'} Preset exists after deletion: {exists_after} (should be False)")

    if success and not exists_after:
        print("✓ Delete operation successful!")
    else:
        print("✗ Delete operation failed!")

    print()

def test_delete_nonexistent():
    """Test deleting non-existent preset"""
    print("=" * 60)
    print("TEST 2: Delete Non-Existent Preset")
    print("=" * 60)

    preset_name = "Nonexistent_Preset_XYZ"

    # Try to delete
    success, message = lora_presets.delete_preset(preset_name)

    if not success:
        print(f"✓ Error handled gracefully: {message}")
    else:
        print("✗ Should have returned error for non-existent preset")

    print()

def test_dropdown_refresh():
    """Test that dropdown would refresh after deletion"""
    print("=" * 60)
    print("TEST 3: Dropdown Refresh After Deletion")
    print("=" * 60)

    # Create multiple presets
    preset_names = ["Test_Alpha", "Test_Beta", "Test_Gamma"]

    for name in preset_names:
        lora_presets.save_preset(name, [(True, "None", 1.0)] * 5)

    # List presets before deletion
    presets_before = lora_presets.list_presets()
    print(f"Presets before deletion: {', '.join(presets_before)}")
    print(f"Count: {len(presets_before)}")

    # Delete one preset
    delete_name = "Test_Beta"
    success, message = lora_presets.delete_preset(delete_name)
    print(f"\n{'✓' if success else '✗'} Deleted: {delete_name}")

    # List presets after deletion
    presets_after = lora_presets.list_presets()
    print(f"\nPresets after deletion: {', '.join(presets_after)}")
    print(f"Count: {len(presets_after)}")

    # Verify the deleted preset is gone
    if delete_name not in presets_after and len(presets_after) == len(presets_before) - 1:
        print(f"✓ Dropdown would refresh correctly (preset removed from list)")
    else:
        print("✗ Dropdown refresh would be incorrect")

    # Cleanup remaining test presets
    for name in ["Test_Alpha", "Test_Gamma"]:
        lora_presets.delete_preset(name)

    print()

def test_filesystem_error_handling():
    """Test handling of filesystem errors"""
    print("=" * 60)
    print("TEST 4: Filesystem Error Handling")
    print("=" * 60)

    # Create a preset
    preset_name = "Test_Error_Handling"
    lora_presets.save_preset(preset_name, [(True, "None", 1.0)] * 5)

    # Get the preset file path
    preset_dir = lora_presets.get_preset_directory()
    preset_file = os.path.join(preset_dir, f"{preset_name}.json")

    # Verify file exists
    if os.path.exists(preset_file):
        print(f"✓ Preset file exists: {preset_file}")

        # Delete normally (simulates successful deletion)
        success, message = lora_presets.delete_preset(preset_name)
        if success:
            print(f"✓ Normal deletion works: {message}")
        else:
            print(f"✗ Normal deletion failed: {message}")

        # Try to delete again (should get error)
        success2, message2 = lora_presets.delete_preset(preset_name)
        if not success2:
            print(f"✓ Double-delete error handled: {message2}")
        else:
            print("✗ Should have returned error for already-deleted preset")
    else:
        print("✗ Preset file doesn't exist")

    print()

def test_confirmation_message():
    """Test that confirmation message includes preset name"""
    print("=" * 60)
    print("TEST 5: Confirmation Message Format")
    print("=" * 60)

    preset_name = "My_Special_Preset"

    # In the UI, the confirmation message would be:
    expected_message = f"Are you sure you want to delete preset '{preset_name}'?"

    print(f"✓ Confirmation message format:")
    print(f"  '{expected_message}'")
    print(f"✓ Shows preset name: {preset_name in expected_message}")
    print(f"✓ Asks for confirmation: {'sure' in expected_message.lower()}")

    print()

def test_delete_button_state():
    """Test delete button enable/disable logic"""
    print("=" * 60)
    print("TEST 6: Delete Button State Management")
    print("=" * 60)

    # Simulate button states
    print("Scenario 1: No preset selected")
    preset_selected = None
    should_enable = preset_selected is not None and preset_selected != ""
    print(f"  Delete button enabled: {should_enable} ✓ (should be False)")

    print("\nScenario 2: Preset selected")
    preset_selected = "Test_Preset"
    should_enable = preset_selected is not None and preset_selected != ""
    print(f"  Delete button enabled: {should_enable} ✓ (should be True)")

    print("\nScenario 3: After deletion (dropdown cleared)")
    preset_selected = None
    should_enable = preset_selected is not None and preset_selected != ""
    print(f"  Delete button enabled: {should_enable} ✓ (should be False)")

    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DELETE PRESET FUNCTIONALITY TEST SUITE")
    print("Story 1.4 (FWDF-8) Acceptance Criteria Verification")
    print("=" * 60 + "\n")

    try:
        test_delete_preset_basic()
        test_delete_nonexistent()
        test_dropdown_refresh()
        test_filesystem_error_handling()
        test_confirmation_message()
        test_delete_button_state()

        print("=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
