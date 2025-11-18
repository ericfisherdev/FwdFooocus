#!/usr/bin/env python3
"""
CalVer version updater for FwdFooocus.

This script automatically updates the version in fooocus_version.py
using Calendar Versioning (CalVer) format: YYYY.MM.MICRO

The MICRO component increments with each PR merge in the same month
and resets to 0 when the month changes.
"""

import os
import re
from datetime import datetime
from pathlib import Path


def read_version_file(file_path: Path) -> str:
    """Read the current version from the version file."""
    return file_path.read_text()


def extract_version(content: str) -> str:
    """Extract version string from file content."""
    match = re.search(r"version = ['\"]([^'\"]+)['\"]", content)
    if not match:
        raise ValueError("Could not find version string in file")
    return match.group(1)


def parse_calver(version: str) -> tuple[int, int, int]:
    """Parse CalVer string into (year, month, micro) components."""
    parts = version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid CalVer format: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def calculate_new_version(current_version: str) -> str:
    """
    Calculate new CalVer version based on current date.

    If current month matches version month, increment MICRO.
    Otherwise, reset to current YYYY.MM.0
    """
    year, month, micro = parse_calver(current_version)

    now = datetime.now()
    current_year = now.year
    current_month = now.month

    if year == current_year and month == current_month:
        # Same month, increment micro
        new_micro = micro + 1
    else:
        # New month, reset micro
        new_micro = 0

    return f"{current_year}.{current_month}.{new_micro}"


def update_version_in_content(content: str, new_version: str) -> str:
    """Replace version string in file content."""
    return re.sub(
        r"(version = ['\"])([^'\"]+)(['\"])",
        rf"\g<1>{new_version}\g<3>",
        content
    )


def write_version_file(file_path: Path, content: str) -> None:
    """Write updated content back to version file."""
    file_path.write_text(content)


def main() -> None:
    """Main entry point for version update script."""
    # Get repository root (script is in .github/scripts/)
    repo_root = Path(__file__).parent.parent.parent
    version_file = repo_root / "fooocus_version.py"

    if not version_file.exists():
        raise FileNotFoundError(f"Version file not found: {version_file}")

    # Read current version
    content = read_version_file(version_file)
    current_version = extract_version(content)

    # Calculate new version
    new_version = calculate_new_version(current_version)

    # Update file
    updated_content = update_version_in_content(content, new_version)
    write_version_file(version_file, updated_content)

    # Print version info for GitHub Actions
    print(f"Updated version: {current_version} -> {new_version}")

    # Write outputs to GitHub Actions environment file
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"version={new_version}\n")
            f.write(f"previous_version={current_version}\n")


if __name__ == "__main__":
    main()
