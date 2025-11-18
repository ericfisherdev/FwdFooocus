# FwdFooocus Versioning System

FwdFooocus uses **Calendar Versioning (CalVer)** to automatically manage version numbers based on when changes are released.

## Version Format

Versions follow the format: `YYYY.MM.MICRO`

- **YYYY**: 4-digit year (e.g., 2025)
- **MM**: Month number without leading zero (e.g., 11 for November, 1 for January)
- **MICRO**: Auto-incrementing number that resets each month (starts at 0)

### Examples

- `2025.11.0` - First release in November 2025
- `2025.11.1` - Second release in November 2025
- `2025.12.0` - First release in December 2025

## How It Works

### Automatic Version Updates

The version number is automatically updated when a pull request is merged to the `main` branch:

1. **PR Merge Trigger**: When a PR is merged to `main`, the `update_version.yml` GitHub Actions workflow runs
2. **Version Calculation**: The `update_version.py` script:
   - Reads the current version from `fooocus_version.py`
   - Compares the version month with the current month
   - If same month: increments MICRO (e.g., `2025.11.0` → `2025.11.1`)
   - If new month: resets to current `YYYY.MM.0` (e.g., `2025.11.5` → `2025.12.0`)
3. **Commit**: The updated version is committed back to `main` with message `chore: bump version to X.Y.Z [skip ci]`

### Manual Testing

To test the version update script locally:

```bash
python .github/scripts/update_version.py
```

This will update `fooocus_version.py` with the next version number.

## Version File Location

The version is stored in `fooocus_version.py`:

```python
# CalVer format: YYYY.MM.MICRO
# MICRO increments with each PR merge in the same month
version = '2025.11.0'
```

## Used By

The version number is imported and displayed in:

- `webui.py` - UI title: "Fooocus {version}"
- `launch.py` - Application startup
- `modules/meta_parser.py` - Metadata generation
- `modules/async_worker.py` - Worker information

## Benefits of CalVer

1. **Date Context**: Immediately know when a version was released
2. **Automatic**: No manual version bumping needed
3. **Consistent**: Every PR merge creates a new version
4. **Simple**: No need to decide between major/minor/patch
5. **Chronological**: Versions are inherently ordered by time

## Migrating from SemVer

Previous versions used Semantic Versioning (e.g., `2.5.5`). The migration to CalVer happened on 2025-11-18.

## Workflow Details

The GitHub Actions workflow (`.github/workflows/update_version.yml`) runs with the following permissions:

- **Trigger**: Pull request merged to `main`
- **Permissions**: `contents: write` (to commit version changes)
- **Python Version**: 3.10
- **Skip CI**: The version commit includes `[skip ci]` to prevent recursive workflow triggers
