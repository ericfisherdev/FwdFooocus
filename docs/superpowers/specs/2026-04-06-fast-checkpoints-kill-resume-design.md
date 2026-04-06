# Design Spec: Fast Checkpoint Storage, Kill on Browser Close, Resume on Close

**Date:** 2026-04-06
**Status:** Approved
**Scope:** Three independent features for FwdFooocus

---

## Feature 1: Fast Checkpoint Storage

### Purpose

Allow users to store their full checkpoint library on a slow/large drive (HDD, external USB) while frequently used checkpoints are automatically cached on a fast drive (SSD/NVMe). The first load of a checkpoint copies it to the fast drive; subsequent loads use the cached copy.

### Config

New option in `modules/config.py`:
- **`path_fast_checkpoints`** -- string path, default `None` (feature disabled).
- Added via `get_dir_or_set_default()`. Directory created only if the value is set.
- When set, this path is checked first for checkpoint files before falling back to `paths_checkpoints`.

### Architecture

New module: **`modules/fast_checkpoint.py`**

**`resolve_checkpoint_path(checkpoint_name: str) -> str`**

1. If `path_fast_checkpoints` is not configured, delegate to `get_file_from_folder_list(name, paths_checkpoints)`.
2. Check if `checkpoint_name` exists in `path_fast_checkpoints`. If yes, return that path.
3. Otherwise, resolve the original path via `get_file_from_folder_list(name, paths_checkpoints)`.
4. Copy the file to `path_fast_checkpoints` using a `.tmp` suffix, then `os.rename` to the final name (atomic on same filesystem).
5. Use `shutil.copy2` to preserve file metadata.
6. Log the copy with file size and elapsed time.
7. Return the new fast path.

**Error handling:**
- If the copy fails (disk full, permissions, etc.), log the error and fall back to loading from the original path. Do not block generation.
- If the original file doesn't exist, fall through to existing behavior (returns constructed path in `paths_checkpoints[0]`).

### Integration Points

- **`modules/default_pipeline.py` -- `refresh_base_model()`**: Replace `get_file_from_folder_list(name, paths_checkpoints)` with `resolve_checkpoint_path(name)`.
- **`modules/default_pipeline.py` -- `refresh_refiner_model()`**: Same replacement.

### Behavior

- Feature disabled by default (`path_fast_checkpoints` not set).
- First use of a checkpoint: blocked by copy time (user sees delay), then loads from fast drive.
- Subsequent uses: loads directly from fast drive, no copy.
- Files are **copied**, not moved. The slow drive remains a complete archive.

---

## Feature 2: Kill on Browser Close

### Purpose

When a user closes the browser after queueing a large batch, the batch continues running with no way to interact with it. This feature cancels remaining batch items when the browser disconnects.

### Architecture

**Client-side heartbeat:**
- JS `setInterval` in the Gradio page that POSTs `/heartbeat` every 5 seconds.
- Starts on page load, runs continuously.
- Added via `gr.HTML` with an inline `<script>` tag or Gradio's JS injection.

**Backend heartbeat tracking (in `webui.py`):**
- Global `last_heartbeat_time` initialized to `time.time()` at startup.
- `POST /heartbeat` endpoint: updates `last_heartbeat_time = time.time()`.
- `is_browser_connected(timeout_seconds=15) -> bool`: returns `True` if `time.time() - last_heartbeat_time < timeout_seconds`.

**Generation loop check (in `async_worker.py`):**
- In `handler()`, at the top of the batch image loop (~line 1281), before each image:
  - Call `is_browser_connected()`.
  - If `False`, log a warning and `break` the loop.
  - The current in-progress image (if any) finishes and is saved.
- Same check at the top of the enhancement loop (~line 1335).

### Behavior

- **Browser open:** heartbeat every 5s, checks always pass, batch runs normally.
- **Browser closed:** heartbeat stops. After ~15s (3 missed beats), the next between-image check cancels remaining batch items. The current image completes.
- **Browser reopened:** heartbeat resumes immediately. New generations work normally. Cancelled batches do not resume.
- **Single-user assumption:** one global heartbeat timestamp. No per-session tracking.
- **No mid-diffusion interruption:** the check only runs between images, not during sampling.

---

## Feature 3: Resume on Close

### Purpose

Preserve the user's prompt, settings, and LoRA configuration between browser sessions. When the user closes and reopens the browser, the UI restores to the state of their last generation. Supports per-base-model state so switching between model families (e.g., Pony vs SDXL) restores the appropriate settings for each.

### Config

New option in `modules/config.py`:
- **`default_base_model`** -- string, default `None`. Declares the base model family (e.g., `"pony"`, `"sdxl"`). Distinct from `default_model` which is the specific checkpoint filename. Used as the key for state storage/retrieval.

### Architecture

New module: **`modules/session_state.py`**

**SQLite database** at `./session_states.db`.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS session_states (
    base_model TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
```

**Functions:**
- `get_db_connection() -> sqlite3.Connection` -- lazily creates database and table on first access. Uses `check_same_thread=False` for thread safety.
- `save_state(base_model: str, state: dict) -> None` -- upserts the state for the given base model. Serializes state to JSON. Sets `updated_at` to current time.
- `load_state(base_model: str) -> dict | None` -- returns the saved state dict, or `None` if not found.

### State Captured

After each successful generation, the following is saved as a JSON dict keyed by `default_base_model`:

| Key | Type | Notes |
|-----|------|-------|
| `prompt` | str | |
| `negative_prompt` | str | |
| `style_selections` | list[str] | |
| `base_model_name` | str | Checkpoint filename |
| `refiner_model_name` | str | |
| `vae_name` | str | |
| `loras` | list[dict] | Each: `{enabled, filename, weight}` |
| `sampler` | str | |
| `scheduler` | str | |
| `steps` | int | |
| `cfg_scale` | float | |
| `width` | int | |
| `height` | int | |
| `seed` | int | **Only stored if != -1**. Omitted for random. |
| `performance` | str | |
| `image_number` | int | |
| `guidance_scale` | float | |
| `sharpness` | float | |
| `inpaint_engine` | str | |

### Save Trigger

In `async_worker.py`, after a generation completes successfully (after yielding `'finish'` in `handler()`):
1. Build the state dict from the task's parameters.
2. Call `save_state(config.default_base_model, state_dict)`.
3. `default_base_model` comes from config, not from the task's checkpoint filename.
4. If `default_base_model` is `None`, do not save (feature effectively disabled).

### Restore at Startup

In `webui.py`, during UI construction:
1. Call `load_state(config.default_base_model)`.
2. If a state is returned, use its values as defaults for Gradio component `value=` parameters instead of config defaults.
3. If no saved state exists (first run, new base model, or feature disabled), fall through to normal config defaults.
4. Seed defaults to -1 unless the saved state explicitly includes it.

### Model Switching Workflow

1. User sets `default_base_model = "pony"` in config, starts app.
2. App loads saved state for `"pony"` (if exists) -- pony-specific LoRAs, prompts, settings.
3. User generates images. State saved under `"pony"` after each generation.
4. User changes config to `default_base_model = "sdxl"`, restarts.
5. App loads saved state for `"sdxl"` -- different LoRAs, prompts, settings.
6. Switching back to `"pony"` later restores the pony state.

### Not Included

- State history or rollback (single most-recent state per base model).
- Live state saving on UI change (only on generation complete).
- Schema migration tooling (single table, version 1).

---

## Dependencies Between Features

These three features are **independent** and can be implemented in any order. They share no modules or data structures. The SQLite foundation from Feature 3 is scoped to `session_state.py` and does not affect the other features.

## Files Modified or Created

| File | Feature | Change |
|------|---------|--------|
| `modules/config.py` | 1, 3 | Add `path_fast_checkpoints` and `default_base_model` config options |
| `modules/fast_checkpoint.py` | 1 | **New.** `resolve_checkpoint_path()` function |
| `modules/default_pipeline.py` | 1 | Call `resolve_checkpoint_path()` instead of `get_file_from_folder_list()` for checkpoints |
| `modules/session_state.py` | 3 | **New.** SQLite state management |
| `modules/async_worker.py` | 2, 3 | Add browser-connected check in batch loop; save state after generation |
| `webui.py` | 2, 3 | Add heartbeat endpoint + JS; restore state at UI construction |
