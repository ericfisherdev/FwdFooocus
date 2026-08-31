"""Tests for modules.image_lists (FWDF-188).

image_lists is UI-agnostic (no gradio import) and takes its output root as
a parameter rather than reading modules.config internally, so these tests
exercise it directly against tmp_path fixtures -- no heavy dependencies
(torch etc.) are involved, so no import-guarding is needed here, matching
tests/test_wildcard_ui.py's approach for the other pure-logic module.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

# modules.image_lists imports modules.private_logger and modules.meta_parser,
# which transitively import gradio (not installed in every sandbox) and
# args_manager -- args_parser.parser.parse_args() runs against sys.argv at
# import time, so pytest's own CLI args must be hidden first. Mirrors
# tests/test_meta_confirm.py and tests/test_async_worker_inpaint.py; the
# gradio ImportError guard mirrors the try/except+skip convention used
# throughout tests/ for heavy/optional dependencies.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    import modules.private_logger as private_logger
    from modules.image_lists import (
        get_list_dir,
        get_metadata,
        list_exists,
        list_image_lists,
        record_metadata,
        sanitize_list_name,
        save_image_to_list,
    )
    _import_error = None
except ImportError as e:  # pragma: no cover - exercised only without gradio installed
    private_logger = None
    get_list_dir = get_metadata = list_exists = list_image_lists = None
    record_metadata = sanitize_list_name = save_image_to_list = None
    _import_error = e
finally:
    sys.argv = _original_argv

pytestmark = pytest.mark.skipif(
    _import_error is not None,
    reason=f'modules.image_lists unavailable: {_import_error}')


@pytest.fixture(autouse=True)
def _clear_registry_and_cache():
    """Each test gets a clean in-process metadata registry and log cache
    so state from one test can't leak into the next."""
    from modules import image_lists as image_lists_module
    image_lists_module._metadata_registry.clear()
    private_logger.log_cache.clear()
    yield
    image_lists_module._metadata_registry.clear()
    private_logger.log_cache.clear()


@pytest.fixture
def root_dir(tmp_path):
    lists_dir = tmp_path / 'lists'
    lists_dir.mkdir()
    return str(lists_dir)


def _make_png(path, parameters=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.new('RGB', (4, 4), color=(10, 20, 30))
    if parameters is not None:
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text('parameters', parameters)
        info.add_text('fooocus_scheme', 'fooocus')
        image.save(path, pnginfo=info)
    else:
        image.save(path)
    return path


class TestSanitizeListName:
    def test_strips_invalid_filesystem_characters(self):
        assert sanitize_list_name('my:list*name?') == 'my_list_name_'

    def test_strips_leading_trailing_dots_and_spaces(self):
        assert sanitize_list_name('  .my list. ') == 'my list'

    def test_empty_after_sanitization_falls_back_to_default(self):
        assert sanitize_list_name('...') == 'unnamed_list'


class TestListDirRoundTrip:
    def test_create_list_then_appears_in_list_image_lists(self, tmp_path, root_dir):
        src = _make_png(str(tmp_path / 'src' / 'a.png'))
        success, _ = save_image_to_list('My List', src, root_dir)

        assert success
        assert 'My List' in list_image_lists(root_dir)
        assert list_exists('My List', root_dir)

    def test_list_exists_false_for_missing_list(self, root_dir):
        assert not list_exists('nope', root_dir)

    def test_get_list_dir_rejects_traversal_outside_root(self, root_dir):
        # sanitize_list_name replaces '/' with '_', so a traversal-looking
        # name can't actually escape root_dir -- assert the resolved dir
        # stays inside it rather than expecting an outright rejection.
        resolved = get_list_dir('../../etc', root_dir)
        assert resolved is not None
        assert os.path.commonpath([os.path.realpath(root_dir), resolved]) == os.path.realpath(root_dir)

    def test_get_list_dir_rejects_symlink_escape(self, tmp_path, root_dir):
        outside = tmp_path / 'outside'
        outside.mkdir()
        symlink_path = os.path.join(root_dir, 'escape')
        os.symlink(str(outside), symlink_path)

        assert get_list_dir('escape', root_dir) is None


class TestSaveImageToList:
    def test_copies_file_byte_identical_and_writes_log_entry(self, tmp_path, root_dir):
        src = _make_png(str(tmp_path / 'src' / 'a.png'), parameters='{"prompt": "a cat"}')

        success, message = save_image_to_list(
            'cats', src, root_dir, metadata=[('Prompt', 'prompt', 'a cat')])

        assert success, message
        list_dir = get_list_dir('cats', root_dir)
        dest = os.path.join(list_dir, 'a.png')
        assert os.path.isfile(dest)
        assert open(dest, 'rb').read() == open(src, 'rb').read()

        log_path = os.path.join(list_dir, 'log.html')
        assert os.path.isfile(log_path)
        log_content = open(log_path, encoding='utf-8').read()
        assert 'a.png' in log_content
        assert 'a cat' in log_content

    def test_second_save_appends_newest_first(self, tmp_path, root_dir):
        src_a = _make_png(str(tmp_path / 'src' / 'a.png'))
        src_b = _make_png(str(tmp_path / 'src' / 'b.png'))

        save_image_to_list('cats', src_a, root_dir, metadata=[('Filename', 'filename', 'a.png')])
        save_image_to_list('cats', src_b, root_dir, metadata=[('Filename', 'filename', 'b.png')])

        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()

        assert 'a.png' in log_content and 'b.png' in log_content
        # newest first: b's entry appears before a's
        assert log_content.index('b.png') < log_content.index('a.png')
        assert os.path.isfile(os.path.join(list_dir, 'a.png'))
        assert os.path.isfile(os.path.join(list_dir, 'b.png'))

    def test_saving_same_filename_twice_overwrites_without_duplicating_log_entry(self, tmp_path, root_dir):
        src = _make_png(str(tmp_path / 'src' / 'a.png'))

        save_image_to_list('cats', src, root_dir, metadata=[('Filename', 'filename', 'a.png')])
        success, message = save_image_to_list('cats', src, root_dir, metadata=[('Filename', 'filename', 'a.png')])

        assert success
        assert 'Updated' in message
        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert log_content.count('a.png') == 1  # one <div id="a_png"> container, not two

    def test_append_survives_log_cache_being_cleared_cold_start(self, tmp_path, root_dir):
        """Simulates a server restart: the module-level log_cache in
        private_logger is empty, but log.html already exists on disk from a
        prior save -- appending must recover and preserve the existing
        entry, not clobber it."""
        src_a = _make_png(str(tmp_path / 'src' / 'a.png'))
        src_b = _make_png(str(tmp_path / 'src' / 'b.png'))

        save_image_to_list('cats', src_a, root_dir, metadata=[('Filename', 'filename', 'a.png')])

        # Simulate restart: process-local cache is gone, but the file is not.
        private_logger.log_cache.clear()

        save_image_to_list('cats', src_b, root_dir, metadata=[('Filename', 'filename', 'b.png')])

        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content
        assert 'b.png' in log_content

    def test_saving_one_image_to_two_lists_produces_copies_and_entries_in_both(self, tmp_path, root_dir):
        src = _make_png(str(tmp_path / 'src' / 'a.png'))

        save_image_to_list('cats', src, root_dir, metadata=[('Filename', 'filename', 'a.png')])
        save_image_to_list('favorites', src, root_dir, metadata=[('Filename', 'filename', 'a.png')])

        cats_dir = get_list_dir('cats', root_dir)
        favorites_dir = get_list_dir('favorites', root_dir)
        assert os.path.isfile(os.path.join(cats_dir, 'a.png'))
        assert os.path.isfile(os.path.join(favorites_dir, 'a.png'))
        assert os.path.isfile(os.path.join(cats_dir, 'log.html'))
        assert os.path.isfile(os.path.join(favorites_dir, 'log.html'))

    def test_default_output_copy_untouched(self, tmp_path, root_dir):
        """save_image_to_list only ever writes under root_dir -- the
        original file at its default-output location is a pure read."""
        src = _make_png(str(tmp_path / 'src' / 'a.png'))
        original_bytes = open(src, 'rb').read()

        save_image_to_list('cats', src, root_dir, metadata=[('Filename', 'filename', 'a.png')])

        assert open(src, 'rb').read() == original_bytes

    def test_missing_source_file_fails_without_raising(self, root_dir):
        success, message = save_image_to_list('cats', '/does/not/exist.png', root_dir)
        assert not success
        assert 'not found' in message

    def test_invalid_list_name_that_escapes_root_fails_gracefully(self, tmp_path, root_dir):
        outside = tmp_path / 'outside'
        outside.mkdir()
        os.symlink(str(outside), os.path.join(root_dir, 'escape'))
        src = _make_png(str(tmp_path / 'src' / 'a.png'))

        success, message = save_image_to_list('escape', src, root_dir)
        assert not success
        assert 'Invalid list name' in message


class TestMetadataRegistry:
    def test_record_then_get_returns_recorded_entry(self, tmp_path):
        path = str(tmp_path / 'a.png')
        task = {'positive': ['a cat'], 'negative': []}
        record_metadata(path, [('Prompt', 'prompt', 'a cat')], task)

        metadata, recorded_task = get_metadata(path)
        assert metadata == [('Prompt', 'prompt', 'a cat')]
        assert recorded_task is task

    def test_registry_miss_falls_back_to_embedded_png_metadata(self, tmp_path):
        path = _make_png(str(tmp_path / 'a.png'), parameters='{"prompt": "a dog"}')

        metadata, task = get_metadata(path)

        assert metadata is not None
        assert task is None
        assert any('prompt' in str(entry) for entry in metadata)

    def test_registry_miss_with_no_embedded_metadata_returns_none_without_raising(self, tmp_path):
        path = _make_png(str(tmp_path / 'a.png'))  # no pnginfo written

        metadata, task = get_metadata(path)

        assert metadata is None
        assert task is None

    def test_registry_miss_on_nonexistent_file_returns_none_without_raising(self, tmp_path):
        metadata, task = get_metadata(str(tmp_path / 'does_not_exist.png'))
        assert metadata is None
        assert task is None

    def test_save_image_to_list_falls_back_to_reduced_entry_when_metadata_missing(self, tmp_path, root_dir):
        """With no recorded metadata and no embedded metadata, saving must
        still succeed with a reduced log entry rather than raising."""
        src = _make_png(str(tmp_path / 'src' / 'a.png'))  # no pnginfo, no registry entry

        success, message = save_image_to_list('cats', src, root_dir)

        assert success, message
        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content
