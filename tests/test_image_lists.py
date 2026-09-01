"""Tests for modules.image_lists (FWDF-188).

image_lists is UI-agnostic (no gradio import) and takes its output root as
a parameter rather than reading modules.config internally, so these tests
exercise it directly against tmp_path fixtures -- no heavy dependencies
(torch etc.) are involved, so no import-guarding is needed here, matching
tests/test_wildcard_ui.py's approach for the other pure-logic module.
"""
import os
import shutil
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
        resolve_checked_gallery_paths,
        sanitize_list_name,
        save_image_to_list,
    )
    _import_error = None
except ImportError as e:  # pragma: no cover - exercised only without gradio installed
    private_logger = None
    get_list_dir = get_metadata = list_exists = list_image_lists = None
    record_metadata = sanitize_list_name = save_image_to_list = None
    resolve_checked_gallery_paths = None
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


@pytest.fixture
def source_root(tmp_path):
    """The directory save_image_to_list confines source_image_path to --
    stands in for modules.config.path_outputs in production."""
    src_dir = tmp_path / 'outputs'
    src_dir.mkdir()
    return str(src_dir)


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

    def test_strips_control_characters(self):
        assert sanitize_list_name('my\nlist\tname\x00') == 'my_list_name_'


class TestListDirRoundTrip:
    def test_create_list_then_appears_in_list_image_lists(self, tmp_path, root_dir, source_root):
        src = _make_png(str(Path(source_root) / 'a.png'))
        success, _ = save_image_to_list('My List', src, root_dir, source_root_dirs=source_root)

        assert success
        assert 'My List' in list_image_lists(root_dir)
        assert list_exists('My List', root_dir)

    def test_list_exists_false_for_missing_list(self, root_dir):
        assert not list_exists('nope', root_dir)

    def test_get_list_dir_rejects_dotdot_traversal(self, root_dir):
        # sanitize_list_name replaces '/' with '_', so a traversal-looking
        # name can't actually escape root_dir -- assert the resolved dir
        # stays inside it rather than expecting an outright rejection.
        resolved = get_list_dir('../../etc', root_dir)
        assert resolved is not None
        assert os.path.commonpath([os.path.realpath(root_dir), resolved]) == os.path.realpath(root_dir)

    def test_get_list_dir_rejects_absolute_path_name(self, root_dir):
        resolved = get_list_dir('/etc/passwd', root_dir)
        assert resolved is not None
        assert os.path.commonpath([os.path.realpath(root_dir), resolved]) == os.path.realpath(root_dir)

    def test_get_list_dir_rejects_name_containing_separator(self, root_dir):
        resolved = get_list_dir('a/b/../../c', root_dir)
        assert resolved is not None
        assert os.path.commonpath([os.path.realpath(root_dir), resolved]) == os.path.realpath(root_dir)

    def test_get_list_dir_rejects_symlink_escape(self, tmp_path, root_dir):
        outside = tmp_path / 'outside'
        outside.mkdir()
        symlink_path = os.path.join(root_dir, 'escape')
        os.symlink(str(outside), symlink_path)

        assert get_list_dir('escape', root_dir) is None


class TestSaveImageToList:
    def test_copies_file_byte_identical_and_writes_log_entry(self, tmp_path, root_dir, source_root):
        src = _make_png(str(Path(source_root) / 'a.png'), parameters='{"prompt": "a cat"}')

        success, message = save_image_to_list(
            'cats', src, root_dir, source_root_dirs=source_root, metadata=[('Prompt', 'prompt', 'a cat')])

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

    def test_second_save_appends_newest_first(self, tmp_path, root_dir, source_root):
        src_a = _make_png(str(Path(source_root) / 'a.png'))
        src_b = _make_png(str(Path(source_root) / 'b.png'))

        save_image_to_list('cats', src_a, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])
        save_image_to_list('cats', src_b, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'b.png')])

        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()

        assert 'a.png' in log_content and 'b.png' in log_content
        # newest first: b's entry appears before a's
        assert log_content.index('b.png') < log_content.index('a.png')
        assert os.path.isfile(os.path.join(list_dir, 'a.png'))
        assert os.path.isfile(os.path.join(list_dir, 'b.png'))

    def test_saving_same_filename_twice_overwrites_without_duplicating_log_entry(self, tmp_path, root_dir, source_root):
        src = _make_png(str(Path(source_root) / 'a.png'))

        save_image_to_list('cats', src, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])
        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])

        assert success
        assert 'Updated' in message
        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert log_content.count('a.png') == 1  # one <div id="a_png"> container, not two

    def test_append_survives_log_cache_being_cleared_cold_start(self, tmp_path, root_dir, source_root):
        """Simulates a server restart: the module-level log_cache in
        private_logger is empty, but log.html already exists on disk from a
        prior save -- appending must recover and preserve the existing
        entry, not clobber it."""
        src_a = _make_png(str(Path(source_root) / 'a.png'))
        src_b = _make_png(str(Path(source_root) / 'b.png'))

        save_image_to_list('cats', src_a, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])

        # Simulate restart: process-local cache is gone, but the file is not.
        private_logger.log_cache.clear()

        save_image_to_list('cats', src_b, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'b.png')])

        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content
        assert 'b.png' in log_content

    def test_saving_one_image_to_two_lists_produces_copies_and_entries_in_both(self, tmp_path, root_dir, source_root):
        src = _make_png(str(Path(source_root) / 'a.png'))

        save_image_to_list('cats', src, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])
        save_image_to_list('favorites', src, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])

        cats_dir = get_list_dir('cats', root_dir)
        favorites_dir = get_list_dir('favorites', root_dir)
        assert os.path.isfile(os.path.join(cats_dir, 'a.png'))
        assert os.path.isfile(os.path.join(favorites_dir, 'a.png'))
        assert os.path.isfile(os.path.join(cats_dir, 'log.html'))
        assert os.path.isfile(os.path.join(favorites_dir, 'log.html'))

    def test_default_output_copy_untouched(self, tmp_path, root_dir, source_root):
        """save_image_to_list only ever writes under root_dir -- the
        original file at its default-output location is a pure read."""
        src = _make_png(str(Path(source_root) / 'a.png'))
        original_bytes = open(src, 'rb').read()

        save_image_to_list('cats', src, root_dir, source_root_dirs=source_root, metadata=[('Filename', 'filename', 'a.png')])

        assert open(src, 'rb').read() == original_bytes

    def test_missing_source_file_fails_without_raising(self, root_dir, source_root):
        missing = os.path.join(source_root, 'does_not_exist.png')
        success, message = save_image_to_list('cats', missing, root_dir, source_root_dirs=source_root)
        assert not success
        assert 'not found' in message

    def test_invalid_list_name_that_escapes_root_fails_gracefully(self, tmp_path, root_dir, source_root):
        outside = tmp_path / 'outside'
        outside.mkdir()
        os.symlink(str(outside), os.path.join(root_dir, 'escape'))
        src = _make_png(str(Path(source_root) / 'a.png'))

        success, message = save_image_to_list('escape', src, root_dir, source_root_dirs=source_root)
        assert not success
        assert 'Invalid list name' in message

    def test_source_path_outside_source_root_is_rejected(self, tmp_path, root_dir, source_root):
        """A source_image_path pointing outside source_root_dir (the
        modules.config.path_outputs stand-in) must be rejected before any
        filesystem operation touches it -- this is the confinement CodeQL's
        py/path-injection check requires at every sink, not just for the
        list name."""
        outside = tmp_path / 'elsewhere'
        outside.mkdir()
        src = _make_png(str(outside / 'a.png'))

        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root)

        assert not success
        assert 'outside allowed directory' in message
        # nothing should have been created under root_dir
        assert list_image_lists(root_dir) == []

    def test_source_path_with_dotdot_traversal_is_rejected(self, tmp_path, root_dir, source_root):
        outside = tmp_path / 'elsewhere'
        outside.mkdir()
        _make_png(str(outside / 'secret.png'))
        traversal_path = os.path.join(source_root, '..', 'elsewhere', 'secret.png')

        success, message = save_image_to_list('cats', traversal_path, root_dir, source_root_dirs=source_root)

        assert not success
        assert 'outside allowed directory' in message

    def test_source_path_via_symlink_escape_is_rejected(self, tmp_path, root_dir, source_root):
        outside = tmp_path / 'elsewhere'
        outside.mkdir()
        real_secret = _make_png(str(outside / 'secret.png'))
        symlinked_path = os.path.join(source_root, 'escape.png')
        os.symlink(real_secret, symlinked_path)

        success, message = save_image_to_list('cats', symlinked_path, root_dir, source_root_dirs=source_root)

        assert not success
        assert 'outside allowed directory' in message

    def test_accepts_source_under_second_allowed_root(self, tmp_path, root_dir, source_root):
        """source_root_dirs takes a list -- webui.py passes both
        path_outputs and temp_path since a gallery path can point into
        either (see private_logger.log's persist_image branch). A source
        under the *second* root, not the first, must still be accepted."""
        temp_root = tmp_path / 'temp'
        temp_root.mkdir()
        src = _make_png(str(temp_root / 'a.png'))

        success, message = save_image_to_list(
            'cats', src, root_dir, source_root_dirs=[source_root, str(temp_root)])

        assert success, message
        list_dir = get_list_dir('cats', root_dir)
        assert os.path.isfile(os.path.join(list_dir, 'a.png'))

    def test_concurrent_saves_of_different_files_both_persist_log_entries(self, tmp_path, root_dir, source_root):
        """save_to_list handlers in webui.py run through Gradio's serialized
        queue (FWDF-194), but this module is UI-agnostic and the new_ui
        FastAPI path can invoke it from concurrent request threads -- the
        module-level _save_lock must serialize the copy-plus-log section so
        two concurrent saves into the same list cannot lose either entry to
        the other's log.html rewrite."""
        import threading

        src_a = _make_png(str(Path(source_root) / 'a.png'))
        src_b = _make_png(str(Path(source_root) / 'b.png'))
        results = {}

        def worker(key, src, filename):
            results[key] = save_image_to_list(
                'cats', src, root_dir, source_root_dirs=source_root,
                metadata=[('Filename', 'filename', filename)])

        t_a = threading.Thread(target=worker, args=('a', src_a, 'a.png'))
        t_b = threading.Thread(target=worker, args=('b', src_b, 'b.png'))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        assert results['a'][0], results['a'][1]
        assert results['b'][0], results['b'][1]
        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content
        assert 'b.png' in log_content

    def test_failed_copy_leaves_no_partial_file_and_retry_still_logs(self, tmp_path, root_dir, source_root, monkeypatch):
        """A copy that fails partway (disk full, crash) must not leave a
        partial file at the final destination -- an in-place O_TRUNC write
        would let a retry see already_saved=True from the partial file and
        permanently skip the log entry. The temp-name-plus-os.replace
        write means the destination only ever exists fully written."""
        import modules.image_lists as image_lists_module

        src = _make_png(str(Path(source_root) / 'a.png'))
        original_copyfileobj = shutil.copyfileobj
        call_count = {'n': 0}

        def flaky_copyfileobj(fsrc, fdst):
            call_count['n'] += 1
            if call_count['n'] == 1:
                fdst.write(b'partial-bytes')
                raise OSError('simulated disk full')
            return original_copyfileobj(fsrc, fdst)

        monkeypatch.setattr(image_lists_module.shutil, 'copyfileobj', flaky_copyfileobj)

        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root)
        assert not success

        list_dir = get_list_dir('cats', root_dir)
        dest = os.path.join(list_dir, 'a.png')
        assert not os.path.exists(dest)
        assert not os.path.exists(dest + '.tmp')

        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root)
        assert success, message
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content

    def test_save_succeeds_without_fchmod_or_fd_utime_support(self, tmp_path, root_dir, source_root, monkeypatch):
        """os.fchmod and fd-based os.utime don't exist / aren't supported
        on Windows under the project's pinned Python 3.10 (a documented
        first-class target via run.bat) -- simulate that absence and
        assert the save still succeeds via the by-path os.utime fallback,
        rather than raising AttributeError partway through the copy."""
        import modules.image_lists as image_lists_module

        monkeypatch.delattr(image_lists_module.os, 'fchmod', raising=False)
        monkeypatch.setattr(image_lists_module.os, 'supports_fd', frozenset(), raising=False)

        src = _make_png(str(Path(source_root) / 'a.png'))
        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root)

        assert success, message
        list_dir = get_list_dir('cats', root_dir)
        assert os.path.isfile(os.path.join(list_dir, 'a.png'))
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content


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

    def test_save_image_to_list_falls_back_to_reduced_entry_when_metadata_missing(self, tmp_path, root_dir, source_root):
        """With no recorded metadata and no embedded metadata, saving must
        still succeed with a reduced log entry rather than raising."""
        src = _make_png(str(Path(source_root) / 'a.png'))  # no pnginfo, no registry entry

        success, message = save_image_to_list('cats', src, root_dir, source_root_dirs=source_root)

        assert success, message
        list_dir = get_list_dir('cats', root_dir)
        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert 'a.png' in log_content


class TestResolveCheckedGalleryPaths:
    """FWDF-191: resolve_checked_gallery_paths() decodes the JSON index
    array javascript/gallery_checkboxes.js writes into #gallery_checked_data
    into the checked subset of gallery_paths. It must never raise -- the
    JSON payload comes from the browser, and gallery_paths is task.results
    verbatim (webui.py's 'finish' yield), which can carry non-str numpy
    entries such as the build_image_wall collage tile."""

    def test_multi_pick_preserves_gallery_order_regardless_of_json_order(self):
        gallery_paths = ['/out/a.png', '/out/b.png', '/out/c.png', '/out/d.png']
        # JSON payload arrives out of order -- result must still follow
        # gallery order, not JSON order.
        resolved = resolve_checked_gallery_paths(gallery_paths, '[3,1]')
        assert resolved == ['/out/b.png', '/out/d.png']

    def test_empty_checked_array_returns_empty_list(self):
        assert resolve_checked_gallery_paths(['/out/a.png'], '[]') == []

    def test_invalid_json_returns_empty_list_without_raising(self):
        assert resolve_checked_gallery_paths(['/out/a.png'], 'not json') == []

    def test_non_list_json_returns_empty_list_without_raising(self):
        assert resolve_checked_gallery_paths(['/out/a.png'], '{"0": true}') == []

    def test_bool_entries_are_ignored_not_treated_as_0_or_1(self):
        gallery_paths = ['/out/a.png', '/out/b.png']
        assert resolve_checked_gallery_paths(gallery_paths, '[true, false]') == []

    def test_float_entries_are_ignored(self):
        gallery_paths = ['/out/a.png', '/out/b.png']
        assert resolve_checked_gallery_paths(gallery_paths, '[0.5, 1.0]') == []

    def test_out_of_range_indices_are_ignored(self):
        gallery_paths = ['/out/a.png', '/out/b.png']
        assert resolve_checked_gallery_paths(gallery_paths, '[-1, 2, 5]') == []

    def test_duplicate_indices_collapse_to_one_entry(self):
        gallery_paths = ['/out/a.png', '/out/b.png']
        assert resolve_checked_gallery_paths(gallery_paths, '[0,0,0]') == ['/out/a.png']

    def test_non_str_gallery_paths_entry_is_skipped(self):
        """A checked index pointing at a non-str gallery_paths entry (the
        build_image_wall collage tile, an enhance/debug intermediate --
        represented here as a numpy-like stand-in) is silently skipped
        rather than raising."""
        import numpy as np

        collage_tile = np.zeros((2, 2, 3), dtype=np.uint8)
        gallery_paths = ['/out/a.png', collage_tile, '/out/b.png']

        resolved = resolve_checked_gallery_paths(gallery_paths, '[0,1,2]')

        assert resolved == ['/out/a.png', '/out/b.png']

    def test_non_list_gallery_paths_returns_empty_list_without_raising(self):
        assert resolve_checked_gallery_paths('not a list', '[0]') == []


class TestMultiSaveAggregation:
    """FWDF-191: the save_to_list_confirm handler in webui.py loops
    save_image_to_list over resolve_checked_gallery_paths' result. These
    tests exercise that same loop pattern directly against
    modules.image_lists to verify N checked images produce exactly N new
    files and N new log.html entries, each independently guarded."""

    def test_checking_two_of_four_saves_exactly_those_two(self, tmp_path, root_dir, source_root):
        paths = [
            _make_png(str(Path(source_root) / f'{name}.png'))
            for name in ('a', 'b', 'c', 'd')
        ]
        checked = resolve_checked_gallery_paths(paths, '[1,3]')
        assert checked == [paths[1], paths[3]]

        results = [
            save_image_to_list('picks', path, root_dir, source_root_dirs=source_root,
                                metadata=[('Filename', 'filename', os.path.basename(path))])
            for path in checked
        ]

        assert all(success for success, _ in results), results

        list_dir = get_list_dir('picks', root_dir)
        saved_files = sorted(f for f in os.listdir(list_dir) if f.endswith('.png'))
        assert saved_files == ['b.png', 'd.png']

        for path in checked:
            saved_bytes = open(os.path.join(list_dir, os.path.basename(path)), 'rb').read()
            assert saved_bytes == open(path, 'rb').read()

        log_content = open(os.path.join(list_dir, 'log.html'), encoding='utf-8').read()
        assert log_content.count('<div id=') == 2
        assert 'b.png' in log_content and 'd.png' in log_content
        assert 'a.png' not in log_content and 'c.png' not in log_content
