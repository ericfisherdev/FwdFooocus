"""Tests for modules.wildcard_ui (FWDF-185) and the fast wildcard-only
file-list refresh added to modules.config.

wildcard_ui itself is pure logic (no modules.config import), so most tests
below exercise it directly against a tmp_path fixture. The
update_wildcard_files test imports modules.config to verify the fast
refresh picks up newly written files without a full update_files() pass.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.wildcard_ui import (
    InvalidWildcardNameError,
    read_wildcard,
    scan_prompt,
    write_wildcard,
)


@pytest.fixture
def wildcard_dir(tmp_path):
    return str(tmp_path)


def _write(wildcard_dir, relative_path, content):
    full_path = os.path.join(wildcard_dir, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return relative_path


class TestScanPrompt:
    def test_finds_existing_and_missing_wildcards(self, wildcard_dir):
        filenames = [_write(wildcard_dir, 'animal.txt', 'cat\ndog\n')]

        result = scan_prompt('a photo of a __animal__ and a __vehicle__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.missing == ('vehicle',)
        assert result.nested == ()

    def test_does_not_raise_when_a_top_level_file_is_not_utf8(self, wildcard_dir):
        filenames = [_write(wildcard_dir, 'animal.txt', 'cat\ndog\n')]
        with open(os.path.join(wildcard_dir, 'animal.txt'), 'wb') as f:
            f.write(b'\xff\xfe not valid utf-8')

        result = scan_prompt('a __animal__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.nested == ()
        assert result.missing == ()

    def test_ignores_non_wildcard_underscores(self, wildcard_dir):
        result = scan_prompt('snake_case_variable and _not_a_wildcard', wildcard_dir, [])

        assert result.top_level == ()
        assert result.missing == ()
        assert result.nested == ()

    def test_nested_wildcard_existing(self, wildcard_dir):
        filenames = [
            _write(wildcard_dir, 'animal.txt', 'a __color__ cat\n'),
            _write(wildcard_dir, 'color.txt', 'red\nblue\n'),
        ]

        result = scan_prompt('a __animal__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.nested == ('color',)
        assert result.missing == ()

    def test_nested_wildcard_missing(self, wildcard_dir):
        filenames = [_write(wildcard_dir, 'animal.txt', 'a __color__ cat\n')]

        result = scan_prompt('a __animal__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.nested == ()
        assert result.missing == ('color',)

    def test_nesting_stops_at_one_level(self, wildcard_dir):
        filenames = [
            _write(wildcard_dir, 'animal.txt', 'a __color__ cat\n'),
            _write(wildcard_dir, 'color.txt', 'a __shade__ red\n'),
            _write(wildcard_dir, 'shade.txt', 'light\ndark\n'),
        ]

        result = scan_prompt('a __animal__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.nested == ('color',)
        assert 'shade' not in result.nested
        assert 'shade' not in result.missing

    def test_subfolder_match_counts_as_existing(self, wildcard_dir):
        filenames = [_write(wildcard_dir, 'sub/animal.txt', 'cat\ndog\n')]

        result = scan_prompt('a __animal__', wildcard_dir, filenames)

        assert result.top_level == ('animal',)
        assert result.missing == ()


class TestReadWildcard:
    def test_returns_empty_string_when_absent(self, wildcard_dir):
        assert read_wildcard('nonexistent', wildcard_dir, []) == ''

    def test_returns_empty_string_when_filenames_none_and_absent(self, wildcard_dir):
        assert read_wildcard('nonexistent', wildcard_dir, None) == ''

    def test_reads_existing_file_content(self, wildcard_dir):
        filenames = [_write(wildcard_dir, 'animal.txt', 'cat\ndog\n')]

        assert read_wildcard('animal', wildcard_dir, filenames) == 'cat\ndog\n'

    def test_traversal_name_returns_empty_when_filenames_is_none(self, tmp_path, wildcard_dir):
        secret_dir = tmp_path.parent
        secret_file = secret_dir / 'evil.txt'
        secret_file.write_text('leaked secret', encoding='utf-8')

        assert read_wildcard('../evil', wildcard_dir, None) == ''

    def test_non_utf8_file_returns_empty_instead_of_raising(self, wildcard_dir):
        target = os.path.join(wildcard_dir, 'binary.txt')
        with open(target, 'wb') as f:
            f.write(b'\xff\xfe\x00garbage')
        filenames = ['binary.txt']

        assert read_wildcard('binary', wildcard_dir, filenames) == ''

    def test_symlink_inside_wildcard_dir_pointing_outside_is_rejected(self, tmp_path, wildcard_dir):
        # A symlink physically inside wildcard_dir, but resolving outside
        # it, must not be followed to leak content from elsewhere on disk.
        # os.path.normpath alone would not catch this -- only realpath
        # resolution (see _resolve_confined) does.
        secret_file = tmp_path.parent / 'secret.txt'
        secret_file.write_text('leaked secret', encoding='utf-8')
        symlink_path = os.path.join(wildcard_dir, 'evil.txt')
        os.symlink(secret_file, symlink_path)

        assert read_wildcard('evil', wildcard_dir, ['evil.txt']) == ''


class TestWriteWildcard:
    def test_creates_file_and_returns_path(self, wildcard_dir):
        target = write_wildcard('animal', 'cat\ndog\n', wildcard_dir)

        assert target == os.path.join(wildcard_dir, 'animal.txt')
        with open(target, encoding='utf-8') as f:
            assert f.read() == 'cat\ndog\n'

    @pytest.mark.parametrize('bad_name', ['../evil', 'a/b', 'evil.txt', ''])
    def test_rejects_unsafe_names(self, wildcard_dir, bad_name):
        with pytest.raises(InvalidWildcardNameError):
            write_wildcard(bad_name, 'content', wildcard_dir)

    def test_symlink_inside_wildcard_dir_pointing_outside_is_rejected(self, tmp_path, wildcard_dir):
        # A symlink physically inside wildcard_dir (name 'evil.txt', so the
        # WILDCARD_NAME_PATTERN check on 'evil' passes) resolving outside
        # it must not be written through -- that would let an attacker
        # overwrite an arbitrary file elsewhere on disk.
        secret_file = tmp_path.parent / 'secret.txt'
        secret_file.write_text('original', encoding='utf-8')
        symlink_path = os.path.join(wildcard_dir, 'evil.txt')
        os.symlink(secret_file, symlink_path)

        with pytest.raises(InvalidWildcardNameError):
            write_wildcard('evil', 'malicious', wildcard_dir)

        assert secret_file.read_text(encoding='utf-8') == 'original'


class TestUpdateWildcardFiles:
    def test_picks_up_newly_written_file_without_full_refresh(self, monkeypatch, wildcard_dir):
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            import modules.config
        finally:
            sys.argv = original_argv

        monkeypatch.setattr(modules.config, 'path_wildcards', wildcard_dir)
        monkeypatch.setattr(modules.config, 'wildcard_filenames', [])

        write_wildcard('animal', 'cat\ndog\n', wildcard_dir)
        modules.config.update_wildcard_files()

        assert 'animal.txt' in modules.config.wildcard_filenames
