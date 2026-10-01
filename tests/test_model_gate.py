"""Tests for the `requires_models` gate (FWDF-160): the decision logic in
tests/_model_gate.py and its wiring in tests/conftest.py."""
import os

import pytest

from tests._model_gate import (
    RUN_MODEL_TESTS_ENV,
    Krea2ModelLocations,
    find_krea2_checkpoints,
    skip_reason,
)

OPTED_IN = {RUN_MODEL_TESTS_ENV: '1'}


@pytest.fixture
def locations(tmp_path):
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    text_encoder = tmp_path / 'qwen3vl.safetensors'
    vae = tmp_path / 'vae.safetensors'
    return Krea2ModelLocations(checkpoint_dirs=[str(checkpoints)], text_encoder_path=str(text_encoder),
                               vae_path=str(vae))


def _populate(locations, checkpoint='krea2_turbo_bf16.safetensors'):
    for path in (os.path.join(locations.checkpoint_dirs[0], checkpoint), locations.text_encoder_path,
                 locations.vae_path):
        open(path, 'wb').close()


class TestFindKrea2Checkpoints:
    def test_finds_krea2_safetensors_case_insensitively_and_in_subfolders(self, tmp_path):
        (tmp_path / 'nested').mkdir()
        for name in ('krea2_raw_bf16.safetensors', 'nested/Krea2-Turbo.SAFETENSORS', 'sdxl_base.safetensors',
                     'krea2_notes.txt'):
            (tmp_path / name).write_bytes(b'')

        found = find_krea2_checkpoints([str(tmp_path)])

        assert sorted(found) == ['krea2_raw_bf16.safetensors', 'nested/Krea2-Turbo.SAFETENSORS']

    def test_missing_directory_yields_nothing(self, tmp_path):
        assert find_krea2_checkpoints([str(tmp_path / 'absent')]) == []


class TestSkipReason:
    def test_skips_without_the_opt_in_even_when_every_file_is_present(self, locations):
        _populate(locations)
        assert RUN_MODEL_TESTS_ENV in skip_reason({}, locations)

    @pytest.mark.parametrize('value', ['0', '', 'true', 'yes'])
    def test_only_the_exact_opt_in_value_enables_the_tests(self, locations, value):
        _populate(locations)
        assert skip_reason({RUN_MODEL_TESTS_ENV: value}, locations) is not None

    def test_runs_when_opted_in_and_every_file_is_present(self, locations):
        _populate(locations)
        assert skip_reason(OPTED_IN, locations) is None

    def test_names_the_missing_checkpoint(self, locations):
        reason = skip_reason(OPTED_IN, locations)
        assert 'krea2' in reason and 'checkpoint' in reason

    def test_names_the_missing_text_encoder(self, locations):
        _populate(locations)
        os.remove(locations.text_encoder_path)
        assert locations.text_encoder_path in skip_reason(OPTED_IN, locations)

    def test_names_the_missing_vae(self, locations):
        _populate(locations)
        os.remove(locations.vae_path)
        assert locations.vae_path in skip_reason(OPTED_IN, locations)


@pytest.mark.requires_models
def test_marked_test_is_skipped_unless_opted_in_with_files_present():
    """Wiring check for tests/conftest.py: reaching this body means the
    developer opted in and the files exist; otherwise the hook skips it (the
    only outcome in a model-free environment)."""
    assert os.environ.get(RUN_MODEL_TESTS_ENV) == '1'
