"""Tests for FWDF-122's encoder-agnostic text-encoder infrastructure.

Covers:
- modules.text_encoder's TextEncoder/Tokenizer/PromptTemplate protocols,
  the TransformerTextEncoder generic loader, and its domain errors.
- The pre-existing pooled=None regression fix in
  modules.default_pipeline.clip_encode() (previously
  `TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'`
  the moment any encoder in the mix has no pooled projection).

modules.default_pipeline unconditionally imports a torchvision-only
inpainting architecture (LaMa, reached via modules.patch ->
modules.inpaint_worker -> modules.upscaler). torchvision is a docker-only
dependency (requirements_docker.txt), not installed in every dev/test
environment. That concern is not part of this ticket's scope, so rather than
changing production import behavior, the fixture below installs a minimal
torchvision stand-in when needed, letting the real, unmodified clip_encode()
/ clip_encode_single() / clone_cond() run end-to-end, and restores the
environment afterward.

(modules.default_pipeline used to also call refresh_everything(...) at
*module import time*, loading a real SDXL checkpoint and GPT2 prompt-expansion
model from disk -- FWDF-167 moved that call into an explicit
initialize_default_pipeline() that application entrypoints call at startup,
so importing the module here no longer requires stubbing that out.)
"""
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest import mock

import pytest
import safetensors.torch
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported (mirrors
# tests/test_new_ui_app.py).
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402
import modules.text_encoder as text_encoder  # noqa: E402

sys.argv = _original_argv


# ---------------------------------------------------------------------------
# modules.text_encoder: protocols, TransformerTextEncoder, domain errors
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def tokenize_with_weights(self, text, return_word_ids=False):
        return text


class TinyTransformer(nn.Module):
    """Minimal stand-in for a non-CLIP transformer text-encoder module,
    shaped like Qwen3-style encoders: no pooled projection head."""

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(4, 4, device=device, dtype=dtype)

    def encode_token_weights(self, tokens):
        cond = self.linear(torch.ones(1, 4, dtype=self.linear.weight.dtype, device=self.linear.weight.device))
        return cond, None


class TestTextEncoderProtocol:
    def test_clip_satisfies_text_encoder_without_modification(self):
        """Regression: ldm_patched.modules.sd.CLIP must structurally satisfy
        TextEncoder without any changes to sd.py."""
        from ldm_patched.modules.sd import CLIP

        clip_stub = CLIP(no_init=True)
        assert isinstance(clip_stub, text_encoder.TextEncoder)

    def test_transformer_text_encoder_satisfies_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, 'tiny.safetensors')
            safetensors.torch.save_file(TinyTransformer().state_dict(), weights_path)

            encoder = text_encoder.TransformerTextEncoder(
                TinyTransformer, {}, weights_path, FakeTokenizer()
            )
            assert isinstance(encoder, text_encoder.TextEncoder)

    def test_identity_prompt_template_satisfies_prompt_template(self):
        template = text_encoder.IdentityPromptTemplate()
        assert isinstance(template, text_encoder.PromptTemplate)
        assert template.apply('a photo of a cat') == 'a photo of a cat'


class TestTransformerTextEncoder:
    def test_encode_from_tokens_returns_none_pooled(self):
        """Encoders without a pooled projection head (Qwen3-style) must
        surface pooled=None rather than erroring or fabricating a value."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, 'tiny.safetensors')
            safetensors.torch.save_file(TinyTransformer().state_dict(), weights_path)

            encoder = text_encoder.TransformerTextEncoder(
                TinyTransformer, {}, weights_path, FakeTokenizer()
            )
            tokens = encoder.tokenize('a photo of a cat')
            cond, pooled = encoder.encode_from_tokens(tokens, return_pooled=True)

            assert isinstance(cond, torch.Tensor)
            assert pooled is None

    def test_prompt_template_applied_before_tokenization(self):
        class UppercaseTemplate:
            def apply(self, text):
                return text.upper()

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, 'tiny.safetensors')
            safetensors.torch.save_file(TinyTransformer().state_dict(), weights_path)

            encoder = text_encoder.TransformerTextEncoder(
                TinyTransformer, {}, weights_path, FakeTokenizer(),
                prompt_template=UppercaseTemplate(),
            )
            assert encoder.tokenize('a photo') == 'A PHOTO'

    def test_missing_encoder_file_raises_actionable_error(self):
        missing_path = os.path.join(tempfile.gettempdir(), 'fwdf-122-does-not-exist.safetensors')
        assert not os.path.isfile(missing_path)

        with pytest.raises(text_encoder.TextEncoderNotFoundError) as exc_info:
            text_encoder.TransformerTextEncoder(TinyTransformer, {}, missing_path, FakeTokenizer())

        message = str(exc_info.value)
        assert missing_path in message
        assert modules.config.path_text_encoders in message

    def test_state_dict_mismatch_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = os.path.join(tmp_dir, 'bad.safetensors')
            safetensors.torch.save_file({'unexpected.key': torch.zeros(2)}, bad_path)

            with pytest.raises(text_encoder.TextEncoderStateDictMismatchError) as exc_info:
                text_encoder.TransformerTextEncoder(TinyTransformer, {}, bad_path, FakeTokenizer())

            assert 'TinyTransformer' in str(exc_info.value)


class TestLoadTextEncoderStateDict:
    def test_missing_file_raises_not_found_error(self):
        missing_path = os.path.join(tempfile.gettempdir(), 'fwdf-122-also-missing.safetensors')
        assert not os.path.isfile(missing_path)

        with pytest.raises(text_encoder.TextEncoderNotFoundError):
            text_encoder.load_text_encoder_state_dict(missing_path)

    def test_existing_file_loads_state_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, 'tiny.safetensors')
            expected = TinyTransformer().state_dict()
            safetensors.torch.save_file(expected, weights_path)

            loaded = text_encoder.load_text_encoder_state_dict(weights_path)
            assert set(loaded.keys()) == set(expected.keys())


class TestPathTextEncodersConfig:
    def test_default_path_is_defined(self):
        assert modules.config.path_text_encoders
        assert os.path.isdir(modules.config.path_text_encoders)

    def test_overridable_via_config_dict(self):
        """path_text_encoders must go through the same
        get_dir_or_set_default() override mechanism as every other model
        directory (paths_checkpoints, path_vae, ...), so it is overridable
        via config.txt the same way."""
        original_value = modules.config.config_dict.get('path_text_encoders')
        try:
            with tempfile.TemporaryDirectory() as override_dir:
                modules.config.config_dict['path_text_encoders'] = override_dir
                resolved = modules.config.get_dir_or_set_default(
                    'path_text_encoders', '../models/text_encoders/'
                )
                assert resolved == override_dir
        finally:
            if original_value is None:
                modules.config.config_dict.pop('path_text_encoders', None)
            else:
                modules.config.config_dict['path_text_encoders'] = original_value


# ---------------------------------------------------------------------------
# modules.default_pipeline.clip_encode(): pooled=None regression fix
# ---------------------------------------------------------------------------


def _install_default_pipeline_test_doubles():
    """Install a torchvision stand-in (when torchvision isn't installed) so
    the real modules.default_pipeline can be imported. Returns a zero-arg
    callable that restores the prior state.

    Order matters: `transformers` must finish its own (real) import before
    a torchvision stand-in is registered, because transformers decides once,
    at import time, whether torchvision is available and caches that
    decision -- if a spec-less stand-in is already in sys.modules when that
    decision is made, the probe itself raises.
    """
    import transformers  # noqa: F401  (forces the real torchvision-unavailable check first)

    restore_actions = []

    torchvision_available = True
    try:
        import torchvision  # noqa: F401
    except ImportError:
        torchvision_available = False

    if not torchvision_available:
        stub_names = ('torchvision', 'torchvision.transforms', 'torchvision.transforms.functional')
        for name in stub_names:
            assert name not in sys.modules, f'unexpected pre-existing stub conflict for {name}'
        functional_stub = types.ModuleType('torchvision.transforms.functional')
        functional_stub.InterpolationMode = object
        functional_stub.rotate = lambda *a, **k: None
        transforms_stub = types.ModuleType('torchvision.transforms')
        transforms_stub.functional = functional_stub
        torchvision_stub = types.ModuleType('torchvision')
        torchvision_stub.transforms = transforms_stub

        sys.modules['torchvision'] = torchvision_stub
        sys.modules['torchvision.transforms'] = transforms_stub
        sys.modules['torchvision.transforms.functional'] = functional_stub
        restore_actions.append(lambda: [sys.modules.pop(n, None) for n in stub_names])

    def _restore():
        for action in reversed(restore_actions):
            action()

    return _restore


@pytest.fixture(scope='module')
def default_pipeline():
    """The real modules.default_pipeline, imported with just enough test
    doubles (see _install_default_pipeline_test_doubles) for the import to
    succeed. clip_encode(), clip_encode_single(), and clone_cond() are the
    genuine, unmodified production functions. The module-level
    refresh_everything(...) bootstrap this fixture used to have to survive
    no longer runs at import time (FWDF-167) -- initialize_default_pipeline()
    is never called here, so final_unet/final_clip/final_vae stay at their
    None defaults unless a test sets them directly."""
    restore = _install_default_pipeline_test_doubles()
    try:
        import modules.default_pipeline as pipeline
    except Exception:
        restore()
        raise
    yield pipeline
    restore()
    # The imported module was bootstrapped against the stubs above; evict it
    # so no later test module receives this stub-bound instance from the
    # sys.modules cache (or via the package attribute).
    sys.modules.pop('modules.default_pipeline', None)
    import modules as _modules_pkg
    if getattr(_modules_pkg, 'default_pipeline', None) is pipeline:
        delattr(_modules_pkg, 'default_pipeline')


class _RecordingEncoder:
    """Synthetic TextEncoder driving clip_encode() end-to-end: a fixed-shape
    cond tensor for every prompt, and a caller-supplied pooled tensor (or
    None) looked up by prompt text."""

    def __init__(self, pooled_by_text, cond_width=4):
        self.fcs_cond_cache = {}
        self._pooled_by_text = pooled_by_text
        self._cond_width = cond_width

    def tokenize(self, text, return_word_ids=False):
        return text

    def encode_from_tokens(self, tokens, return_pooled=False):
        cond = torch.ones(1, 1, self._cond_width)
        pooled = self._pooled_by_text.get(tokens)
        if return_pooled:
            return cond, pooled
        return cond


class TestClipEncodePooledNone:
    def test_clip_encode_tolerates_pooled_none(self, default_pipeline):
        """The concrete pre-existing bug this ticket fixes: pooled_acc used
        to start at 0 (an int), so `pooled_acc += pooled` raised
        TypeError the moment any prompt's pooled output was None."""
        default_pipeline.final_clip = _RecordingEncoder(
            pooled_by_text={'a': None, 'b': None, 'c': None}
        )

        result = default_pipeline.clip_encode(['a', 'b', 'c'], pool_top_k=2)

        assert result is not None
        cond, extra = result[0]
        assert isinstance(cond, torch.Tensor)
        assert cond.shape == (1, 3, 4)  # 3 prompts concatenated along dim=1
        assert extra['pooled_output'] is None

    def test_clip_encode_accumulates_pooled_across_pool_top_k(self, default_pipeline):
        """Regression: when pooled output is present (the CLIP case), the
        pool_top_k accumulation must still sum exactly as before the fix."""
        pooled_by_text = {
            'a': torch.tensor([[1.0, 1.0]]),
            'b': torch.tensor([[2.0, 2.0]]),
            'c': torch.tensor([[100.0, 100.0]]),  # beyond pool_top_k=2, must be excluded
        }
        default_pipeline.final_clip = _RecordingEncoder(pooled_by_text=pooled_by_text)

        result = default_pipeline.clip_encode(['a', 'b', 'c'], pool_top_k=2)

        cond, extra = result[0]
        assert torch.equal(extra['pooled_output'], torch.tensor([[3.0, 3.0]]))

    def test_clip_encode_mixed_pooled_and_none_does_not_raise(self, default_pipeline):
        """Mixed encoders (e.g. one prompt pooled, another None) must not
        crash -- the first non-None value seeds the accumulator."""
        pooled_by_text = {
            'a': None,
            'b': torch.tensor([[5.0, 5.0]]),
        }
        default_pipeline.final_clip = _RecordingEncoder(pooled_by_text=pooled_by_text)

        result = default_pipeline.clip_encode(['a', 'b'], pool_top_k=2)

        cond, extra = result[0]
        assert torch.equal(extra['pooled_output'], torch.tensor([[5.0, 5.0]]))

    def test_clip_encode_returns_none_for_empty_texts(self, default_pipeline):
        default_pipeline.final_clip = _RecordingEncoder(pooled_by_text={})
        assert default_pipeline.clip_encode([]) is None

    def test_clone_cond_tolerates_none_pooled(self, default_pipeline):
        conds = [[torch.ones(1, 1, 4), {'pooled_output': None}]]
        cloned = default_pipeline.clone_cond(conds)
        assert cloned[0][1]['pooled_output'] is None
        assert torch.equal(cloned[0][0], torch.ones(1, 1, 4))


# ---------------------------------------------------------------------------
# FWDF-167: initialize_default_pipeline() replaces the old import-time
# refresh_everything() side effect. It must run the (expensive) model load
# exactly once, hold the lock across the whole load so a concurrent caller
# never returns against a half-loaded pipeline, and mark itself done only
# after refresh_everything() succeeds so a failed load can be retried.
# ---------------------------------------------------------------------------
class TestInitializeDefaultPipeline:
    @pytest.fixture(autouse=True)
    def _reset_init_flag(self, default_pipeline):
        """initialize_default_pipeline()'s done-flag is module global state;
        save/restore it so these tests neither see nor leak a set flag."""
        saved = default_pipeline._pipeline_initialized
        default_pipeline._pipeline_initialized = False
        yield
        default_pipeline._pipeline_initialized = saved

    def test_repeated_calls_run_the_load_exactly_once(self, default_pipeline):
        with mock.patch.object(default_pipeline, 'refresh_everything') as refresh:
            default_pipeline.initialize_default_pipeline()
            default_pipeline.initialize_default_pipeline()
            default_pipeline.initialize_default_pipeline()

        refresh.assert_called_once()
        assert default_pipeline._pipeline_initialized is True

    def test_concurrent_caller_blocks_until_the_load_completes(self, default_pipeline):
        # The lock is held for the whole load, so a second caller must not
        # return until refresh_everything() has finished. Under the old
        # (flag-set-before-load, lock-released-before-load) code the second
        # caller would see the flag already set and return early -- against a
        # pipeline whose weights are not loaded yet.
        first_in_load = threading.Event()
        load_done = threading.Event()
        returned_early = threading.Event()

        def slow_refresh(**kwargs):
            first_in_load.set()
            time.sleep(0.3)  # widen the window a real race would exploit
            load_done.set()

        def second_caller():
            assert first_in_load.wait(timeout=5), 'first caller never entered the load'
            default_pipeline.initialize_default_pipeline()
            if not load_done.is_set():
                returned_early.set()

        with mock.patch.object(default_pipeline, 'refresh_everything', side_effect=slow_refresh):
            first = threading.Thread(target=default_pipeline.initialize_default_pipeline)
            second = threading.Thread(target=second_caller)
            first.start()
            second.start()
            first.join(timeout=5)
            second.join(timeout=5)

        assert not first.is_alive() and not second.is_alive(), 'a caller deadlocked'
        assert not returned_early.is_set(), 'second caller returned before the load finished'
        assert default_pipeline._pipeline_initialized is True

    def test_a_failed_load_leaves_the_pipeline_retryable(self, default_pipeline):
        # refresh_everything() raising must NOT mark the pipeline initialized,
        # so a later call retries the load rather than skipping it forever.
        attempts = []

        def fail_then_succeed(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('simulated model-load failure')

        with mock.patch.object(default_pipeline, 'refresh_everything', side_effect=fail_then_succeed):
            with pytest.raises(RuntimeError, match='simulated model-load failure'):
                default_pipeline.initialize_default_pipeline()
            assert default_pipeline._pipeline_initialized is False

            default_pipeline.initialize_default_pipeline()  # retry

        assert default_pipeline._pipeline_initialized is True
        assert len(attempts) == 2
