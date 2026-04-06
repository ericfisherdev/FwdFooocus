import os
import tempfile
import unittest


class TestSessionState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_session.db')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_module(self):
        """Get a fresh session_state module pointing at the test DB."""
        from modules import session_state
        session_state._db_path = self.db_path
        session_state._connection = None
        return session_state

    def test_load_state_returns_none_when_empty(self):
        mod = self._get_module()
        result = mod.load_state('pony')
        self.assertIsNone(result)

    def test_save_and_load_state(self):
        mod = self._get_module()
        state = {
            'prompt': 'a cat',
            'negative_prompt': 'bad quality',
            'base_model_name': 'ponyDiffusion.safetensors',
            'loras': [{'enabled': True, 'filename': 'detail.safetensors', 'weight': 0.8}],
            'steps': 20,
            'cfg_scale': 7.0,
        }
        mod.save_state('pony', state)
        loaded = mod.load_state('pony')
        self.assertEqual(loaded['prompt'], 'a cat')
        self.assertEqual(loaded['steps'], 20)
        self.assertEqual(len(loaded['loras']), 1)

    def test_save_overwrites_previous_state(self):
        mod = self._get_module()
        mod.save_state('pony', {'prompt': 'first'})
        mod.save_state('pony', {'prompt': 'second'})
        loaded = mod.load_state('pony')
        self.assertEqual(loaded['prompt'], 'second')

    def test_separate_base_models(self):
        mod = self._get_module()
        mod.save_state('pony', {'prompt': 'pony prompt'})
        mod.save_state('sdxl', {'prompt': 'sdxl prompt'})
        self.assertEqual(mod.load_state('pony')['prompt'], 'pony prompt')
        self.assertEqual(mod.load_state('sdxl')['prompt'], 'sdxl prompt')

    def test_seed_negative_one_is_excluded(self):
        mod = self._get_module()
        state = {'prompt': 'test', 'seed': -1}
        mod.save_state('pony', state)
        loaded = mod.load_state('pony')
        self.assertNotIn('seed', loaded)

    def test_seed_explicit_value_is_stored(self):
        mod = self._get_module()
        state = {'prompt': 'test', 'seed': 42}
        mod.save_state('pony', state)
        loaded = mod.load_state('pony')
        self.assertEqual(loaded['seed'], 42)

    def test_database_created_on_first_access(self):
        mod = self._get_module()
        self.assertFalse(os.path.exists(self.db_path))
        mod.save_state('pony', {'prompt': 'test'})
        self.assertTrue(os.path.exists(self.db_path))


if __name__ == '__main__':
    unittest.main()
