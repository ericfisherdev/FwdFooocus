import json
import os
import tempfile
import threading
import unittest


class TestUIPrefs(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.prefs_path = os.path.join(self.tmp_dir.name, 'ui_prefs.json')

    def test_fresh_instance_returns_ask_and_creates_no_file(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        prefs = UIPrefs(self.prefs_path)

        for field in PrefField:
            self.assertEqual(prefs.get(field), RememberDecision.ASK)
        self.assertFalse(os.path.exists(self.prefs_path))

    def test_set_get_round_trip_persists_across_instances(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        prefs = UIPrefs(self.prefs_path)
        prefs.set(PrefField.CHECKPOINT, RememberDecision.USE_METADATA)

        other_prefs = UIPrefs(self.prefs_path)
        self.assertEqual(other_prefs.get(PrefField.CHECKPOINT), RememberDecision.USE_METADATA)

    def test_corrupt_json_degrades_to_ask(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        with open(self.prefs_path, 'w', encoding='utf-8') as f:
            f.write('{not valid json')

        prefs = UIPrefs(self.prefs_path)
        for field in PrefField:
            self.assertEqual(prefs.get(field), RememberDecision.ASK)

    def test_non_dict_json_degrades_to_ask(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        with open(self.prefs_path, 'w', encoding='utf-8') as f:
            json.dump([1, 2, 3], f)

        prefs = UIPrefs(self.prefs_path)
        for field in PrefField:
            self.assertEqual(prefs.get(field), RememberDecision.ASK)

    def test_unknown_enum_value_degrades_only_that_field(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        with open(self.prefs_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata_load': {
                    'checkpoint': 'not_a_real_decision',
                    'sampler': 'use_metadata',
                }
            }, f)

        prefs = UIPrefs(self.prefs_path)
        self.assertEqual(prefs.get(PrefField.CHECKPOINT), RememberDecision.ASK)
        self.assertEqual(prefs.get(PrefField.SAMPLER), RememberDecision.USE_METADATA)
        self.assertEqual(prefs.get(PrefField.SCHEDULER), RememberDecision.ASK)
        self.assertEqual(prefs.get(PrefField.VAE), RememberDecision.ASK)

    def test_reset_all_restores_ask_and_persists(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        prefs = UIPrefs(self.prefs_path)
        prefs.set(PrefField.CHECKPOINT, RememberDecision.USE_METADATA)
        prefs.set(PrefField.VAE, RememberDecision.KEEP_CURRENT)

        prefs.reset_all()

        other_prefs = UIPrefs(self.prefs_path)
        for field in PrefField:
            self.assertEqual(other_prefs.get(field), RememberDecision.ASK)

    def test_label_decision_round_trip(self):
        from modules.ui_prefs import RememberDecision, decision_to_label, label_to_decision

        for decision in RememberDecision:
            label = decision_to_label(decision)
            self.assertEqual(label_to_decision(label), decision)

    def test_unknown_label_returns_ask(self):
        from modules.ui_prefs import RememberDecision, label_to_decision

        self.assertEqual(label_to_decision('not a real label'), RememberDecision.ASK)

    def test_concurrent_set_smoke(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        prefs = UIPrefs(self.prefs_path)
        decisions = [RememberDecision.ASK, RememberDecision.USE_METADATA, RememberDecision.KEEP_CURRENT]

        def worker(field, decision):
            prefs.set(field, decision)

        threads = [
            threading.Thread(target=worker, args=(field, decisions[i % len(decisions)]))
            for i, field in enumerate(PrefField)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(self.prefs_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        group = payload['metadata_load']
        valid_values = {d.value for d in RememberDecision}
        for field in PrefField:
            self.assertIn(field.value, group)
            self.assertIn(group[field.value], valid_values)

    def test_set_from_separate_instances_does_not_clobber(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        instance_a = UIPrefs(self.prefs_path)
        instance_b = UIPrefs(self.prefs_path)

        # Both instances lazily load the same (empty) on-disk state before
        # either writes.
        instance_a.get(PrefField.CHECKPOINT)
        instance_b.get(PrefField.SAMPLER)

        instance_a.set(PrefField.CHECKPOINT, RememberDecision.USE_METADATA)
        instance_b.set(PrefField.SAMPLER, RememberDecision.KEEP_CURRENT)

        verifying_instance = UIPrefs(self.prefs_path)
        self.assertEqual(verifying_instance.get(PrefField.CHECKPOINT), RememberDecision.USE_METADATA)
        self.assertEqual(verifying_instance.get(PrefField.SAMPLER), RememberDecision.KEEP_CURRENT)

    def test_invalid_utf8_degrades_to_ask(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        with open(self.prefs_path, 'wb') as f:
            f.write(b'\xff\xfe not valid utf-8')

        prefs = UIPrefs(self.prefs_path)
        for field in PrefField:
            self.assertEqual(prefs.get(field), RememberDecision.ASK)

    def test_save_preserves_unrelated_top_level_group(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        with open(self.prefs_path, 'w', encoding='utf-8') as f:
            json.dump({'other_group': {'some_key': 'some_value'}}, f)

        prefs = UIPrefs(self.prefs_path)
        prefs.set(PrefField.CHECKPOINT, RememberDecision.USE_METADATA)

        with open(self.prefs_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.assertEqual(payload['other_group'], {'some_key': 'some_value'})
        self.assertEqual(payload['metadata_load']['checkpoint'], 'use_metadata')

    def test_on_disk_layout_is_namespaced(self):
        from modules.ui_prefs import PrefField, RememberDecision, UIPrefs

        prefs = UIPrefs(self.prefs_path)
        prefs.set(PrefField.SAMPLER, RememberDecision.USE_METADATA)

        with open(self.prefs_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.assertIn('metadata_load', payload)
        self.assertEqual(payload['metadata_load']['sampler'], 'use_metadata')


if __name__ == '__main__':
    unittest.main()
