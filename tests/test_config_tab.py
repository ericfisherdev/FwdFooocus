"""
Tests for the FWDF-183 Config tab wired into webui.py.

webui.py builds the Gradio app at module import time and blocks on
shared.gradio_root.launch()/block_thread() at the bottom of the file, so it
cannot be imported directly in a unit test (mirrors the constraint noted in
tests/test_meta_confirm.py:222 about referencing webui.py by line number
rather than importing it). Instead these tests parse webui.py's AST and
assert on the specific structures the Config tab depends on:

- the Config tab is a sibling of the Advanced tab inside advanced_column
- the per-field radios persist via `.input` (user interaction only, not
  `.change`, which also fires on the programmatic gr.update() renders from
  page load / reset / modal Apply) wired through a handler *factory* (a
  function called with `field` as an argument inside the loop), not a
  lambda that closes over the loop variable directly (the late-binding trap
  called out in the ticket)
- the FWDF-182 modal's merged Apply handler outputs are extended with the
  Config tab's radios, and its handler updates them from default_prefs in
  both the "no pending metadata" and normal-completion branches
- the Config tab refreshes from default_prefs on page load
"""

import ast
import os
import unittest

WEBUI_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webui.py')


class TestConfigTabStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WEBUI_PATH, 'r', encoding='utf-8') as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source, filename=WEBUI_PATH)

    def _find_with_gr_tab_config(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if not isinstance(call, ast.Call):
                    continue
                if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'Tab'):
                    continue
                for kw in call.keywords:
                    if kw.arg == 'label' and isinstance(kw.value, ast.Constant) and kw.value.value == 'Config':
                        return node
        return None

    def test_config_tab_exists_as_sibling_tab(self):
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab, "Expected a `with gr.Tab(label='Config'):` block in webui.py")

    def test_config_tab_builds_one_radio_per_pref_field(self):
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        body_src = ast.unparse(config_tab)
        self.assertIn('config_pref_radio_list', body_src)
        self.assertIn('meta_confirm_field_order', body_src)
        self.assertIn('gr.Radio', body_src)
        self.assertIn('modules.ui_prefs.DECISION_LABELS', body_src)

    def test_input_handler_uses_a_factory_not_a_loop_closure(self):
        """
        Guard against the late-binding trap: the .input handler must come
        from calling a factory function with `field` as an argument at each
        loop iteration, not from a lambda that reads the loop variable
        `field` directly from the enclosing scope (which would make every
        handler act on whatever `field` happens to be after the loop ends).
        """
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        body_src = ast.unparse(config_tab)

        self.assertIn('def make_config_pref_change_handler(field)', body_src)
        self.assertIn('make_config_pref_change_handler(field)', body_src)
        self.assertNotRegex(
            body_src,
            r'radio\.(change|input)\(\s*lambda',
            "radio.input should be wired via make_config_pref_change_handler(field), "
            "not an inline lambda over the loop variable",
        )

    def test_persistence_is_wired_to_input_not_change(self):
        """
        .change fires on programmatic gr.update() renders too (page-load
        refresh, reset button, modal Apply all render these radios), so
        wiring persistence to .change would re-save a stale value on every
        such render. .input only fires for genuine user interaction.
        """
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        body_src = ast.unparse(config_tab)

        self.assertIn('radio.input(make_config_pref_change_handler(field)', body_src)
        self.assertNotIn('radio.change(', body_src)

    def test_reset_button_restores_ask_for_every_field(self):
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        body_src = ast.unparse(config_tab)
        self.assertIn('config_pref_reset_button', body_src)
        self.assertIn('default_prefs.reset_all()', body_src)
        self.assertIn('RememberDecision.ASK', body_src)

    def test_config_tab_refreshes_on_page_load(self):
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        body_src = ast.unparse(config_tab)
        self.assertIn('shared.gradio_root.load(config_pref_refresh', body_src)
        self.assertIn('outputs=config_pref_radio_list', body_src)

    def _find_function(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def test_apply_click_handler_returns_config_pref_updates_on_both_branches(self):
        handler = self._find_function('meta_confirm_apply_click')
        self.assertIsNotNone(handler, 'Expected meta_confirm_apply_click to still be defined')
        src = ast.unparse(handler)
        self.assertEqual(
            src.count('config_pref_refresh()'), 2,
            'Expected config_pref_refresh() appended to result on both the '
            'early-return (no pending_metadata) and normal-completion branches',
        )
        self.assertNotIn(
            '_config_pref_updates', src,
            'meta_confirm_apply_click should reuse config_pref_refresh() rather than '
            'a separate duplicate helper',
        )

    def test_config_pref_refresh_reads_current_default_prefs(self):
        config_tab = self._find_with_gr_tab_config()
        self.assertIsNotNone(config_tab)
        for node in ast.walk(config_tab):
            if isinstance(node, ast.FunctionDef) and node.name == 'config_pref_refresh':
                src = ast.unparse(node)
                self.assertIn('modules.ui_prefs.default_prefs.get(field)', src)
                self.assertIn('meta_confirm_field_order', src)
                return
        self.fail('Expected a config_pref_refresh() function inside the Config tab block')

    def test_meta_confirm_apply_outputs_include_config_pref_radios(self):
        found = False
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'click'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'meta_confirm_apply'):
                continue
            for kw in node.keywords:
                if kw.arg == 'outputs':
                    outputs_src = ast.unparse(kw.value)
                    self.assertIn('config_pref_radio_list', outputs_src)
                    found = True
        self.assertTrue(found, 'Expected to find meta_confirm_apply.click(...) with an outputs= kwarg')


if __name__ == '__main__':
    unittest.main()
