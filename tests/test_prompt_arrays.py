"""Tests for [[a, b, c]] prompt arrays (FWDF-212): arrays must expand on every
line of a multi-line prompt, with one combination shared across all lines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.sdxl_styles import apply_arrays, apply_arrays_to_lines


class TestApplyArraysToLines:
    def test_array_on_a_later_line_is_expanded(self):
        lines = ['(score_9), 1woman1man', '(1woman, brunette)', 'Eating [[pie, cake, salmon]] at a restaurant']

        result = apply_arrays_to_lines(lines, 1)

        assert result == ['(score_9), 1woman1man', '(1woman, brunette)', 'Eating  cake at a restaurant']

    def test_line_count_and_order_are_preserved(self):
        lines = ['first', '[[a,b]]', 'third', '[[c,d]]']

        result = apply_arrays_to_lines(lines, 0)

        assert len(result) == len(lines)
        assert result[0] == 'first'
        assert result[2] == 'third'

    def test_arrays_on_different_lines_form_one_product(self):
        lines = ['[[a,b]]', '[[x,y]]']

        combinations = {tuple(apply_arrays_to_lines(lines, i)) for i in range(4)}

        assert combinations == {('a', 'x'), ('b', 'x'), ('a', 'y'), ('b', 'y')}

    def test_matches_apply_arrays_for_a_single_line(self):
        text = 'a [[red,blue]] and [[cat,dog]]'

        for i in range(4):
            assert apply_arrays_to_lines([text], i) == [apply_arrays(text, i)]

    def test_lines_without_arrays_are_unchanged(self):
        lines = ['plain', 'also plain']

        assert apply_arrays_to_lines(lines, 3) == lines

    def test_empty_first_line_is_kept(self):
        assert apply_arrays_to_lines(['', '[[a,b]]'], 1) == ['', 'b']

    def test_empty_prompt_stays_a_single_empty_line(self):
        assert apply_arrays_to_lines([''], 0) == ['']
