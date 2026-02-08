"""Tests for _match_to_choice and evaluate_answer in runner.

Loads only the needed functions from runner.py without triggering
heavy dependency imports (torch, etc.).
"""

import sys
import os
import re
import types
import importlib.util

# Load runner.py as a standalone module, bypassing runner/__init__.py
_runner_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runner", "runner.py",
)
_spec = importlib.util.spec_from_file_location("_runner_standalone", _runner_path)
_mod = types.ModuleType(_spec.name)
_mod.__file__ = _runner_path

# Execute only the top-level functions we need (stop before class Runner)
with open(_runner_path, "r") as _f:
    _source = _f.read()

# Extract the portion from the start up to class Runner (not including it)
_class_pos = _source.find("\nclass Runner")
if _class_pos == -1:
    _class_pos = len(_source)
_header = (
    "import re\n"
    + _source[_source.find("def make_options"):_class_pos]
)
exec(compile(_header, _runner_path, "exec"), _mod.__dict__)

make_options = _mod.make_options
_match_to_choice = _mod._match_to_choice
evaluate_answer = _mod.evaluate_answer


# ============================================================
# Helpers
# ============================================================

CHOICES_IMAGE = ["the first image", "the second image", "the third image", "the fourth image"]
CHOICES_SHORT = ["yes", "no"]
CHOICES_NUMBERS = ["1", "2", "3", "4"]
CHOICES_LONG = ["a cat sitting on a mat", "a dog running in the park", "a bird flying in the sky"]

PREFIXES_LETTER_4 = ["A", "B", "C", "D"]
PREFIXES_LETTER_2 = ["A", "B"]
PREFIXES_LETTER_3 = ["A", "B", "C"]
PREFIXES_NUMERIC_4 = ["1", "2", "3", "4"]


def assert_match(predicted, choices, prefixes, expected, msg=""):
    """Assert _match_to_choice returns expected choice text."""
    result = _match_to_choice(predicted, choices, prefixes)
    assert result == expected, (
        f"FAIL: predicted={predicted!r} → got {result!r}, expected {expected!r}"
        + (f" ({msg})" if msg else "")
    )


def assert_eval(predicted, ground_truth, choices, expected_correct, option_format='letter', msg=""):
    """Assert evaluate_answer returns expected boolean."""
    result = evaluate_answer(predicted, ground_truth, "multi-choice", choices, option_format)
    assert result == expected_correct, (
        f"FAIL: predicted={predicted!r}, gt={ground_truth!r} → got {result}, expected {expected_correct}"
        + (f" ({msg})" if msg else "")
    )


# ============================================================
# Tests for _match_to_choice
# ============================================================

def test_pure_letter():
    """Single letter prefix — most common case."""
    assert_match("A", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("B", CHOICES_IMAGE, PREFIXES_LETTER_4, "the second image")
    assert_match("C", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("D", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image")


def test_lowercase_letter():
    """Lowercase single letter."""
    assert_match("a", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("c", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("d", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image")


def test_letter_with_parens():
    """Letter wrapped in parentheses."""
    assert_match("(A)", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("(C)", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("(d)", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image")


def test_letter_with_dot():
    """Letter followed by dot."""
    assert_match("A.", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("C.", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_letter_with_closing_paren():
    """Letter followed by closing paren only."""
    assert_match("A)", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("C)", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_letter_with_comma():
    """Letter followed by comma and explanation."""
    assert_match("C, because it looks real", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_letter_with_colon():
    """Letter followed by colon."""
    assert_match("B: the second image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the second image")


def test_full_option_format():
    """Full option format like '(C) the third image'."""
    assert_match("(A) the first image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("(C) the third image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("(D) the fourth image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image")


def test_letter_space_content():
    """Letter + space + choice text."""
    assert_match("A the first image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("C the third image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_letter_dot_content():
    """Letter + dot + space + choice text."""
    assert_match("A. the first image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("C. the third image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_pure_choice_text():
    """Exact choice text without prefix."""
    assert_match("the first image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")
    assert_match("the second image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the second image")
    assert_match("the third image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("the fourth image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image")


def test_choice_text_substring():
    """Choice text as part of a longer response."""
    assert_match("I think the third image is correct", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_partial_choice_in_predicted():
    """Predicted is a substring of a choice text."""
    assert_match("first image", CHOICES_IMAGE, PREFIXES_LETTER_4, "the first image")


# ============================================================
# THE ORIGINAL BUG — these must all pass
# ============================================================

def test_bug_c_not_matched_to_second():
    """The original bug: 'C' was matched to 'the second image' via 'c' in 'second'."""
    assert_match("C", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image",
                 msg="original bug: C must match third, not second")
    assert_match("c", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image",
                 msg="original bug: c must match third, not second")


def test_bug_d_not_matched_to_second():
    """Same bug: 'D' was matched to 'the second image' via 'd' in 'second'."""
    assert_match("D", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image",
                 msg="original bug: D must match fourth, not second")
    assert_match("d", CHOICES_IMAGE, PREFIXES_LETTER_4, "the fourth image",
                 msg="original bug: d must match fourth, not second")


# ============================================================
# Numeric format
# ============================================================

def test_numeric_prefix():
    """Numeric prefix format."""
    prefixes = ["1", "2", "3", "4"]
    assert_match("1", CHOICES_IMAGE, prefixes, "the first image")
    assert_match("3", CHOICES_IMAGE, prefixes, "the third image")
    assert_match("4", CHOICES_IMAGE, prefixes, "the fourth image")


def test_numeric_with_parens():
    """Numeric prefix with parentheses."""
    prefixes = ["1", "2", "3", "4"]
    assert_match("(1)", CHOICES_IMAGE, prefixes, "the first image")
    assert_match("(3)", CHOICES_IMAGE, prefixes, "the third image")


def test_numeric_with_dot():
    """Numeric prefix with dot."""
    prefixes = ["1", "2", "3", "4"]
    assert_match("1.", CHOICES_IMAGE, prefixes, "the first image")
    assert_match("3.", CHOICES_IMAGE, prefixes, "the third image")


def test_multi_digit_numeric():
    """Multi-digit numeric prefix."""
    choices_12 = [f"option {i}" for i in range(12)]
    prefixes_12 = [str(i + 1) for i in range(12)]
    assert_match("12", choices_12, prefixes_12, "option 11")
    assert_match("1", choices_12, prefixes_12, "option 0")
    assert_match("(10)", choices_12, prefixes_12, "option 9")


# ============================================================
# Short choices
# ============================================================

def test_short_choices_yes_no():
    """Short choice texts like yes/no."""
    assert_match("A", CHOICES_SHORT, PREFIXES_LETTER_2, "yes")
    assert_match("B", CHOICES_SHORT, PREFIXES_LETTER_2, "no")
    assert_match("yes", CHOICES_SHORT, PREFIXES_LETTER_2, "yes")
    assert_match("no", CHOICES_SHORT, PREFIXES_LETTER_2, "no")


def test_short_choices_number_text():
    """Choices that are numbers as text."""
    assert_match("A", CHOICES_NUMBERS, PREFIXES_LETTER_4, "1")
    assert_match("C", CHOICES_NUMBERS, PREFIXES_LETTER_4, "3")
    assert_match("3", CHOICES_NUMBERS, PREFIXES_LETTER_4, "3",
                 msg="exact text match on '3'")


# ============================================================
# Edge cases
# ============================================================

def test_empty_predicted():
    """Empty predicted returns None."""
    assert_match("", CHOICES_IMAGE, PREFIXES_LETTER_4, None)
    assert_match("  ", CHOICES_IMAGE, PREFIXES_LETTER_4, None)


def test_no_match():
    """Predicted that doesn't match anything."""
    assert_match("Z", CHOICES_IMAGE, PREFIXES_LETTER_4, None,
                 msg="Z is not a valid prefix for 4 choices")
    assert_match("something completely unrelated", CHOICES_IMAGE, PREFIXES_LETTER_4, None)


def test_word_starting_with_prefix_letter():
    """Text starting with a valid prefix letter but followed by alnum — should NOT match prefix."""
    # "the" starts with "t" but should not match as a prefix
    # (only relevant if there are 20+ choices where T is a prefix)
    choices_20 = [f"option {i}" for i in range(20)]
    prefixes_20 = [chr(ord("A") + i) for i in range(20)]  # A-T
    assert_match("the third image", choices_20, prefixes_20, None,
                 msg="'the' should not match prefix T")


def test_mismatched_parens():
    """Mismatched parens still work reasonably."""
    assert_match("(C", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image",
                 msg="opening paren only")


def test_whitespace_handling():
    """Leading/trailing whitespace stripped."""
    assert_match("  C  ", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")
    assert_match("  the third image  ", CHOICES_IMAGE, PREFIXES_LETTER_4, "the third image")


def test_long_choices():
    """Longer choice texts."""
    assert_match("A", CHOICES_LONG, PREFIXES_LETTER_3, "a cat sitting on a mat")
    assert_match("a cat sitting on a mat", CHOICES_LONG, PREFIXES_LETTER_3, "a cat sitting on a mat")
    assert_match("cat sitting on a mat", CHOICES_LONG, PREFIXES_LETTER_3, "a cat sitting on a mat",
                 msg="substring match")


# ============================================================
# Tests for evaluate_answer (integration)
# ============================================================

def test_eval_correct_letter():
    """Correct answer via letter prefix."""
    assert_eval("C", "the third image", CHOICES_IMAGE, True)
    assert_eval("A", "the first image", CHOICES_IMAGE, True)


def test_eval_wrong_letter():
    """Wrong answer via letter prefix."""
    assert_eval("B", "the third image", CHOICES_IMAGE, False)
    assert_eval("D", "the first image", CHOICES_IMAGE, False)


def test_eval_correct_text():
    """Correct answer via choice text."""
    assert_eval("the third image", "the third image", CHOICES_IMAGE, True)


def test_eval_wrong_text():
    """Wrong answer via choice text."""
    assert_eval("the second image", "the third image", CHOICES_IMAGE, False)


def test_eval_empty():
    """Empty predicted is always wrong."""
    assert_eval("", "the third image", CHOICES_IMAGE, False)


def test_eval_no_match():
    """Unrecognized predicted is wrong."""
    assert_eval("Z", "the third image", CHOICES_IMAGE, False)


def test_eval_original_bug_fixed():
    """The critical bug fix: C should correctly match the third image."""
    assert_eval("C", "the third image", CHOICES_IMAGE, True,
                msg="CRITICAL: was False due to 'c' in 'second' bug")
    assert_eval("D", "the fourth image", CHOICES_IMAGE, True,
                msg="CRITICAL: was False due to 'd' in 'second' bug")


def test_eval_numeric_format():
    """Numeric option format."""
    assert_eval("3", "the third image", CHOICES_IMAGE, True, option_format='numeric')
    assert_eval("1", "the first image", CHOICES_IMAGE, True, option_format='numeric')


def test_eval_no_choices_fallback():
    """No choices → direct comparison."""
    assert_eval("the third image", "the third image", None, True)
    assert_eval("something else", "the third image", None, False)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
