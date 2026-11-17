"""Regression policy for autonomous evolution: runtime changes require green tests."""

from ubique.evolution import _requires_passing_suite


def test_runtime_python_changes_require_regression_suite():
    assert _requires_passing_suite(["src/ubique/brain/runtime.py"])
    assert _requires_passing_suite(["docs/notes.md", "src/ubique/providers/router.py"])


def test_non_runtime_changes_do_not_block_on_unrelated_regressions():
    assert not _requires_passing_suite(["docs/architecture.md"])
    assert not _requires_passing_suite(["tests/test_cognition.py"])
    assert not _requires_passing_suite(["src/ubique/data.json"])
    assert not _requires_passing_suite([])
