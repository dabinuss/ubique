from ubique.fzg import FZG_VERSION, fzg_profile_template, validate_case


def test_fzg_version_is_pinned():
    assert FZG_VERSION == "1.0"


def test_validate_case_requires_all_preconditions():
    errors = validate_case({"S": "Ubique"})
    assert any("A" in error for error in errors)
    assert any("M_S_minus" in error for error in errors)


def test_valid_case_has_no_errors():
    case = {
        "S": "Ubique agent",
        "A": "complete labelled repository task",
        "C": "GitHub Actions cycle",
        "Q": "task completion score",
        "M_S": "planner-provider-executor loop",
        "M_S_minus": "same cycle with planner disabled",
    }
    assert validate_case(case) == []


def test_intelligence_is_profile_not_scalar():
    profile = fzg_profile_template()
    assert set(profile["I_C"]) == {"Z", "K", "R", "L"}
    assert "intelligence_score" not in profile
