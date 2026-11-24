def should_promote(candidate, incumbent, baseline, direction="lower"):
    """
    Compare validation scores only.

    direction:
        lower  -> RMSE, MAE, circular MAE
        higher -> F1, PR-AUC, R2
    """

    if direction not in ("lower", "higher"):
        raise ValueError("direction must be 'lower' or 'higher'")

    values = (candidate, incumbent, baseline)

    if any(value is None for value in values):
        return False, "Missing comparable validation score."

    if not all(isinstance(value, (int, float)) for value in values):
        return False, "Validation scores must be numeric."

    if direction == "lower":
        incumbent_improved = candidate < incumbent
        baseline_beaten = candidate < baseline
    else:
        incumbent_improved = candidate > incumbent
        baseline_beaten = candidate > baseline

    if not incumbent_improved:
        return False, "Candidate does not improve the registered model."

    if not baseline_beaten:
        return False, "Candidate does not beat the naive baseline."

    return True, "Candidate qualifies for a new registry version."


if __name__ == "__main__":
    # Basic policy tests; no registry access or model changes.
    assert should_promote(15, 17, 20, "lower")[0] is True
    assert should_promote(18, 17, 20, "lower")[0] is False
    assert should_promote(15, 17, 14, "lower")[0] is False
    assert should_promote(0.35, 0.30, 0.20, "higher")[0] is True
    assert should_promote(0.25, 0.30, 0.20, "higher")[0] is False
    assert should_promote(None, 17, 20, "lower")[0] is False

    print("All promotion policy tests passed.")
