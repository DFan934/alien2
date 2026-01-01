import re


def _extract_run_roots_from_log(log_text: str) -> set[str]:
    """
    Extract artifact root candidates from logs. We detect both:
      - [RunContext] artifacts_root=...
      - any path segments containing \\artifacts\\a2_YYYY... or \\scripts\\artifacts\\a2_YYYY...
    Returns a set of normalized roots.
    """
    roots = set()

    # 1) explicit root echoes
    for m in re.finditer(r"\[RunContext\]\s+artifacts_root=([^\r\n]+)", log_text):
        roots.add(m.group(1).strip().rstrip("\\/"))

    # 2) implicit occurrences of a2 run roots
    # Windows-style (your logs)
    for m in re.finditer(r"([A-Za-z]:\\[^\r\n]*?\\artifacts\\a2_\d{8}_\d{6})", log_text):
        roots.add(m.group(1).strip().rstrip("\\/"))
    for m in re.finditer(r"([A-Za-z]:\\[^\r\n]*?\\scripts\\artifacts\\a2_\d{8}_\d{6})", log_text):
        roots.add(m.group(1).strip().rstrip("\\/"))

    return roots


def test_phase3_single_artifacts_root_in_one_run_log():
    """
    Phase 3 invariant:
      In a single run, artifacts must live under exactly one root.
      Not both ...\\artifacts\\a2_... and ...\\scripts\\artifacts\\a2_...
    """
    # Simulated "bad" log snippet that would fail Phase 3:
    bad = """
    [RunContext] artifacts_root=C:\\repo\\artifacts\\a2_20251231_160710
    [FE] Loading pre-fitted pipeline from C:\\repo\\scripts\\artifacts\\a2_20251231_160710\\_fe_meta\\pipeline.pkl
    """

    roots = _extract_run_roots_from_log(bad)
    assert len(roots) >= 2  # sanity: this snippet includes two different roots
    has_scripts_artifacts = any("\\scripts\\artifacts\\" in r for r in roots)
    has_artifacts_a2 = any("\\artifacts\\a2_" in r for r in roots)

    assert has_scripts_artifacts and has_artifacts_a2, f"Expected both roots in bad snippet, got {roots}"

    # Now a "good" snippet should pass:
    good = """
    [RunContext] artifacts_root=C:\\repo\\artifacts\\a2_20251231_160710
    [FE] Loading pre-fitted pipeline from C:\\repo\\artifacts\\a2_20251231_160710\\_fe_meta\\pipeline.pkl
    """
    roots2 = _extract_run_roots_from_log(good)
    assert len(roots2) == 1, f"Expected exactly 1 artifacts root, got {roots2}"
