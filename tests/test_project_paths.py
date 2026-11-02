from pathlib import Path
import ast
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import project_paths


@pytest.mark.parametrize("prefix", project_paths.LEGACY_LOCAL_ROOTS)
def test_legacy_manifests_rebase_to_current_checkout(tmp_path, prefix):
    assert project_paths.resolve_project_path(prefix + "/data/a.wav", tmp_path) == tmp_path / "data/a.wav"


def test_windows_separators_and_case_are_supported(tmp_path):
    raw = r"e:\DOCUMENTS\tts-audio\tts-exp\data\a.wav"
    assert project_paths.resolve_project_path(raw, tmp_path) == tmp_path / "data/a.wav"


def test_old_file_existing_does_not_change_resolution(tmp_path, monkeypatch):
    old, current = tmp_path / "old", tmp_path / "current"
    old.mkdir(); current.mkdir()
    (old / "a.wav").write_bytes(b"old")
    (current / "a.wav").write_bytes(b"current")
    monkeypatch.setattr(project_paths, "LEGACY_LOCAL_ROOTS", (str(old),))
    assert project_paths.resolve_project_path(old / "a.wav", current).read_bytes() == b"current"


def test_unrelated_paths_are_not_rebased(tmp_path):
    external = tmp_path.parent / "other" / "a.wav"
    assert project_paths.resolve_project_path(external, tmp_path) == external.resolve()
    assert not project_paths.is_legacy_project_path("/home/wjj/tts-audio/tts-exp/data/a.wav")
    assert not project_paths.is_legacy_project_path("/mnt/e/Documents/tts-audio/tts-exp-other/a.wav")


def test_legacy_traversal_cannot_escape_current_checkout(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        project_paths.resolve_project_path(project_paths.LEGACY_LOCAL_ROOTS[0] + "/../outside", tmp_path)


# Execute the production resolver functions in isolation: these path operations
# must remain testable without importing GPU, audio, or VSR inference runtimes.
USERS = [
    ("42_extract_rhythm_factors.py", "resolve_path", "REPO_ROOT"),
    ("43_build_rhythm_counterfactuals.py", "resolve_path", "REPO_ROOT"),
    ("46_build_pause_counterfactuals.py", "resolve_path", "REPO"),
    ("prepare_aishell1_n15_sampling_rate_isolation.py", "local_path", "repo"),
    ("prepare_aishell1_mfa_linear_cohort.py", "localize_path", "repo"),
    ("run_knn_vc_poc.py", "_legacy_to_local", "REPO_ROOT"),
    ("experiments/vsr_ditto50_linkage.py", "_resolve_source", "ROOT"),
]


@pytest.mark.parametrize("filename, function_name, root_name", USERS)
def test_script_resolvers_use_the_current_checkout(tmp_path, filename, function_name, root_name):
    source = Path(__file__).resolve().parents[1] / "scripts" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    ns = {"Path": Path, "Any": object, "REPO": tmp_path, root_name: tmp_path,
          "resolve_project_path": project_paths.resolve_project_path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    (tmp_path / "a.wav").write_bytes(b"current")
    old = project_paths.LEGACY_LOCAL_ROOTS[0] + "/a.wav"
    if function_name == "local_path":
        actual = ns[function_name](tmp_path, old)
    else:
        actual = ns[function_name](old)
    assert actual == tmp_path / "a.wav"


def test_diagnostic_manifest_maps_foreign_path_before_windows_path_checks(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/run_two_stage_diagnostics.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_localize_path")
    ns = {"Path": Path, "Any": object,
          "resolve_project_path": project_paths.resolve_project_path,
          "is_legacy_project_path": project_paths.is_legacy_project_path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    (tmp_path / "a.wav").write_bytes(b"current")
    kwargs = dict(source_manifest=tmp_path / "manifests/input.json", repo_root=tmp_path, field="audio")
    for prefix in project_paths.LEGACY_LOCAL_ROOTS:
        assert ns["_localize_path"](prefix + "/a.wav", **kwargs) == str(tmp_path / "a.wav")
    with pytest.raises(FileNotFoundError):
        ns["_localize_path"](project_paths.LEGACY_LOCAL_ROOTS[0] + "/missing.wav", **kwargs)


@pytest.mark.parametrize("filename", ["download_vfhq_samples.py", "download_vfhq_v2.py", "download_vfhq_v3.py"])
def test_downloaders_output_belongs_to_their_own_checkout(tmp_path, filename):
    source = Path(__file__).resolve().parents[1] / "scripts" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in {"REPO", "OUT"} for t in n.targets)]
    ns = {"Path": Path, "__file__": str(tmp_path / "scripts" / filename)}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec"), ns)
    assert ns["OUT"].is_relative_to(tmp_path / "data")
