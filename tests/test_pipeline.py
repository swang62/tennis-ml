"""Hermetic tests for the standalone training pipeline runner.

Covers the papermill notebook call contract (live cell output streaming via
``log_output=True``). No live database, MLflow, or papermill execution.
"""

import ast
import json
import re
from pathlib import Path

from src.flows import pipeline


def test_run_notebook_streams_cell_output_and_keeps_existing_contract(monkeypatch, tmp_path):
    """Cell stdout/stderr is streamed live (``log_output=True``) while the
    input/output paths, kernel, and parameters are forwarded unchanged."""
    calls: list[dict[str, object]] = []

    def _fake_execute_notebook(**kwargs):
        calls.append(kwargs)
        Path(str(kwargs["output_path"])).write_text("executed notebook")
        return None

    monkeypatch.setattr(pipeline.pm, "execute_notebook", _fake_execute_notebook)
    monkeypatch.setattr(pipeline, "OUTPUTS", tmp_path / "out")
    monkeypatch.setattr(pipeline, "PARAMS", tmp_path / "params")
    monkeypatch.setattr(pipeline, "ensure_kernel", lambda: "test-kernel")

    pipeline.run_notebook("00_test.ipynb", parameters={"alpha": 0.5})

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["log_output"] is True
    assert kwargs["input_path"] == str(tmp_path / "params" / "00_test.ipynb")
    assert kwargs["kernel_name"] == "test-kernel"
    assert kwargs["parameters"] == {"alpha": 0.5}
    output = Path(str(kwargs["output_path"]))
    assert output.parent == tmp_path / "out"
    assert re.fullmatch(r"\d{8}_\d{6}_00_test\.ipynb", output.name)
    assert (tmp_path / "out" / "latest_00_test.ipynb").read_text() == "executed notebook"
    assert set(kwargs) == {
        "input_path",
        "output_path",
        "kernel_name",
        "parameters",
        "log_output",
    }


def test_selected_notebooks_runs_only_evaluation_for_promotion():
    assert pipeline.selected_notebooks(promote_only=True) == ["04_evaluate.ipynb"]
    assert pipeline.selected_notebooks(promote_only=False) == pipeline.NB_ORDER


def test_pipeline_source_has_no_navigation_build_or_mlflow_pins():
    source = pipeline.__file__
    assert source is not None
    text = Path(source).read_text()
    assert "refresh_snapshot" not in text
    assert "build_similarity_index" not in text
    assert "similarity_pins.json" not in text
    assert "mlflow" not in text
    notebook = Path("notebooks/parameters/03_train_ensemble.ipynb").read_text()
    assert "similarity_pins.json" not in notebook


def test_parameter_notebooks_declare_a_validating_nbformat():
    """Every parameter notebook must validate under the runtime nbformat
    schema. Cell ids are an nbformat 4.5+ field, so any notebook carrying them
    must declare nbformat_minor >= 5 — a notebook declared as 4.4 with ids
    previously failed papermill with 'Additional properties are not allowed
    (\"id\" was unexpected)'."""
    import nbformat

    for path in sorted(Path("notebooks/parameters").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        if any("id" in cell for cell in nb.cells):
            assert nb.nbformat_minor >= 5, (
                f"{path.name} carries cell ids but declares nbformat_minor={nb.nbformat_minor}"
            )
        nbformat.validate(nb)  # raises NotebookValidationError on the original mismatch


def test_parameter_notebooks_use_central_split_constants():
    """Split fractions and the CV fold count live in src.constants only: the
    01 split notebook imports all four central constants, every 02 tuner
    imports CV_FOLDS, and no parameter notebook defines local test_size /
    val_size / cv_folds."""
    notebook_names = {
        "01_train_test_split.ipynb",
        "02_tune_linear.ipynb",
        "02_tune_gbdt.ipynb",
        "02_tune_nn.ipynb",
    }
    local_split_names = {"test_size", "val_size", "cv_folds"}
    for path in sorted(Path("notebooks/parameters").glob("*.ipynb")):
        if path.name not in notebook_names:
            continue
        notebook = json.loads(path.read_text())
        code_sources = [
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert code_sources, f"{path.name}: expected at least one code cell"
        imported: set[str] = set()
        for src in code_sources:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "src.constants":
                    imported.update(alias.name for alias in node.names)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in local_split_names:
                            raise AssertionError(f"{path.name}: local {target.id} definition")
        if path.name.startswith("01_"):
            assert {"TRAIN_FRACTION", "VAL_FRACTION", "TEST_FRACTION", "CV_FOLDS"} <= imported, (
                f"{path.name}: missing central split constants"
            )
        else:
            assert "CV_FOLDS" in imported, f"{path.name}: missing central CV_FOLDS import"


def test_split_fractions_are_90_5_5():
    """The chronological split is 90/5/5: these exact fraction constants drive
    the 01 cutoffs and must stay the pipeline's source of truth."""
    from src.constants import TEST_FRACTION, TRAIN_FRACTION, VAL_FRACTION

    assert (TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION) == (0.90, 0.05, 0.05)
    assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9


def test_02_linear_notebook_selects_no_svm_candidate():
    """The linear tuner offers only LogisticRegression and GaussianNB. SVC was
    removed: it is superquadratic on the current data scale and needs internal
    cross-validation for probabilities, breaking the native-predict_proba
    contract shared with generic sklearn serving."""
    notebook = Path("notebooks/parameters/02_tune_linear.ipynb").read_text()
    assert "SVC" not in notebook
    assert re.search(r"svm", notebook, re.IGNORECASE) is None


def test_02_linear_notebook_never_passes_penalty_to_logistic_regression():
    """sklearn 1.8 deprecated LogisticRegression's penalty param (removal in
    1.10). Every construction — Optuna builder, final refit, OOF refit — must
    keep the L2 default and express the unregularized alternative as
    C=float('inf') behind the ``unpenalized`` search flag, with no penalty
    keyword ever reaching the constructor."""
    notebook = json.loads(Path("notebooks/parameters/02_tune_linear.ipynb").read_text())
    code_sources = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_sources, "expected at least one code cell"
    flag_seen = False
    for src in code_sources:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LogisticRegression"
            ):
                assert not any(kw.arg == "penalty" for kw in node.keywords if kw.arg is not None)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "trial"
                and node.func.attr == "suggest_categorical"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "unpenalized"
                and not any(kw.arg == "penalty" for kw in node.keywords)
            ):
                flag_seen = True
    assert flag_seen, "expected the 'unpenalized' search flag to survive"
