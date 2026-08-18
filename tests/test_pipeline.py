"""Hermetic tests for the standalone training pipeline runner.

Covers the player-similarity rebuild wiring and the papermill notebook call
contract (live cell output streaming via ``log_output=True``). No live
database, MLflow, or papermill execution.
"""

import ast
import json
import re
from pathlib import Path

from src.constants import (
    PLAYSTYLE_CLUSTER_LABELS,
    PLAYSTYLE_N_CLUSTERS,
    PLAYSTYLE_RANDOM_STATE,
)
from src.flows import pipeline
from src.models import similarity


def test_generate_cluster_artifacts_uses_snapshot_query_and_configured_config(monkeypatch):
    """The runner generates cluster artifacts from the fresh DuckDB snapshot
    with the configured count/labels/seed — never a live DB; the index build
    then consumes them."""

    class _FakeQuery:
        def __call__(self, _sql):
            return None

    fake_query = _FakeQuery()
    monkeypatch.setattr(pipeline.training, "to_dataframe", fake_query)
    captured: dict[str, object] = {}

    def _fake_build(n_clusters, labels, query=None, *, random_state=42):
        captured.update(
            n_clusters=n_clusters, labels=labels, query=query, random_state=random_state
        )

    monkeypatch.setattr(similarity, "build_cluster_artifacts", _fake_build)

    pipeline.generate_cluster_artifacts()

    assert captured["query"] is fake_query
    assert captured["n_clusters"] == PLAYSTYLE_N_CLUSTERS
    assert captured["labels"] == PLAYSTYLE_CLUSTER_LABELS
    assert captured["random_state"] == PLAYSTYLE_RANDOM_STATE


def test_run_notebook_streams_cell_output_and_keeps_existing_contract(monkeypatch, tmp_path):
    """Cell stdout/stderr is streamed live (``log_output=True``) while the
    input/output paths, kernel, and parameters are forwarded unchanged."""
    calls: list[dict[str, object]] = []

    def _fake_execute_notebook(**kwargs):
        calls.append(kwargs)
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
    assert set(kwargs) == {
        "input_path",
        "output_path",
        "kernel_name",
        "parameters",
        "log_output",
    }


def test_pipeline_source_has_no_navigation_build_or_mlflow_pins():
    source = pipeline.__file__
    assert source is not None
    text = Path(source).read_text()
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


def test_02_nn_notebook_tuning_budget_is_pinned():
    """The NN tuner's budget is pinned in the tagged parameter cell: 20 Optuna
    trials, 50 max epochs, 10-epoch early-stopping patience. The pipeline runs
    this notebook with no overrides, so these defaults are exactly what the
    next run executes. Every code cell must still parse (AST only, no
    execution)."""
    notebook = json.loads(Path("notebooks/parameters/02_tune_nn.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    param_cell = next(
        cell
        for cell in notebook["cells"]
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    )
    budget = {}
    for node in ast.parse("".join(param_cell["source"])).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"n_trials", "max_epochs", "patience"}
        ):
            budget[node.targets[0].id] = ast.literal_eval(node.value)
    assert budget == {"n_trials": 20, "max_epochs": 50, "patience": 10}
