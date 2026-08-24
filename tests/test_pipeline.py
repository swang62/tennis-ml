"""Hermetic tests for the standalone training pipeline."""

from pathlib import Path

import nbformat

from src.flows import pipeline


def test_run_notebook_copies_latest_output(monkeypatch, tmp_path):
    def _fake_execute_notebook(output_path, **_kwargs):
        Path(str(output_path)).write_text("executed notebook")
        return None

    monkeypatch.setattr(pipeline.pm, "execute_notebook", _fake_execute_notebook)
    monkeypatch.setattr(pipeline, "OUTPUTS", tmp_path / "out")
    monkeypatch.setattr(pipeline, "PARAMS", tmp_path / "params")
    monkeypatch.setattr(pipeline, "ensure_kernel", lambda: "test-kernel")

    pipeline.run_notebook("00_test.ipynb")

    assert (tmp_path / "out" / "latest_00_test.ipynb").read_text() == "executed notebook"


def test_selected_notebooks_runs_only_evaluation_for_promotion():
    assert pipeline.selected_notebooks(promote_only=True) == ["04_evaluate.ipynb"]


def test_parameter_notebooks_declare_a_validating_nbformat():
    """Every parameter notebook must validate under the runtime nbformat schema."""

    for path in sorted(Path("notebooks/parameters").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        if any("id" in cell for cell in nb.cells):
            assert nb.nbformat_minor >= 5, (
                f"{path.name} carries cell ids but declares nbformat_minor={nb.nbformat_minor}"
            )
        nbformat.validate(nb)  # raises NotebookValidationError on the original mismatch


def test_tuning_notebooks_write_score_json_to_output_dir():
    """03_train_ensemble reads `{name}_score.json` from output_dir for every
    STACK_ORDER model; each 02 tuning notebook must write its score there, never
    to input_dir. This pins the contract that caused the missing
    /models/linear_score.json fresh-run failure."""

    from src.constants import STACK_ORDER

    for name in STACK_ORDER:
        path = Path("notebooks/parameters") / f"02_tune_{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        source = "\n".join(cell.get("source", "") for cell in nb.cells if cell.cell_type == "code")
        assert f'f"{{output_dir}}/{name}_score.json"' in source, (
            f"{path.name} must write {name}_score.json to output_dir for the ensemble"
        )
        assert f'f"{{input_dir}}/{name}_score.json"' not in source, (
            f"{path.name} must not write {name}_score.json to input_dir"
        )
