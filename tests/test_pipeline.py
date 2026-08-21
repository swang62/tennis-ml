"""Hermetic tests for the standalone training pipeline runner."""

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


def test_parameter_notebooks_declare_a_validating_nbformat():
    """Every parameter notebook must validate under the runtime nbformat schema."""
    import nbformat

    for path in sorted(Path("notebooks/parameters").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        if any("id" in cell for cell in nb.cells):
            assert nb.nbformat_minor >= 5, (
                f"{path.name} carries cell ids but declares nbformat_minor={nb.nbformat_minor}"
            )
        nbformat.validate(nb)  # raises NotebookValidationError on the original mismatch
