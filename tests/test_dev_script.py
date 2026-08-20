"""Static guard: `just dev` stages the FAISS similarity artifacts before serving.

dev.sh is a monolithic script with database and server side effects, so it
cannot run hermetically. This guard statically asserts the Task-3 wiring: the
snapshot similarity staging runs after the database/tool preflight and before
any server starts, fails the script on error — so Bento can never start with
stale or missing similarity assets — and no player-directory/MiniSearch build
step remains anywhere.
"""

import re
from pathlib import Path

DEV_SH = Path(__file__).resolve().parent.parent / "scripts" / "dev.sh"
JUSTFILE = Path(__file__).resolve().parent.parent / "justfile"

_GENERATOR = "generate_similarity_artifacts"
_NODE_BUILDER = "build-player-index.mjs"


def _lines() -> list[str]:
    return DEV_SH.read_text().splitlines()


def _index(needle: str, lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if needle in line:
            return i
    raise AssertionError(f"dev.sh is missing expected marker {needle!r}")


def test_similarity_staging_runs_after_preflight_and_before_servers():
    lines = _lines()
    preflight_end = _index("command -v pnpm", lines)
    generator = _index(_GENERATOR, lines)
    server_start = _index("bentoml serve", lines)

    assert preflight_end < generator < server_start, (
        "similarity artifact staging must run after the database/tool preflight and "
        "before Bento/Vite start"
    )


def test_similarity_staging_fails_fast():
    lines = _lines()
    generator = _index(_GENERATOR, lines)
    server_start = _index("bentoml serve", lines)

    # The staging step is guarded by `' || {` ... `exit 1` before the servers start.
    assert any(line.endswith("' || {") for line in lines[generator:server_start]), (
        "similarity artifact staging must be guarded with `' || {`"
    )
    assert any(re.match(r"^\s*exit 1", line) for line in lines[generator:server_start]), (
        "similarity artifact staging failure must exit 1"
    )


def test_dev_has_no_player_directory_or_minisearch_build():
    """The node index builder and raw-directory staging are gone from dev: the
    only deploy-time staging left is the FAISS similarity generation shared
    with `just deploy`."""
    dev = DEV_SH.read_text()
    just = JUSTFILE.read_text()
    assert _NODE_BUILDER not in dev
    assert "player-directory" not in dev
    assert "generate_navigation_artifacts" not in dev
    # Both dev and deploy consume the same snapshot similarity generator.
    assert _GENERATOR in dev
    assert "src/flows/deploy.py" in just
