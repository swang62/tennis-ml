"""Static guard: `just dev` rebuilds the player index before serving.

dev.sh is a monolithic script with database and server side effects, so it
cannot run hermetically. This guard statically asserts the Task-5 wiring: the
player-directory generator and the Node index builder run after the
database/tool preflight and before any server starts, and each step fails the
script on error — so Vite can never serve a stale or fixture directory.
"""

import re
from pathlib import Path

DEV_SH = Path(__file__).resolve().parent.parent / "scripts" / "dev.sh"

_GENERATOR = "generate_navigation_artifacts"
_NODE_BUILDER = "web/scripts/build-player-index.mjs"


def _lines() -> list[str]:
    return DEV_SH.read_text().splitlines()


def _index(needle: str, lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if needle in line:
            return i
    raise AssertionError(f"dev.sh is missing expected marker {needle!r}")


def test_index_rebuild_runs_after_preflight_and_before_servers():
    lines = _lines()
    preflight_end = _index("command -v pnpm", lines)
    generator = _index(_GENERATOR, lines)
    node_build = _index(_NODE_BUILDER, lines)
    server_start = _index("bentoml serve", lines)

    assert preflight_end < generator < node_build < server_start, (
        "player-index rebuild must run after the database/tool preflight and "
        "before Bento/Vite start"
    )


def test_index_rebuild_steps_fail_fast():
    lines = _lines()
    generator = _index(_GENERATOR, lines)
    node_build = _index(_NODE_BUILDER, lines)
    server_start = _index("bentoml serve", lines)

    # The generator step is guarded by `' || {` ... `exit 1` before the node step.
    assert any(line.endswith("' || {") for line in lines[generator:node_build]), (
        "player-directory generation must be guarded with `' || {`"
    )
    assert any(re.match(r"^\s*exit 1", line) for line in lines[generator:node_build]), (
        "player-directory generation failure must exit 1"
    )

    # The node step is guarded by `|| {` ... `exit 1` before the servers start.
    assert any(line.endswith("|| {") for line in lines[node_build:server_start]), (
        "player index build must be guarded with `|| {`"
    )
    assert any(re.match(r"^\s*exit 1", line) for line in lines[node_build:server_start]), (
        "player index build failure must exit 1"
    )
