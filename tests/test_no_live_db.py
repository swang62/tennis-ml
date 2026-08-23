"""Static guard against live database access from tests."""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

_FORBIDDEN_FIXTURE_NAMES = {
    "postgres_ready",
    "gold_ready",
    "seeded_test_db",
    "_postgres_reachable",
}
_CONNECTION_FUNCS = {"get_conn", "get_pool", "connection"}


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name-like token: identifiers, attribute names, and string literals."""
    ids: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            ids.add(node.value)
    return ids


def _failures(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    ids = _identifiers(tree)
    mentions_psycopg = "psycopg" in ids  # boundary-mocked files patch psycopg
    mocks_copy_tables = "_copy_tables" in ids

    client_aliases: set[str] = set()
    problems: list[str] = []

    def record(node: ast.AST, message: str) -> None:
        problems.append(f"line {getattr(node, 'lineno', '?'):>3}: {message}")

    for node in ast.walk(tree):
        # Deleted live-db fixture names are never legitimate in test code.
        forbidden_name: str | None = None
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_FIXTURE_NAMES:
            forbidden_name = node.id
        elif isinstance(node, ast.arg) and node.arg in _FORBIDDEN_FIXTURE_NAMES:
            forbidden_name = node.arg
        if forbidden_name is not None:
            record(node, f"forbidden live-db fixture name {forbidden_name!r}")
            continue

        # Direct imports bind the connection-opening client.
        if isinstance(node, ast.ImportFrom):
            if node.module == "src.db.client":
                for alias in node.names:
                    if alias.name in _CONNECTION_FUNCS:
                        record(node, f"imports {node.module}.{alias.name}")
            elif node.module == "src.db":
                for alias in node.names:
                    if alias.name == "client":
                        client_aliases.add(alias.asname or "client")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.db.client":
                    client_aliases.add(alias.asname or "client")

        # Reject connection calls without a boundary mock signal.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _CONNECTION_FUNCS:
                record(node, f"unqualified {func.id}() call")
            elif isinstance(func, ast.Attribute):
                if (
                    func.attr in _CONNECTION_FUNCS
                    and isinstance(func.value, ast.Name)
                    and func.value.id in client_aliases
                    and not mentions_psycopg
                ):
                    record(node, f"client {func.attr}() call without a psycopg mock")
                if func.attr == "refresh_snapshot" and not mocks_copy_tables:
                    record(node, "refresh_snapshot() call without a _copy_tables mock")

    return problems


def test_no_live_database_usage_in_test_code() -> None:
    failures: list[str] = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == Path(__file__).name:
            continue  # the guard's own constants name the forbidden fixtures
        failures.extend(f"{path.name}: {message}" for message in _failures(path))
    assert not failures, "live-database usage found in test code:\n" + "\n".join(failures)
