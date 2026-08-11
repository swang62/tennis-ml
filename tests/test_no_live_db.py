"""Static guard: test code must never reach for a live database.

Fails when a test module reintroduces the deleted live-db fixture names
(postgres_ready / gold_ready / seeded_test_db) or opens a production database
client connection (importing get_conn, calling it, or refreshing the training
snapshot) without demonstrably mocking the connection boundary. The guard
deliberately allows the legitimate boundary-mocked patterns:
- module imports of src.db.client (test_db_client unit-tests the client with a
  fake psycopg.connect and never opens a real connection);
- qualified get_conn() calls inside a file that references psycopg (the mock
  signal), as in the client's own unit tests;
- snapshot.refresh_snapshot() inside a file that mocks _copy_tables.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

_FORBIDDEN_FIXTURE_NAMES = {
    "postgres_ready",
    "gold_ready",
    "seeded_test_db",
    "_postgres_reachable",
}
_CONNECTION_FUNCS = {"get_conn"}


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
        # Deleted live-db fixture names are never legitimate in test code —
        # neither as identifiers nor as fixture parameters (def test_x(...)).
        forbidden_name: str | None = None
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_FIXTURE_NAMES:
            forbidden_name = node.id
        elif isinstance(node, ast.arg) and node.arg in _FORBIDDEN_FIXTURE_NAMES:
            forbidden_name = node.arg
        if forbidden_name is not None:
            record(node, f"forbidden live-db fixture name {forbidden_name!r}")
            continue

        # Direct function imports bind the connection-opening client.
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

        # Connection calls that are not demonstrably mocked at the boundary.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _CONNECTION_FUNCS:
                record(node, f"unqualified {func.id}() call")
            elif isinstance(func, ast.Attribute):
                if (
                    func.attr == "get_conn"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in client_aliases
                    and not mentions_psycopg
                ):
                    record(node, "client get_conn() call without a psycopg mock")
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
