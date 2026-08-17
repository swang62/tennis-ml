"""Shared test fixtures.

No live-database fixtures live here: the seeded_test_db / postgres_ready /
gold_ready fixtures were removed because the suite must be hermetic (see
AGENTS.md). Every test mocks the database boundary or uses an in-memory
fixture; the inference-builder suite keeps its own in-memory DuckDB stand-in
inside test_inference_features.py.

An autouse fixture blocks external network access so a test that reaches a
real MLflow/DagsHub, Prefect, or any other remote service fails immediately
instead of leaking a live request. Loopback stays open so local-only code
paths behave unchanged.
"""

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch):
    """Fail any test that tries to open a connection to a non-loopback address.

    The suite must be hermetic: no live services, ever (see AGENTS.md). The
    guard patches socket.socket.connect (the shared path for TCP, HTTP, and
    HTTPS through requests/urllib3/http.client), allows loopback, and raises
    with a clear message for anything else.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if not isinstance(address, tuple):
            # Non-tuple addresses are local AF_UNIX sockets, not the network.
            return real_connect(self, address)
        host = address[0]
        if isinstance(host, str) and host.startswith(("127.", "::1", "localhost")):
            return real_connect(self, address)
        raise RuntimeError(
            f"test attempted an external network connection to {address!r} "
            "- the suite must be hermetic; mock the external service boundary"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
