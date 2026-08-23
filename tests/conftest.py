"""Shared hermetic test fixtures."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch):
    """Block external network connections while allowing loopback."""
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
