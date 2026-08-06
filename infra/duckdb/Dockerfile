# Production DuckDB companion image: serves /data/tennis.duckdb over Quack.
# The database is mounted at /data (never baked into any image); the Bento
# connects to this server rather than shipping the DB itself.
FROM python:3.12-slim

# duckdb pinned to the same version the rest of the stack uses; the `quack`
# extension installs from the `core` repository at runtime.
RUN pip install --no-cache-dir duckdb==1.5.4

COPY server.py /usr/local/bin/quack_server.py
COPY healthcheck.py /usr/local/bin/quack_healthcheck.py
COPY init.sql /usr/local/share/quack/init.sql
COPY entrypoint.py /usr/local/bin/quack_entrypoint.py

# The server owns the production DB. Compose mounts a named volume here.
VOLUME ["/data"]

EXPOSE 9494

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "/usr/local/bin/quack_healthcheck.py"]

CMD ["python", "/usr/local/bin/quack_entrypoint.py"]