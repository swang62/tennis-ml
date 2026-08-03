FROM python:3.12-slim

WORKDIR /app

# Minimal runtime deps for the dashboard only — see
# dashboard-requirements.txt (the full project pulls torch/lightning/etc.
# which this image does not need).
COPY infra/manifests/deploy/dashboard-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# The dashboard queries DuckDB directly; without the DB file the container
# crashes at startup (the app's get_players() opens it at import time).
COPY data/tennis.duckdb data/tennis.duckdb

EXPOSE 8501

CMD ["panel", "serve", "src/dashboard/app.py", "--port=8501", "--allow-websocket-origin=*"]
