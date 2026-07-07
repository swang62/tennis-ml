FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 5006

CMD ["panel", "serve", "src/dashboard/app.py", "--port=5006", "--allow-websocket-origin=*"]
