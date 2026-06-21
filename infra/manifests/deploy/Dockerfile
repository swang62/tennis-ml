FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir streamlit plotly

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.headless=true"]
