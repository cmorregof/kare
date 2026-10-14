# KARE-E como bot de Telegram (Railway u otro servidor con Docker).
#
# Los PDF y el índice viven en un volumen montado en /data: los volúmenes se
# montan al arrancar, no al construir, así que la descarga y la ingesta van en
# el comando de inicio. Ingerir los dos PDF toma unos 20 segundos.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Primero las dependencias, que cambian poco; luego el código.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY app ./app
COPY config ./config
COPY corpus/manifest.yaml ./corpus/manifest.yaml
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    CORPUS_DIR=/data/fuentes \
    DB_PATH=/data/kare.db

# Si una fuente no baja o no coincide con su huella, el bot arranca igual con
# lo que haya indexado; los registros dicen qué faltó.
CMD ["sh", "-c", "kare descargar; kare ingerir; exec kare bot"]
