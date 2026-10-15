# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM python:3.12-slim-bookworm

# g++ is required at runtime by CPU torch.compile / TorchInductor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 ca-certificates tini util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp \
    MPLCONFIGDIR=/cache/matplotlib \
    XDG_CACHE_HOME=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    PYTEST_ADDOPTS="-o cache_dir=/cache/pytest"
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv sync --locked --group docs --no-managed-python
# Match Galaxy; the second flag is read by the pinned PyTorch version.
ENV TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
COPY Makefile zensical.toml README.md ./
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/
# A sticky workspace lets arbitrary host UIDs create/remove their own Zensical
# cache while protecting the root-owned source files from deletion.
RUN groupadd --gid 1000 benchmark \
    && useradd --uid 1000 --gid benchmark --no-create-home benchmark \
    && mkdir -p training_data artifacts site /cache \
    && chmod 1777 /app artifacts site /cache
USER 1000:1000
EXPOSE 8000
# Forward stop signals to the command group; reap orphaned loader workers.
ENTRYPOINT ["/usr/local/bin/python3", "/app/scripts/container_entrypoint.py"]
CMD ["make", "help"]
