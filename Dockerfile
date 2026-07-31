# APGI Validation Framework
#
# Pins one Python version so results are reproducible across machines. The
# repository previously had no container definition and no lockfile, so
# "works on my machine" was the only guarantee available.
#
# Build:
#   docker build -t apgi-validation .
#
# Run the CLI:
#   docker run --rm -e APGI_ALLOW_EPHEMERAL_MASTER_KEY=1 apgi-validation --help
#
# Run with a persistent master key and mounted results:
#   docker run --rm \
#     -e APGI_MASTER_KEY="$APGI_MASTER_KEY" \
#     -v "$PWD/results:/app/results" \
#     apgi-validation comprehensive-validation

FROM python:3.11-slim-bookworm

# Tk is required by the GUI entry points; the scientific stack needs a
# toolchain for packages without manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        python3-tk \
        tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APGI_ENV=production \
    APGI_PRODUCTION_MODE=1 \
    APGI_LOG_LEVEL=INFO

# Dependencies first so code edits don't invalidate the layer.
COPY requirements.txt requirements-protocols.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

RUN python -m pip install --no-deps -e .

# Key material is generated at runtime into a directory owned by this user;
# never bake secrets into the image.
RUN useradd --create-home --uid 10001 apgi && \
    mkdir -p /app/.keys /app/logs /app/results && \
    chown -R apgi:apgi /app && \
    chmod 700 /app/.keys
USER apgi

# Fails the build if the CLI cannot start.
RUN APGI_ALLOW_EPHEMERAL_MASTER_KEY=1 python main.py --help > /dev/null

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
