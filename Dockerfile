# APGI Validation Framework
#
# Pins one Python version so results are reproducible across machines. The
# repository previously had no container definition and no lockfile, so
# "works on my machine" was the only guarantee available.
#
# Build:
#   docker build -t apgi-validation .
#
# Run the CLI (the image runs with APGI_ENV=production, so a master key is
# mandatory; ephemeral keys are refused in production — APGI-IMP-002):
#   docker run --rm -e APGI_MASTER_KEY="$APGI_MASTER_KEY" apgi-validation --help
#
# Run with a persistent master key and mounted results:
#   docker run --rm \
#     -e APGI_MASTER_KEY="$APGI_MASTER_KEY" \
#     -v "$PWD/results:/app/results" \
#     apgi-validation comprehensive-validation

# Base image pinned by digest (APGI-IMP-001); the tag is informational.
# Refresh deliberately: re-resolve the digest, rebuild, record it in the release manifest.
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

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

# Dependencies first so code edits don't invalidate the layer. Every package is
# installed from the hash-free but exact-version lock (requirements.lock) so the
# image resolves the same set CI tests (APGI-IMP-001).
COPY requirements.txt requirements-protocols.txt requirements.lock ./
RUN python -m pip install -r requirements.txt -c requirements.lock

COPY . .

RUN python -m pip install --no-deps -e .

# Key material is generated at runtime into a directory owned by this user;
# never bake secrets into the image.
RUN useradd --create-home --uid 10001 apgi && \
    mkdir -p /app/.keys /app/logs /app/results && \
    chown -R apgi:apgi /app && \
    chmod 700 /app/.keys
USER apgi

# Fails the build if the CLI cannot start. APGI_ENV is overridden for this one
# build-time check only, because production refuses ephemeral keys.
RUN APGI_ENV=build-check APGI_ALLOW_EPHEMERAL_MASTER_KEY=1 python main.py --help > /dev/null

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
