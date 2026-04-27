# syntax=docker/dockerfile:1.9
#
# Multi-stage Docker build for mnem-http.
#
# Default build: no onnx feature. Small, ~40MB final image.
# To enable the in-process neural-sparse lane:
#   docker build --build-arg FEATURES="onnx" -t mnem-http:onnx .
# That pulls the ort + tokenizers + hf-hub deps and the runtime image
# ships the onnxruntime shared library (~18MB extra).
#
# The container listens on 9876 by default. Mount a host directory at
# /data and point mnem-http at it:
#   docker run --rm -v $(pwd)/repo:/data -p 9876:9876 mnem-http:latest
#
# OCI image labels are set from ARGs so CI can inject the git SHA.

ARG RUST_VERSION=1.95
ARG FEATURES=""

# -----------------------------------------------------------------------------
# Build stage
# -----------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-trixie AS build

ARG FEATURES

WORKDIR /work

# Deps: pkg-config + libssl for any transitive TLS needs; ca-certificates so
# runtime HTTPS (ollama, OpenAI) works. `protobuf-compiler` not needed for
# the default feature set but kept commented for ort future needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency layers by copying the manifests first. A workspace with
# many crates means a monolithic copy of `crates/` causes a full rebuild on
# any code change; breaking it in two keeps the dep layer hot.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build the HTTP server binary. The `--features` flag is a no-op when the
# `FEATURES` arg is empty; otherwise we enable whatever the caller asked for
# on mnem-http (e.g. FEATURES="onnx" -> mnem-http[onnx]).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/target \
    set -eux; \
    if [ -n "${FEATURES}" ]; then \
        cargo build --release --locked -p mnem-http --features "${FEATURES}"; \
    else \
        cargo build --release --locked -p mnem-http; \
    fi; \
    cp target/release/mnem-http /usr/local/bin/mnem-http

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0

LABEL org.opencontainers.image.title="mnem-http"
LABEL org.opencontainers.image.description="HTTP server for mnem, the git-for-knowledge-graphs agent-memory substrate"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/Uranid/mnem"
LABEL org.opencontainers.image.documentation="https://github.com/Uranid/mnem/tree/main/docs"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# audit-2026-04-25 C5-1 (BENCH-2): when built with --build-arg FEATURES=onnx,
# the ort crate dynamically links against libonnxruntime.so but the runtime
# image does not ship it -- ldd shows the symbol unresolved and the ONNX
# provider fails silently at runtime. Download the upstream onnxruntime
# tarball into /usr/local/lib and add LD_LIBRARY_PATH so the loader finds
# it. Skipped automatically when FEATURES is empty (no-op layer; ~18MB
# regression only on -onnx images). Pinned to v1.20.1 to match the ort
# 2.0.0-rc.x ABI used in mnem-http's Cargo.toml.
ARG FEATURES
RUN if [ "${FEATURES}" = "onnx" ] || echo "${FEATURES}" | grep -qw onnx; then \
        ARCH=$(uname -m); \
        case "${ARCH}" in \
            x86_64)  ORT_ARCH=x64 ;; \
            aarch64) ORT_ARCH=aarch64 ;; \
            *)       echo "unsupported arch ${ARCH} for onnxruntime" >&2; exit 1 ;; \
        esac; \
        curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-${ORT_ARCH}-1.20.1.tgz" \
            | tar xz -C /tmp \
            && mv /tmp/onnxruntime-linux-${ORT_ARCH}-1.20.1/lib/* /usr/local/lib/ \
            && rm -rf /tmp/onnxruntime*; \
    fi
ENV LD_LIBRARY_PATH=/usr/local/lib

# Non-root user: the container only needs read/write on /data (repo file)
# and /models (optional HF cache). The mnem-http binary resolves config
# from `<repo>/.mnem/config.toml` (i.e. `/data/.mnem/config.toml` with
# the default `--repo /data`); there is no separate `/config` directory.
RUN groupadd --system --gid 1000 mnem \
    && useradd  --system --uid 1000 --gid mnem --home /home/mnem --shell /bin/false mnem \
    && mkdir -p /data /models \
    && chown -R mnem:mnem /data /models

COPY --from=build /usr/local/bin/mnem-http /usr/local/bin/mnem-http

USER mnem
WORKDIR /data

# HuggingFace cache lives inside the container so first-run model downloads
# don't hit the read-only layer. For airgapped users, bind-mount /models
# from the host and set HF_HOME=/models.
ENV HF_HOME=/models \
    RUST_LOG=info

EXPOSE 9876

# The binary resolves ./.mnem/repo.redb by default; callers override with
# `--repo /data/my-repo` or mount their own .mnem directory at /data/.mnem.
ENTRYPOINT ["/usr/local/bin/mnem-http"]
CMD ["--bind", "0.0.0.0:9876", "--repo", "/data"]
