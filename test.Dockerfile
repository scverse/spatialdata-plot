# Dockerfile for Testing Plot Correctness
#
# This Dockerfile is used to create a consistent testing environment for
# verifying the correctness of plot outputs against ground truth images.
# The ground truth images are stored in `tests/_images` and are generated
# using a GitHub Action on Ubuntu. This ensures consistency across different
# environments, as local differences in OS or library versions can cause
# slight variations in plot rendering (e.g., ticks or padding).

# Define a build argument for the target platform.
# Default is set to linux/amd64 for x86_64 machines.
ARG TARGETPLATFORM=linux/amd64

# Use the specified platform to pull the correct base image.
# Override TARGETPLATFORM during build for different architectures, such as linux/arm64 for Apple Silicon.
# For example, to build for ARM64 architecture (e.g., Apple Silicon),
# use the following command on the command line:
#
#     docker build --build-arg TARGETPLATFORM=linux/arm64 -t my-arm-image .
#
# Similarly, to build for the default x86_64 architecture, you can use:
#
#     docker build --build-arg TARGETPLATFORM=linux/amd64 -t my-amd64-image .
#
FROM --platform=$TARGETPLATFORM ubuntu:latest
LABEL authors="Luca Marconato"

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=agg

WORKDIR /spatialdata-plot
COPY . /spatialdata-plot

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

RUN pip install -e ".[dev,test]"

CMD ["pytest", "-v", "tests/pl/test_render.py"]
