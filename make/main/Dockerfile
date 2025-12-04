FROM python:3.10-slim
ARG MIRROR
ARG APT_MIRROR
LABEL authors="nexent"

# Set correct permissions as root
USER root
RUN umask 0022

# Configure apt sources based on build argument
RUN if [ "$APT_MIRROR" = "tsinghua" ]; then \
        rm -f /etc/apt/sources.list.d/* && \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install system dependencies required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv $(test -n "$MIRROR" && echo "-i $MIRROR")
WORKDIR /opt/backend

# 提前设置虚拟环境变量（关键：确保后续命令使用虚拟环境）
ENV VIRTUAL_ENV=/opt/backend/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch

# Layer 0: install base dependencies（解决索引策略和依赖冲突）
COPY backend/pyproject.toml /opt/backend/pyproject.toml
RUN uv sync --no-cache-dir $(test -n "$MIRROR" && echo "-i $MIRROR") \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match && \
    uv cache clean

# Layer 1: install sdk in link mode
COPY sdk /opt/sdk
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/sdk $(test -n "$MIRROR" && echo "-i $MIRROR")

# Layer 2: copy backend code
COPY backend /opt/backend

# Copy static files
COPY baidu_stopwords.txt /opt/baidu_stopwords.txt

# Create necessary directories with proper permissions
RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && mkdir -p /opt/models \
    && mkdir -p /opt/ssh-keys \
    && chmod -R 755 /root/.cache/torch \
    && chmod -R 755 /opt/models \
    && chmod -R 755 /opt/ssh-keys

# Create placeholder model file (if needed for code compatibility)
RUN touch /opt/models/placeholder_model.pth && \
    echo "Placeholder for pathology model (using official ResNet18)" > /opt/models/model_info.txt

# Volume for SSH keys
VOLUME ["/opt/ssh-keys"]

# Set working directory
WORKDIR /opt

# Expose service port
EXPOSE 5010