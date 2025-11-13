FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv - the fast Python package installer
# Download the latest installer
ADD https://astral.sh/uv/0.7.19/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create virtual environment using uv (much faster than python -m venv)
RUN uv venv /opt/venv

# Set PATH to include virtual environment
ENV PATH="/opt/venv/bin:$PATH"


# Install CPU-only PyTorch first to avoid CUDA dependencies which significantly slow down the docker build process
RUN uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# Install Python dependencies using uv (significantly faster than pip)
RUN uv pip install --no-cache -r requirements.txt
RUN uv pip install 'protobuf<=3.20.1' --force-reinstall

# Debug: Check what's installed and where
RUN echo "=== Virtual environment contents ===" && \
    ls -la /opt/venv/bin/ && \
    echo "=== Checking streamlit installation ===" && \
    uv pip list | grep streamlit && \
    echo "=== Python path ===" && \
    python -c "import sys; print(sys.path)" && \
    echo "=== Trying to find streamlit ===" && \
    find /opt/venv -name "*streamlit*" -type f

# Expose port for Streamlit -- Keeping for documentation purposes
EXPOSE 8501

# Show protobuf version for debugging
RUN uv pip show protobuf

# Command to run the application
CMD ["/bin/bash", "-c", "echo 'Starting application...' && ls -la /opt/venv/bin/ && python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501"]