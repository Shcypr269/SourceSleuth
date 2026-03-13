# Dockerfile for SourceSleuth MCP Server
#
# This Dockerfile containerizes the MCP server for local sandboxing.
# It maintains stdio communication required by the Model Context Protocol.
#
# Usage:
#   docker build -t sourcesleuth:latest .
#   docker run -i --rm -v ./student_pdfs:/app/student_pdfs -v sourcesleuth_data:/app/data sourcesleuth:latest
#
# For Claude Desktop integration, see README.md for configuration.

FROM python:3.10-slim

# Set environment variables to prevent Python from buffering stdout/stderr
# This is CRITICAL for MCP stdio communication - buffering will break JSON-RPC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Define environment variables for the MCP server
ENV SOURCESLEUTH_PDF_DIR=/app/student_pdfs
ENV SOURCESLEUTH_DATA_DIR=/app/data
ENV SOURCESLEUTH_MODEL=all-MiniLM-L6-v2
ENV SOURCESLEUTH_LOG_LEVEL=INFO

WORKDIR /app

# Install system dependencies required by FAISS, PyMuPDF, and OCR
# Using slim image + build-essential keeps final size reasonable
# 
# OCR Language Support:
#   Default: English only (tesseract-ocr package includes eng training data)
#   To add languages, append: tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa
#   Example: RUN apt-get install -y ... tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
# Using --no-cache-dir reduces image size
# Installing [dev,ui,ocr] for full feature support
# Also download NLTK data required for query expansion
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,ui,ocr]" && \
    python -m nltk.downloader wordnet averaged_perceptron_tagger punkt punkt_tab omw-1.4

# Copy the application code
COPY src/ ./src/

# Create mount points for volumes
# These will be replaced by actual volumes at runtime
RUN mkdir -p /app/student_pdfs /app/data

# Create non-root user for security
# Running as non-root prevents file ownership conflicts when mounting host volumes
RUN groupadd --gid 1000 sourcesleuth && \
    useradd --uid 1000 --gid sourcesleuth --shell /bin/bash --create-home sourcesleuth && \
    chown -R sourcesleuth:sourcesleuth /app

# Switch to non-root user
# This is critical for security and prevents permission issues with mounted volumes
USER sourcesleuth:sourcesleuth

# Health check to verify the server can start
# Note: This doesn't test MCP functionality, just that Python can import modules
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.mcp_server import mcp; print('OK')" || exit 1

# Run the MCP server via standard module execution
# The server communicates over stdio (stdin/stdout), not HTTP
ENTRYPOINT ["python", "-m", "src.mcp_server"]
