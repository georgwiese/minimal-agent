FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash agent

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && \
    uv sync --frozen

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose port
EXPOSE 7860

# Run the web app
CMD ["uv", "run", "python", "web_app.py"]