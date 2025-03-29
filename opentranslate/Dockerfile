# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Copy project files
COPY . .

# Install npm dependencies
RUN npm install

# Compile smart contracts
RUN npm run compile

# Expose ports
EXPOSE 8000 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    uvicorn opentranslate.api.app:app --host 0.0.0.0 --port 8000 --reload\n\
elif [ "$1" = "web" ]; then\n\
    streamlit run opentranslate/web/app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "worker" ]; then\n\
    celery -A opentranslate.worker worker --loglevel=info\n\
elif [ "$1" = "flower" ]; then\n\
    celery -A opentranslate.worker flower --port=5555\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["api"] 