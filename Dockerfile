# trunk-ignore-all(checkov/CKV_DOCKER_3)
# trunk-ignore-all(checkov/CKV_DOCKER_2)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set workdir
WORKDIR /app

# Copy files
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry to not use virtualenvs
RUN poetry config virtualenvs.create false

# Fix lock file inside container if needed
RUN poetry lock --no-update && poetry install --no-root

# Copy source code
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run your app (adjust the command as needed)
CMD ["poetry", "run", "python", "src/servers/main.py"]