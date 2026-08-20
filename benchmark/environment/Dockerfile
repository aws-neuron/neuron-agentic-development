FROM python:3.11-slim

WORKDIR /app

# Install system deps for kiro-cli
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Install kiro-cli
RUN curl -fsSL https://cli.kiro.dev/install | bash

ENV PATH="/root/.local/bin:$PATH"

COPY . .
