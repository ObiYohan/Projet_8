FROM python:3.13-slim

WORKDIR /app

# Installer uv
RUN pip install --no-cache-dir uv

# Copier les fichiers de configuration
COPY pyproject.toml uv.lock* ./

# Créer un environnement virtuel et installer les dépendances
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY gradio_app/ ./gradio_app/

# Create data directory
RUN mkdir -p ./data

# Expose ports
EXPOSE 8000 7860

# Variables d'environnement
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run both services
CMD uvicorn api.api:app --host 0.0.0.0 --port 8000 & python gradio_app/app.py