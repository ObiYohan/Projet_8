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

# Build argument to control data download
ARG DOWNLOAD_DATA=true
ARG HF_TOKEN

# Download data files from Hugging Face Space storage (only in production)
RUN if [ "$DOWNLOAD_DATA" = "true" ]; then \
        if [ -z "$HF_TOKEN" ]; then \
            echo "⚠️ HF_TOKEN is empty, skipping download"; \
        else \
            . /opt/venv/bin/activate && \
            HF_TOKEN="$HF_TOKEN" python -c "from huggingface_hub import hf_hub_download; import os; \
files = ['application_train_processed.csv', 'application_test_processed.csv']; \
[hf_hub_download(repo_id='0biyohan/Projet_8', filename=f'data/{file}', repo_type='space', local_dir='/app', token=os.environ.get('HF_TOKEN')) or print(f'✅ Downloaded {file}') for file in files]"; \
        fi; \
    else \
        echo "⏭️ Skipping data download (local mode)"; \
    fi

# Expose ports
EXPOSE 8000 7860

# Variables d'environnement
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run both services
CMD uvicorn api.api:app --host 0.0.0.0 --port 8000 & python gradio_app/app.py