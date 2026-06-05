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
        . /opt/venv/bin/activate && \
        python - <<'PY'
from huggingface_hub import hf_hub_download
import os

files = [
    'application_train_processed.csv',
    'application_test_processed.csv',
]

for file in files:
    try:
        hf_hub_download(
            repo_id='0biyohan/Projet_8',
            filename=f'data/{file}',
            repo_type='space',
            local_dir='/app',
            local_dir_use_symlinks=False,
            token=os.environ.get('HF_TOKEN')
        )
        print(f'✅ Downloaded {file}')
    except Exception as e:
        print(f'⚠️ Skip {file}: {e}')

    else \
        echo "⏭️ Skipping data download (local mode)"; \
    fi
PY

# Copy local data if available (for local development). Use a shell copy so build
# won't fail if the local data/ directory is absent.
RUN mkdir -p ./data && cp -r data/* ./data/ 2>/dev/null || true

# Expose ports
EXPOSE 8000 7860

# Variables d'environnement
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Run both services
CMD ["./start.sh"]