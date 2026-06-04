from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging
import threading

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="XGBoost Training API", version="1.0.0")

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPT_PATH = PROJECT_ROOT / "src" / "run_xgb_classifier.py"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Stockage des statuts d'exécution
training_status = {}


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    started_at: str
    log_file: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    log_file: Optional[str] = None


def run_training_script(job_id: str):
    """
    Execute le script d'entraînement en arrière-plan
    """
    log_file = LOGS_DIR / f"{job_id}.log"
    
    try:
        logger.info(f"🚀 Starting training for job {job_id}")
        logger.info(f"📝 Log file: {log_file}")
        training_status[job_id]["status"] = "running"
        
        # Ouvrir le fichier de log
        with open(log_file, 'w', encoding='utf-8') as log_f:
            log_f.write(f"=== Training Job {job_id} ===\n")
            log_f.write(f"Started at: {datetime.now().isoformat()}\n")
            log_f.write(f"Script: {SCRIPT_PATH}\n")
            log_f.write(f"Working directory: {PROJECT_ROOT / 'src'}\n")
            log_f.write(f"Python executable: {sys.executable}\n")  # Log du Python utilisé
            log_f.write("=" * 50 + "\n\n")
            log_f.flush()
            
            # Utiliser le même Python que celui qui exécute l'API
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH)],  # Utiliser sys.executable au lieu de "python"
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT / "src")
            )
        
        logger.info(f"✅ Training completed with return code: {result.returncode}")
        
        if result.returncode == 0:
            training_status[job_id]["status"] = "completed"
            training_status[job_id]["completed_at"] = datetime.now().isoformat()
            logger.info(f"✅ Training completed successfully for job {job_id}")
        else:
            training_status[job_id]["status"] = "failed"
            # Lire les dernières lignes du log pour l'erreur
            with open(log_file, 'r', encoding='utf-8') as log_f:
                lines = log_f.readlines()
                error_msg = ''.join(lines[-20:])  # Dernières 20 lignes
            training_status[job_id]["error"] = error_msg
            training_status[job_id]["completed_at"] = datetime.now().isoformat()
            logger.error(f"❌ Training failed for job {job_id}")
            
    except Exception as e:
        logger.exception(f"💥 Exception during training for job {job_id}")
        training_status[job_id]["status"] = "failed"
        training_status[job_id]["error"] = str(e)
        training_status[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Écrire l'exception dans le log
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n\n=== EXCEPTION ===\n")
            log_f.write(str(e))


@app.get("/")
async def root():
    """
    Endpoint racine avec informations sur l'API
    """
    return {
        "message": "XGBoost Training API",
        "endpoints": {
            "/train": "POST - Lance l'entraînement du modèle XGBoost",
            "/status/{job_id}": "GET - Vérifie le statut d'un entraînement",
            "/logs/{job_id}": "GET - Récupère les logs d'un entraînement",
            "/health": "GET - Vérifie la santé de l'API"
        }
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """
    Lance l'entraînement du modèle XGBoost en arrière-plan
    """
    # Vérifier que le script existe
    if not SCRIPT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Script not found: {SCRIPT_PATH}"
        )
    
    # Générer un ID unique pour ce job
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = str(LOGS_DIR / f"{job_id}.log")
    
    # Initialiser le statut
    training_status[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": log_file
    }
    
    logger.info(f"📋 New training job created: {job_id}")
    
    # Lancer l'entraînement dans un thread séparé
    thread = threading.Thread(
        target=run_training_script,
        args=(job_id,),
        daemon=True
    )
    thread.start()
    
    logger.info(f"🧵 Thread started for job {job_id}")
    
    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training started successfully",
        started_at=training_status[job_id]["started_at"],
        log_file=log_file
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_training_status(job_id: str):
    """
    Récupère le statut d'un entraînement
    """
    if job_id not in training_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID not found: {job_id}"
        )
    
    return StatusResponse(
        job_id=job_id,
        **training_status[job_id]
    )


@app.get("/logs/{job_id}")
async def get_training_logs(job_id: str, tail: int = 100):
    """
    Récupère les logs d'un entraînement
    
    Args:
        job_id: ID du job
        tail: Nombre de dernières lignes à retourner (default: 100)
    """
    if job_id not in training_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID not found: {job_id}"
        )
    
    log_file = Path(training_status[job_id]["log_file"])
    
    if not log_file.exists():
        return {"logs": "Log file not yet created"}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Retourner les dernières 'tail' lignes
            return {"logs": ''.join(lines[-tail:])}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading log file: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Vérifie la santé de l'API
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "script_exists": SCRIPT_PATH.exists(),
        "logs_dir": str(LOGS_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)