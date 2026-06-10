from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging
import threading
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Default Risk API", version="1.0.0")

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

class PredictionRequest(BaseModel):
    features: Dict[str, Any]  # Dictionary of feature names and values

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    probability: float  # Probability of class 1
    threshold: float  # Threshold used for prediction

# Global variables for model and preprocessors
loaded_model = None
imputer = None
scaler = None
feature_names = None


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

def load_model_from_mlflow():
    """
    Load the latest XGBoost model from MLflow
    """
    global loaded_model, imputer, scaler, feature_names
    
    try:
        import mlflow
        import sys
        
        # Add src to path to import mlflow_config
        src_path = PROJECT_ROOT / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from mlflow_config import setup_mlflow_auto
        
        # Setup MLflow
        setup_mlflow_auto()
        
        # Get the latest model from the experiment
        experiment_name = "home_credit_default_risk"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get all runs from the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise ValueError("No runs found in the experiment")
        
        run_id = runs.iloc[0].run_id
        
        # Load the model
        model_uri = f"runs:/{run_id}/xgboost_Classifier"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Load feature names from run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Reconstruct feature names from logged batches
        feature_names = []
        i = 1
        while f"features_batch_{i}" in run.data.params:
            batch = run.data.params[f"features_batch_{i}"]
            feature_names.extend(batch.split(", "))
            i += 1
        
        # Initialize preprocessors (same as in training)
        imputer = SimpleImputer(strategy='median')
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        logger.info(f"✅ Model loaded successfully from run {run_id}")
        logger.info(f"✅ Model expects {len(feature_names)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction using the loaded XGBoost model
    """
    global loaded_model, imputer, scaler, feature_names
    
    # Load model if not already loaded
    if loaded_model is None:
        success = load_model_from_mlflow()
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model from MLflow"
            )
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Add engineered features (same as in training)
        if 'AMT_CREDIT' in input_df.columns and 'AMT_INCOME_TOTAL' in input_df.columns:
            input_df['CREDIT_INCOME_PERCENT'] = input_df['AMT_CREDIT'] / input_df['AMT_INCOME_TOTAL']
        
        if 'AMT_ANNUITY' in input_df.columns and 'AMT_INCOME_TOTAL' in input_df.columns:
            input_df['ANNUITY_INCOME_PERCENT'] = input_df['AMT_ANNUITY'] / input_df['AMT_INCOME_TOTAL']
        
        if 'AMT_ANNUITY' in input_df.columns and 'AMT_CREDIT' in input_df.columns:
            input_df['CREDIT_TERM'] = input_df['AMT_ANNUITY'] / input_df['AMT_CREDIT']
        
        if 'DAYS_EMPLOYED' in input_df.columns and 'DAYS_BIRTH' in input_df.columns:
            input_df['DAYS_EMPLOYED_PERCENT'] = input_df['DAYS_EMPLOYED'] / input_df['DAYS_BIRTH']
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            # Fill missing features with NaN (will be imputed)
            for feature in missing_features:
                input_df[feature] = np.nan
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Preprocess (impute and scale)
        input_array = imputer.transform(input_df)
        input_array = scaler.transform(input_array)
        
        # Make prediction
        y_pred_proba = loaded_model.predict_proba(input_array)[0, 1]
        
        # Get threshold from model metadata (default 0.5)
        threshold = 0.5  # You can retrieve this from MLflow run params if logged
        y_pred = int(y_pred_proba >= threshold)
        
        return PredictionResponse(
            prediction=y_pred,
            probability=float(y_pred_proba),
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    global loaded_model, feature_names
    
    if loaded_model is None:
        return {
            "status": "not_loaded",
            "message": "No model loaded yet"
        }
    
    return {
        "status": "loaded",
        "model_type": type(loaded_model).__name__,
        "num_features": len(feature_names) if feature_names else 0,
        "features": feature_names[:10] if feature_names else []  # First 10 features
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)