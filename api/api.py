from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from evidently import Report
from evidently.presets import DataDriftPreset
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging
import threading
import pandas as pd
import numpy as np
import os
import mlflow
import joblib
import tempfile
import json

# Ajouter src/ au PYTHONPATH dès le début
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mlflow_config import setup_mlflow_auto

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        return json.dumps(log_data)

app = FastAPI(title="Home Credit Default Risk API", version="1.0.0")

# Configuration des chemins
SCRIPT_PATH = PROJECT_ROOT / "src" / "run_xgb_classifier.py"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure JSON logging
handler = logging.FileHandler(LOGS_DIR / "api_structured.log")
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)


# Stockage des statuts d'exécution
training_status = {}

# ✅ Variables globales avec noms cohérents
model = None
imputer = None
scaler = None
feature_names = None
model_threshold = 0.5  # Valeur par défaut

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
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float

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
            log_f.write(f"Python executable: {sys.executable}\n")
            log_f.write("=" * 50 + "\n\n")
            log_f.flush()
            
            # Créer un environnement avec UTF-8 forcé
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Utiliser le même Python que celui qui exécute l'API
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH)],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT / "src"),
                env=env  # Passer l'environnement modifié
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
                error_msg = ''.join(lines[-20:])
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
    # Vérifier si un entraînement est déjà en cours
    for job_id, status in training_status.items():
        if status["status"] in ["pending", "running"]:
            raise HTTPException(
                status_code=409,  # Conflict
                detail={
                    "message": "Training already in progress",
                    "current_job_id": job_id,
                    "status": status["status"],
                    "started_at": status["started_at"]
                }
            )
    
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

def load_model_from_mlflow(experiment_name = "home_credit_risk_training"):
    """
    Charge le modèle et les preprocessors depuis MLflow
    """
    global model, imputer, scaler, feature_names, model_threshold
    
    try:
        logger.info("⏳ Loading model from MLflow...")
        
        setup_mlflow_auto(experiment_name)
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No runs found in experiment")
        
        run_id = runs[0].info.run_id
        logger.info(f"📦 Loading model from run: {run_id}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Charger les preprocessors
                preprocessors_dir = client.download_artifacts(
                    run_id, 
                    "preprocessors",
                    tmpdir
                )
                
                imputer = joblib.load(Path(preprocessors_dir) / "imputer.pkl")
                scaler = joblib.load(Path(preprocessors_dir) / "scaler.pkl")
                feature_names = joblib.load(Path(preprocessors_dir) / "feature_names.pkl")
                
                logger.info("✅ Preprocessors loaded successfully")
                
                # Charger les données de référence
                data_dir = client.download_artifacts(
                    run_id, 
                    "data",
                    tmpdir
                )
                reference_csv_path = Path(data_dir) / "reference_data.csv"
                
                # Lire le CSV
                reference_data = pd.read_csv(reference_csv_path)
                logger.info(f"✅ Reference data loaded: {reference_data.shape}")
                
            except Exception as e:
                logger.error(f"❌ Error loading artifacts: {e}")
                raise
        
        # Charger le modèle
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Charger le threshold
        threshold_param = runs[0].data.params.get('threshold_value', '0.5')
        model_threshold = float(threshold_param)
        
        logger.info(f"✅ Model loaded successfully (threshold: {model_threshold})")
        return True, model_uri, reference_data
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction using the loaded XGBoost model
    """
    global model, imputer, scaler, feature_names, model_threshold
    
    # Variable pour stocker reference_data
    reference_data = None
    model_uri = None
    
    # ✅ Charger le modèle si nécessaire
    if model is None:
        logger.info("⏳ Model not loaded, loading now...")
        success, model_uri, reference_data = load_model_from_mlflow()
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
        missing_features = [f for f in feature_names if f not in input_df.columns]
        
        if missing_features:
            logger.info(f"Adding {len(missing_features)} missing features")
            missing_df = pd.DataFrame(
                np.nan, 
                index=input_df.index, 
                columns=missing_features
            )
            input_df = pd.concat([input_df, missing_df], axis=1)
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Apply preprocessing
        input_array = imputer.transform(input_df)
        input_array = scaler.transform(input_array)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_array)[0, 1]
        prediction = int(prediction_proba >= model_threshold)

        # Log prediction metrics to MLflow
        setup_mlflow_auto("home_credit_risk_inference")
        run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


        with mlflow.start_run(run_name=run_name):
            # Link to training model
            mlflow.set_tag("baseline_model_uri", model_uri)
            
            # Log only inference-specific data
            mlflow.log_metric("prediction_probability", prediction_proba)
            mlflow.log_metric("prediction_class", prediction)
            
            # Drift detection avec Evidently (seulement si reference_data est disponible)
            if reference_data is not None :
                # Filter out columns that are empty in current data
                non_empty_cols = input_df.columns[input_df.notna().any()].tolist()
                
                # Keep only common columns between reference and current that are non-empty
                common_cols = [col for col in reference_data.columns if col in non_empty_cols]
                
                if len(common_cols) > 0:
                    drift_report = Report(metrics=[DataDriftPreset()])
                    mon_evaluation = drift_report.run(
                        reference_data=reference_data[common_cols], 
                        current_data=input_df[common_cols]
                    )
                    
                    # Save report as HTML
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                        temp_path = tmp.name

                    mon_evaluation.save_html(temp_path)
                    
                    # Log drift report to MLflow
                    mlflow.log_artifact(temp_path, "drift_reports")
                    os.unlink(temp_path)
                    
                    logger.info(f"✅ Drift report logged ({len(common_cols)} features)")
                else:
                    logger.warning("⚠️ No common non-empty columns for drift detection")
        
        logger.info(f"✅ Prediction: {prediction} (proba: {prediction_proba:.4f}, threshold: {model_threshold})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(prediction_proba),
            threshold=float(model_threshold)
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
    global model, feature_names, model_threshold
    
    if model is None:
        return {
            "status": "not_loaded",
            "message": "No model loaded yet"
        }
    
    return {
        "status": "loaded",
        "model_type": type(model).__name__,
        "num_features": len(feature_names) if feature_names else 0,
        "threshold": float(model_threshold),
        "features_sample": feature_names[:10] if feature_names else []
    }

@app.post("/model/load")
async def load_model():
    """
    Load the model from MLflow manually
    """
    global model, feature_names
    
    if model is not None:
        return {
            "status": "already_loaded",
            "message": "Model is already loaded",
            "num_features": len(feature_names) if feature_names else 0
        }
    
    logger.info("⏳ Loading model from MLflow...")
    success = load_model_from_mlflow()
    
    if success:
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "num_features": len(feature_names) if feature_names else 0,
            "model_type": type(model).__name__
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to load model from MLflow"
        )

@app.get("/model/template")
async def get_prediction_template():
    """
    Get a JSON template with all required features
    """
    global model, feature_names
    
    if model is None:
        logger.info("⏳ Model not loaded, loading now...")
        success = load_model_from_mlflow()
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model from MLflow"
            )
    
    if feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Feature names not available"
        )
    
    template = {feature: 0 for feature in feature_names}
    
    return {
        "features": template,
        "num_features": len(feature_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)