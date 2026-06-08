import mlflow
import dagshub
from pathlib import Path
import os

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.resolve() # Use resolve() to get absolute path

def is_running_on_huggingface():
    """
    Detect if the script is running on Hugging Face Spaces
    """
    return os.environ.get('SPACE_ID') is not None or \
           os.environ.get('SPACE_AUTHOR_NAME') is not None or \
           os.environ.get('SYSTEM') == 'spaces'

def setup_mlflow():
    """
    Configure MLflow pour le projet avec SQLite
    """
    db_path = PROJECT_ROOT / "mlflow.db"
    
    # Use as_uri() for both tracking and artifacts to guarantee correct Windows formatting
    # SQLite requires a specific format, so we keep sqlite:/// but use posix for the path part
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "home_credit_default_risk"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        artifact_location = PROJECT_ROOT / "mlartifacts"
        # Let pathlib handle the file:/// generation automatically
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location.as_uri() 
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name} (ID: {experiment_id})")
    
    return experiment_id

def setup_dagshub_mlflow():
    """
    Configure MLflow pour le projet avec Dagshub
    Utilise un token pour l'authentification automatisée
    """
    # Check if DAGSHUB_TOKEN is available
    dagshub_token = os.environ.get('DAGSHUB_TOKEN')
    
    if not dagshub_token:
        print("⚠️ DAGSHUB_TOKEN not found in environment variables")
        print("Falling back to local MLflow setup")
        return setup_mlflow()
    
    try:
        # Initialize DagsHub with token authentication
        dagshub.init(
            repo_owner='obiyohan',
            repo_name='my-ml-mlflow',
            mlflow=True
        )
        
        # Set MLflow tracking URI manually with token
        mlflow.set_tracking_uri('https://dagshub.com/obiyohan/my-ml-mlflow.mlflow')
        
        # Set credentials via environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'obiyohan'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        experiment_name = "home_credit_default_risk"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"✅ Experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        print(f"❌ Error setting up DagsHub MLflow: {e}")
        print("Falling back to local MLflow setup")
        return setup_mlflow()

def setup_mlflow_auto():
    """
    Configure MLflow automatiquement selon l'environnement
    - HF Spaces ou environnement automatisé: DagsHub avec token
    - Local: SQLite
    """
    IS_HF_SPACE = is_running_on_huggingface()
    HAS_DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN') is not None
    
    if IS_HF_SPACE or HAS_DAGSHUB_TOKEN:
        print("🌍 Using DagsHub MLflow (automated environment)")
        return setup_dagshub_mlflow()
    else:
        print("💻 Using local SQLite MLflow")
        return setup_mlflow()