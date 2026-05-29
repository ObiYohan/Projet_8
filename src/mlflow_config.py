import mlflow
from pathlib import Path

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.resolve() # Use resolve() to get absolute path

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