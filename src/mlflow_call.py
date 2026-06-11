import mlflow
import mlflow.sklearn
from mlflow_config import setup_mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# Configuration MLflow
setup_mlflow()

def call_mlflow_start_run(app_train, 
                          scores_dict,  
                          model,
                          cm,
                          model_name="classification_model",
                          run_name=None,
                          description="My model run",
                          model_params=None
                          ):
    
    """
    Log model and metrics to MLflow (compatible with any classification model).
    
    Args:
        app_train: Training dataframe
        scores_dict: Dictionary of cross-validation scores
        model: Trained model (any scikit-learn compatible classifier)
        cm: Confusion matrix
        model_name: Name for the logged model artifact (default: "classification_model")
        run_name: Name of the MLflow run (default: auto-generated from model type)
        description: Description of the run
        model_params: Optional dict of model parameters to log
    """

    # Séparer les métriques numériques des métadonnées textuelles
    numeric_metrics = {}
    text_params = {}
    
    for key, value in scores_dict.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_metrics[key] = float(value)
        elif isinstance(value, str):
            text_params[key] = value
        # Ignorer les autres types (listes, dicts, etc.)
    
    # Logger les métriques numériques
    mlflow.log_metrics(numeric_metrics)
    
    # Logger les paramètres textuels
    if text_params:
        mlflow.log_params(text_params)
    
    # Logger les paramètres du modèle si fournis
    if model_params:
        mlflow.log_params(model_params)
    
    # # Change run name if not provided
    # if run_name is None:
    #     mlflow.set_tag("mlflow.runName", run_name)

    # # Change description if not provided
    # if description is None:
    #     mlflow.set_tag("mlflow.runDescription", description)
    #     mlflow.set_tag("mlflow.note.content", description)


    train_labels = app_train['TARGET']

    # Drop the target from the training data
    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()
        
    # Feature names
    features = list(train.columns)
      
    # Log model type
    mlflow.log_param("model_type", type(model).__name__)

    # Log features in batches instead of all at once
    log_features_in_batches(features, batch_size=100)
    
    # Log model hyperparameters
    if model_params is None:
        model_params = model.get_params()
    
    for param_name, param_value in model_params.items():
        # Convert non-serializable values to strings
        if isinstance(param_value, (int, float, str, bool, type(None))):
            mlflow.log_param(param_name, param_value)
        else:
            mlflow.log_param(param_name, str(param_value))

    # Determine serialization format based on model type
    model_type_name = type(model).__name__
    
    # Use pickle for XGBoost and LightGBM (they have their own safe serialization)
    # Use skops for pure scikit-learn models
    if 'XGB' in model_type_name or 'LGBM' in model_type_name or 'LightGBM' in model_type_name:
        mlflow.sklearn.log_model(
            model, 
            artifact_path=model_name,
            serialization_format="pickle"
        )
    else:
        mlflow.sklearn.log_model(
            model, 
            artifact_path=model_name,
            serialization_format="skops",
            skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
        )
    
    
    # Create and log confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {type(model).__name__}')
    
    mlflow.log_figure(disp.figure_, "confusion_matrix.png")
    plt.close()

    # Log datasets
    train_dataset = mlflow.data.from_pandas(
        app_train,
        source="application_train_processed.csv",
        name="training_data"
    )
    mlflow.log_input(train_dataset, context="training")
        
def log_features_in_batches(features, batch_size=100):
    """
    Log features in batches to avoid MLflow's 6000 character limit.
    
    Args:
        features: List of feature names
        batch_size: Number of features per batch (default: 100)
    """
    total_features = len(features)
    num_batches = (total_features + batch_size - 1) // batch_size
    
    # Log total number of features
    mlflow.log_param("total_features", total_features)
    mlflow.log_param("num_feature_batches", num_batches)
    
    # Log features in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_features)
        batch_features = features[start_idx:end_idx]
        
        mlflow.log_param(f"features_batch_{i+1}", batch_features)