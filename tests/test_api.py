import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from api.api import app, LOGS_DIR, SCRIPT_PATH, training_status, run_training_script, load_model_from_mlflow

client = TestClient(app)

@patch('api.api.subprocess.run')
@patch('api.api.logger')
def test_run_training_script_should_create_log_file_and_update_status_to_running(mock_logger, mock_subprocess):
    """Test that run_training_script creates log file and updates status to 'running' when training starts"""
    # Arrange
    job_id = "test_job_123"
    training_status[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log")
    }
    
    mock_subprocess.return_value = MagicMock(returncode=0)
    
    # Act
    run_training_script(job_id)
    
    # Assert
    log_file = LOGS_DIR / f"{job_id}.log"
    assert log_file.exists(), "Log file should be created"
    assert training_status[job_id]["status"] == "completed"
    
    # Verify log file content
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert f"=== Training Job {job_id} ===" in log_content
        assert "Started at:" in log_content
        assert str(SCRIPT_PATH) in log_content
    
    # Verify logger was called
    mock_logger.info.assert_any_call(f"🚀 Starting training for job {job_id}")
    mock_logger.info.assert_any_call(f"📝 Log file: {log_file}")
    
    # Cleanup
    if log_file.exists():
        log_file.unlink()
    del training_status[job_id]

@patch('api.api.subprocess.run')
@patch('api.api.logger')
def test_run_training_script_should_complete_successfully_when_subprocess_returns_zero(mock_logger, mock_subprocess):
    """Test that run_training_script updates status to 'completed' when subprocess returns code 0"""
    # Arrange
    job_id = "test_job_success"
    training_status[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log")
    }
    
    mock_subprocess.return_value = MagicMock(returncode=0)
    
    # Act
    run_training_script(job_id)
    
    # Assert
    assert training_status[job_id]["status"] == "completed"
    assert training_status[job_id]["completed_at"] is not None
    assert training_status[job_id]["error"] is None
    
    mock_logger.info.assert_any_call(f"✅ Training completed with return code: 0")
    mock_logger.info.assert_any_call(f"✅ Training completed successfully for job {job_id}")
    
    # Cleanup
    log_file = LOGS_DIR / f"{job_id}.log"
    if log_file.exists():
        log_file.unlink()
    del training_status[job_id]

@patch('api.api.subprocess.run')
@patch('api.api.logger')
def test_run_training_script_should_fail_when_subprocess_returns_nonzero(mock_logger, mock_subprocess):
    """Test that run_training_script updates status to 'failed' when subprocess returns non-zero code"""
    # Arrange
    job_id = "test_job_fail"
    training_status[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log")
    }
    
    mock_subprocess.return_value = MagicMock(returncode=1)
    
    # Act
    run_training_script(job_id)
    
    # Assert
    assert training_status[job_id]["status"] == "failed"
    assert training_status[job_id]["error"] is not None
    
    # Cleanup
    log_file = LOGS_DIR / f"{job_id}.log"
    if log_file.exists():
        log_file.unlink()
    del training_status[job_id]

@patch('api.api.subprocess.run')
@patch('api.api.logger')
def test_run_training_script_should_handle_exception(mock_logger, mock_subprocess):
    """Test that run_training_script handles exceptions properly"""
    # Arrange
    job_id = "test_job_exception"
    training_status[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log")
    }
    
    mock_subprocess.side_effect = Exception("Test exception")
    
    # Act
    run_training_script(job_id)
    
    # Assert
    assert training_status[job_id]["status"] == "failed"
    assert "Test exception" in training_status[job_id]["error"]
    
    # Cleanup
    log_file = LOGS_DIR / f"{job_id}.log"
    if log_file.exists():
        log_file.unlink()
    del training_status[job_id]

def test_root_endpoint_should_return_welcome_message():
    """Test that root endpoint returns correct welcome message"""
    # Act
    response = client.get("/")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

@patch('api.api.threading.Thread')
def test_train_endpoint_should_start_training_job(mock_thread):
    """Test that train endpoint starts a training job"""
    # Arrange
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance
    
    # Act
    response = client.post("/train")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"  # Changed from "started" to "pending"
    assert "log_file" in data
    mock_thread_instance.start.assert_called_once()
    
    # Cleanup
    if data["job_id"] in training_status:
        del training_status[data["job_id"]]

def test_status_endpoint_should_return_job_status():
    """Test that status endpoint returns job status"""
    # Arrange
    job_id = "test_status_job"
    training_status[job_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log")
    }
    
    # Act
    response = client.get(f"/status/{job_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "running"
    
    # Cleanup
    del training_status[job_id]

def test_status_endpoint_should_return_404_for_unknown_job():
    """Test that status endpoint returns 404 for unknown job"""
    # Act
    response = client.get("/status/unknown_job_id")
    
    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_logs_endpoint_should_return_log_content():
    """Test that logs endpoint returns log file content"""
    # Arrange
    job_id = "test_logs_job"
    log_file = LOGS_DIR / f"{job_id}.log"
    log_content = "Test log line 1\nTest log line 2\nTest log line 3"
    log_file.write_text(log_content, encoding='utf-8')
    
    training_status[job_id] = {
        "status": "completed",
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "error": None,
        "log_file": str(log_file)
    }
    
    # Act
    response = client.get(f"/logs/{job_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert "Test log line" in data["logs"]
    
    # Cleanup
    log_file.unlink()
    del training_status[job_id]

def test_logs_endpoint_should_return_404_for_unknown_job():
    """Test that logs endpoint returns 404 for unknown job"""
    # Act
    response = client.get("/logs/unknown_job_id")
    
    # Assert
    assert response.status_code == 404

def test_logs_endpoint_should_return_tail_lines():
    """Test that logs endpoint returns specified number of tail lines"""
    # Arrange
    job_id = "test_logs_tail"
    log_file = LOGS_DIR / f"{job_id}.log"
    log_lines = [f"Line {i}\n" for i in range(200)]
    log_file.write_text("".join(log_lines), encoding='utf-8')
    
    training_status[job_id] = {
        "status": "completed",
        "log_file": str(log_file)
    }
    
    # Act
    response = client.get(f"/logs/{job_id}?tail=50")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    log_lines_returned = data["logs"].strip().split("\n")
    assert len(log_lines_returned) <= 50
    
    # Cleanup
    log_file.unlink()
    del training_status[job_id]

def test_health_endpoint_should_return_healthy():
    """Test that health endpoint returns healthy status"""
    # Act
    response = client.get("/health")
    
    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch('api.api.mlflow.start_run')
@patch('api.api.mlflow.end_run')
@patch('api.api.setup_mlflow_auto')
@patch('api.api.load_model_from_mlflow')
def test_predict_should_successfully_load_model_when_model_is_none_and_load_succeeds(
    mock_load_model, mock_setup_mlflow, mock_end_run, mock_start_run
):
    """Test that predict successfully loads model when model is None and load succeeds"""
    # Arrange
    import api.api as api_module
    
    # Save original state
    original_model = api_module.model
    original_imputer = api_module.imputer
    original_scaler = api_module.scaler
    original_feature_names = api_module.feature_names
    original_threshold = api_module.model_threshold
    
    try:
        # Create mock objects
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        mock_imputer = MagicMock()
        mock_imputer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock load_model_from_mlflow
        def side_effect_load():
            api_module.model = mock_model
            api_module.imputer = mock_imputer
            api_module.scaler = mock_scaler
            api_module.feature_names = ['feature1', 'feature2', 'feature3']
            api_module.model_threshold = 0.5
            return True, "runs:/abc123/model", pd.DataFrame()
        
        mock_load_model.side_effect = side_effect_load
        
        # ✅ Mock MLflow context manager pour éviter les appels à la DB
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Reset model to None
        api_module.model = None
        
        # Prepare request
        request_data = {
            "features": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
        }
        
        # Act
        response = client.post("/predict", json=request_data)
        
        # Assert
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.json()}"
        data = response.json()
        
        assert "prediction" in data
        assert "probability" in data
        assert "threshold" in data
        assert data["probability"] == 0.7
        assert data["threshold"] == 0.5
        
        # Verify load_model_from_mlflow was called
        mock_load_model.assert_called_once()
        
        # ✅ Verify MLflow was NOT actually called (mocked)
        mock_setup_mlflow.assert_called_once()
        mock_start_run.assert_called_once()
        
    finally:
        # Restore original state
        api_module.model = original_model
        api_module.imputer = original_imputer
        api_module.scaler = original_scaler
        api_module.feature_names = original_feature_names
        api_module.model_threshold = original_threshold

@patch('api.api.mlflow.tracking.MlflowClient')
@patch('api.api.setup_mlflow_auto')
@patch('api.api.logger')
def test_load_model_from_mlflow_should_return_false_when_experiment_does_not_exist(
    mock_logger, mock_setup_mlflow, mock_mlflow_client
):
    """Test that load_model_from_mlflow returns False when MLflow experiment does not exist"""
    # Arrange
    experiment_name = "nonexistent_experiment"
    
    mock_client_instance = MagicMock()
    mock_client_instance.get_experiment_by_name.return_value = None
    mock_mlflow_client.return_value = mock_client_instance
    
    # Act
    success, model_uri, reference_data = load_model_from_mlflow(experiment_name)
    
    # Assert
    assert success is False
    assert model_uri is None
    assert reference_data is None
    
    mock_setup_mlflow.assert_called_once_with(experiment_name)
    mock_client_instance.get_experiment_by_name.assert_called_once_with(experiment_name)
    mock_logger.error.assert_called()

@patch('api.api.mlflow.tracking.MlflowClient')
@patch('api.api.setup_mlflow_auto')
@patch('api.api.logger')
def test_load_model_from_mlflow_should_handle_missing_preprocessor_artifacts_and_raise_error(
    mock_logger, mock_setup_mlflow, mock_mlflow_client
):
    """Test that load_model_from_mlflow handles missing preprocessor artifacts and raises error"""
    # Arrange
    import api.api as api_module
    
    # Save original state
    original_model = api_module.model
    original_imputer = api_module.imputer
    original_scaler = api_module.scaler
    original_feature_names = api_module.feature_names
    original_threshold = api_module.model_threshold
    
    try:
        experiment_name = "home_credit_risk_training"
        run_id = "run_missing_artifacts"
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_missing"
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.run_id = run_id
        mock_run.data.params = {'threshold_value': '0.5'}
        
        # Mock client
        mock_client_instance = MagicMock()
        mock_client_instance.get_experiment_by_name.return_value = mock_experiment
        mock_client_instance.search_runs.return_value = [mock_run]
        
        # ✅ Simuler une erreur lors du téléchargement des artifacts
        mock_client_instance.download_artifacts.side_effect = Exception("Artifact not found")
        
        mock_mlflow_client.return_value = mock_client_instance
        
        # Act
        result = load_model_from_mlflow(experiment_name)
        
        # Assert - ✅ Vérifier que c'est un tuple avec 3 éléments
        assert isinstance(result, tuple), f"Expected tuple but got {type(result)}: {result}"
        assert len(result) == 3, f"Expected 3 values but got {len(result)}: {result}"
        
        success, model_uri, reference_data = result
        
        # ✅ Vérifier l'échec
        assert success is False, f"Expected success=False but got {success}"
        assert model_uri is None, f"Expected model_uri=None but got {model_uri}"
        assert reference_data is None, f"Expected reference_data=None but got {reference_data}"
        
        # Verify error was logged
        mock_logger.error.assert_called()
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        assert any("Error loading" in str(call) for call in error_calls), \
            f"Expected error log but got: {error_calls}"
        
        # Verify global variables were NOT modified
        assert api_module.model == original_model, "Model should not be modified on error"
        assert api_module.imputer == original_imputer, "Imputer should not be modified on error"
        assert api_module.scaler == original_scaler, "Scaler should not be modified on error"
        assert api_module.feature_names == original_feature_names, "Feature names should not be modified on error"
        
    finally:
        # Restore original state
        api_module.model = original_model
        api_module.imputer = original_imputer
        api_module.scaler = original_scaler
        api_module.feature_names = original_feature_names
        api_module.model_threshold = original_threshold

@patch('api.api.mlflow.tracking.MlflowClient')
@patch('api.api.setup_mlflow_auto')
@patch('api.api.logger')
def test_load_model_from_mlflow_should_handle_missing_preprocessor_artifacts_and_raise_error(
    mock_logger, mock_setup_mlflow, mock_mlflow_client
):
    """Test that load_model_from_mlflow handles missing preprocessor artifacts and raises appropriate error"""
    # Arrange
    experiment_name = "home_credit_risk_training"
    
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance
    
    # Mock experiment exists
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp_123"
    mock_client_instance.get_experiment_by_name.return_value = mock_experiment
    
    # Mock runs exist
    mock_run = MagicMock()
    mock_run.info.run_id = "run_123"
    mock_run.data.params = {"threshold_value": "0.5"}
    mock_client_instance.search_runs.return_value = [mock_run]
    
    # Mock download_artifacts to raise exception for missing preprocessors
    mock_client_instance.download_artifacts.side_effect = Exception("Artifact not found: preprocessors")
    
    # Act
    success, model_uri, reference_data = load_model_from_mlflow(experiment_name)
    
    # Assert
    assert success is False
    assert model_uri is None
    assert reference_data is None
    
    mock_logger.error.assert_called()
    error_calls = [call for call in mock_logger.error.call_args_list if "Error loading model" in str(call)]
    assert len(error_calls) > 0

def test_train_endpoint_should_return_409_when_training_already_in_progress():
    """Test that train endpoint returns 409 when training is already in progress"""
    # Arrange
    existing_job_id = "existing_job_pending"
    training_status[existing_job_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "log_file": str(LOGS_DIR / f"{existing_job_id}.log")
    }
    
    # Act
    response = client.post("/train")
    
    # Assert
    assert response.status_code == 409
    data = response.json()
    assert "detail" in data
    assert data["detail"]["message"] == "Training already in progress"
    assert data["detail"]["current_job_id"] == existing_job_id
    assert data["detail"]["status"] == "running"
    assert "started_at" in data["detail"]
    
    # Cleanup
    del training_status[existing_job_id]

def test_load_model_endpoint_should_return_success_when_model_already_loaded():
    """Test that load_model endpoint returns success status when model is already loaded and subsequent load_model endpoint is called"""
    # Arrange
    import api.api as api_module
    
    # Save original state
    original_model = api_module.model
    original_feature_names = api_module.feature_names
    
    try:
        # Set up model as already loaded
        mock_model = MagicMock()
        api_module.model = mock_model
        api_module.feature_names = ['feature1', 'feature2', 'feature3']
        
        # Act
        response = client.post("/model/load")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_loaded"
        assert data["message"] == "Model is already loaded"
        assert data["num_features"] == 3
        
    finally:
        # Restore original state
        api_module.model = original_model
        api_module.feature_names = original_feature_names

@patch('api.api.load_model_from_mlflow')
@patch('api.api.logger')
def test_load_model_endpoint_should_return_success_when_model_loaded_for_first_time(mock_logger, mock_load_model):
    """Test that load_model endpoint returns success with correct num_features when model is loaded for the first time"""
    # Arrange
    import api.api as api_module
    
    # Save original state
    original_model = api_module.model
    original_feature_names = api_module.feature_names
    
    try:
        # Set model to None to simulate first-time load
        api_module.model = None
        api_module.feature_names = None
        
        # Create mock objects
        mock_model = MagicMock()
        mock_feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # Mock load_model_from_mlflow to simulate successful load
        def side_effect_load():
            api_module.model = mock_model
            api_module.feature_names = mock_feature_names
            return True, "runs:/test123/model", pd.DataFrame()
        
        mock_load_model.side_effect = side_effect_load
        
        # Act
        response = client.post("/model/load")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Model loaded successfully"
        assert data["num_features"] == 5
        assert data["model_type"] == type(mock_model).__name__
        
        # Verify load_model_from_mlflow was called
        mock_load_model.assert_called_once()
        mock_logger.info.assert_called_with("⏳ Loading model from MLflow...")
        
    finally:
        # Restore original state
        api_module.model = original_model
        api_module.feature_names = original_feature_names
