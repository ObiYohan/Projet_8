import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_train_model_success():
    """Test starting a training job successfully"""
    response = client.post("/train")
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert "started_at" in data
    assert "log_file" in data
    assert data["message"] == "Training started successfully"
    assert data["job_id"].startswith("train_")

def test_run_training_script_success(tmp_path, monkeypatch):
    """Test that training_status is updated to 'completed' when subprocess returns exit code 0"""
    from api.api import run_training_script, training_status, LOGS_DIR
    from unittest.mock import Mock, patch, mock_open
    import subprocess
    
    # Setup
    job_id = "test_job_123"
    monkeypatch.setattr('api.api.LOGS_DIR', tmp_path)
    
    # Initialize training_status for this job
    training_status[job_id] = {
        "status": "pending",
        "started_at": "2024-01-01T00:00:00",
        "completed_at": None,
        "error": None,
        "log_file": str(tmp_path / f"{job_id}.log")
    }
    
    # Mock subprocess.run to return exit code 0 (success)
    mock_result = Mock()
    mock_result.returncode = 0
    
    with patch('subprocess.run', return_value=mock_result):
        run_training_script(job_id)
    
    # Assertions
    assert training_status[job_id]["status"] == "completed"
    assert training_status[job_id]["completed_at"] is not None
    assert training_status[job_id]["error"] is None

def test_run_training_script_failure(tmp_path, monkeypatch):
    """Test that training_status is updated to 'failed' when subprocess returns non-zero exit code"""
    from api.api import run_training_script, training_status, LOGS_DIR
    from unittest.mock import Mock, patch
    import subprocess
    
    # Setup
    job_id = "test_job_failed_123"
    monkeypatch.setattr('api.api.LOGS_DIR', tmp_path)
    
    # Initialize training_status for this job
    training_status[job_id] = {
        "status": "pending",
        "started_at": "2024-01-01T00:00:00",
        "completed_at": None,
        "error": None,
        "log_file": str(tmp_path / f"{job_id}.log")
    }
    
    # Mock subprocess.run to return non-zero exit code (failure)
    mock_result = Mock()
    mock_result.returncode = 1
    
    with patch('subprocess.run', return_value=mock_result):
        run_training_script(job_id)
    
    # Assertions
    assert training_status[job_id]["status"] == "failed"
    assert training_status[job_id]["completed_at"] is not None
    assert training_status[job_id]["error"] is not None

def test_run_training_script_exception_logged(tmp_path, monkeypatch):
    """Test that exception details are appended to log file when an unexpected exception occurs"""
    from api.api import run_training_script, training_status
    from unittest.mock import patch
    
    # Setup
    job_id = "test_job_exception_123"
    monkeypatch.setattr('api.api.LOGS_DIR', tmp_path)
    
    # Initialize training_status for this job
    training_status[job_id] = {
        "status": "pending",
        "started_at": "2024-01-01T00:00:00",
        "completed_at": None,
        "error": None,
        "log_file": str(tmp_path / f"{job_id}.log")
    }
    
    # Mock subprocess.run to raise an exception
    exception_message = "Unexpected training error"
    with patch('subprocess.run', side_effect=Exception(exception_message)):
        run_training_script(job_id)
    
    # Assertions
    assert training_status[job_id]["status"] == "failed"
    assert training_status[job_id]["completed_at"] is not None
    assert training_status[job_id]["error"] == exception_message
    
    # Verify exception was written to log file
    log_file = tmp_path / f"{job_id}.log"
    assert log_file.exists()
    log_content = log_file.read_text(encoding='utf-8')
    assert "=== EXCEPTION ===" in log_content
    assert exception_message in log_content

def test_get_training_logs_job_not_found():
    """Test that getting logs returns 404 when job_id does not exist"""
    response = client.get("/logs/nonexistent_job_id")
    assert response.status_code == 404
    assert "Job ID not found" in response.json()["detail"]