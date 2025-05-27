"""MLflow logging utilities."""

import logging
import mlflow
import git
from pathlib import Path
from typing import Optional


def setup_mlflow_logging(
    experiment_name: str, tracking_uri: str = "http://127.0.0.1:8080"
) -> None:
    """Setup MLflow logging with git commit tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Log git commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.object.hexsha
        mlflow.log_param("git_commit_id", commit_id)
    except Exception as e:
        logging.warning(f"Could not log git commit ID: {e}")


class MLFlowLogger:
    """Custom logger that integrates with MLflow."""

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self.logger = logging.getLogger(__name__)

    def log_param(self, param: str, value: any) -> None:
        """Log parameter to MLflow and console."""
        mlflow.log_param(param, value)
        if self.logger.isEnabledFor(self.level):
            self.logger.info(f"PARAM - {param}: {str(value)}")

    def log_metric(self, metric: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to MLflow and console."""
        mlflow.log_metric(metric, value, step)
        if self.logger.isEnabledFor(self.level):
            self.logger.info(f"METRIC - {metric}: {value}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(f"ARTIFACT - {local_path}")
