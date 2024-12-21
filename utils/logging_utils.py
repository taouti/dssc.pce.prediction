from pathlib import Path
from datetime import datetime
import json
import time
import joblib
import pandas as pd

class ExecutionLogger:
    """Handles logging of model execution, configs, and results."""

    def __init__(self, dft:str, base_log_dir: str = "log"):
        """
        Initialize the execution logger.

        Args:
            base_log_dir (str): Base directory for logs
        """
        self.base_log_dir = Path(base_log_dir)
        self.timestamp = datetime.now()
        self.execution_dir = None
        self.setup_directories(dft)

    def setup_directories(self, dft: str) -> None:
        """Create necessary directory structure for logging."""
        # Create base log directory if it doesn't exist
        self.base_log_dir.mkdir(exist_ok=True)

        # Create execution-specific directory
        execution_dirname = f"ex_{self.timestamp.strftime('%d.%m.%y_%H-%M')}_{dft}"
        self.execution_dir = self.base_log_dir / execution_dirname
        self.execution_dir.mkdir(exist_ok=True)

    def _get_millisecond_timestamp(self) -> str:
        """Generate millisecond timestamp for file names."""
        return str(int(time.time() * 1000))

    def save_model(self, model, prefix: str = "model") -> Path:
        """
        Save model with timestamp.

        Args:
            model: Trained model to save
            prefix (str): Prefix for the model file

        Returns:
            Path: Path where model was saved
        """
        timestamp = self._get_millisecond_timestamp()
        model_path = self.execution_dir / f"{prefix}-{timestamp}.joblib"
        joblib.dump(model, model_path)
        return model_path

    def save_execution_info(self, config: dict, metrics: dict, results: pd.DataFrame, prefix: str = "notion") -> Path:
        """
        Save configuration and results.

        Args:
            config (dict): Configuration used for the execution
            metrics (dict): Performance metrics
            prefix (str): Prefix for the info file

        Returns:
            Path: Path where info was saved
            :param prefix:
            :param config:
            :param metrics:
            :param results:
        """
        timestamp = self._get_millisecond_timestamp()
        info_path = self.execution_dir / f"{prefix}-{timestamp}.txt"

        with open(info_path, 'w') as f:
            f.write("=== Execution Information ===\n")
            f.write(f"Timestamp: {self.timestamp.strftime('%Y.%m.%d_%H-%M-%S')}\n\n")
            f.write(f"Total number of compounds: {len(results)}\n")
            f.write(f"Training set size: {len(results[results['Dataset'] == 'Training'])}\n")
            f.write(f"Test set size: {len(results[results['Dataset'] == 'Testing'])}\n\n")

            f.write("=== Configuration ===\n")
            f.write(json.dumps(config, indent=2))
            f.write("\n\n")

            f.write("=== Performance Metrics ===\n")
            f.write(json.dumps(metrics, indent=2))

        return info_path

    def save_results_file(self, results_file: str) -> Path:
        """
        Save the results file to the log directory.

        Args:
            results_file (str): Path to the results file to be saved.

        Returns:
            Path: Path where the results file was saved.
        """
        try:
            # Ensure the results file exists
            results_path = Path(results_file)
            if not results_path.exists():
                raise FileNotFoundError(f"Results file '{results_file}' not found.")

            # Save the file in the execution directory
            new_results_path = self.execution_dir / results_path.name
            results_path.rename(new_results_path)

            return new_results_path
        except Exception as e:
            raise RuntimeError(f"Error saving results file: {e}")
