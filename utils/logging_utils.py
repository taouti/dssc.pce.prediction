from pathlib import Path
from datetime import datetime
import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ExecutionLogger:
    """Handles logging of model execution, configs, and results."""

    def __init__(self, dft: str, base_log_dir: str = "log"):
        """
        Initialize the execution logger.

        Args:
            dft: DFT method used
            base_log_dir: Base directory for logs
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

        # Create subdirectories
        (self.execution_dir / "plots").mkdir(exist_ok=True)
        (self.execution_dir / "models").mkdir(exist_ok=True)
        (self.execution_dir / "results").mkdir(exist_ok=True)

    def _get_millisecond_timestamp(self) -> str:
        """Generate millisecond timestamp for file names."""
        return str(int(time.time() * 1000))

    def _numpy_json_handler(self, obj):
        """Handle NumPy types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    def save_model(self, model, prefix: str = "model") -> Path:
        """Save model with timestamp."""
        timestamp = self._get_millisecond_timestamp()
        model_path = self.execution_dir / "models" / f"{prefix}-{timestamp}.joblib"
        joblib.dump(model, model_path)
        return model_path

    def save_plot(self, fig: plt.Figure, name: str) -> Path:
        """
        Save a matplotlib figure.

        Args:
            fig: matplotlib Figure object
            name: Base name for the plot file

        Returns:
            Path where the plot was saved
        """
        timestamp = self._get_millisecond_timestamp()
        plot_path = self.execution_dir / "plots" / f"{name}_{timestamp}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        return plot_path

    def save_results_file(self, results_file: str) -> Path:
        """Save results file to the results directory."""
        try:
            results_path = Path(results_file)
            if not results_path.exists():
                raise FileNotFoundError(f"Results file '{results_file}' not found.")

            new_results_path = self.execution_dir / "results" / results_path.name
            results_path.rename(new_results_path)

            return new_results_path
        except Exception as e:
            raise RuntimeError(f"Error saving results file: {e}")

    def save_execution_info(self, config: dict, metrics: dict, results: dict) -> Path:
        """Save execution information."""
        timestamp = self._get_millisecond_timestamp()
        info_path = self.execution_dir / f"execution_info_{timestamp}.txt"

        with open(info_path, 'w') as f:
            f.write("=== Execution Information ===\n")
            f.write(f"Timestamp: {self.timestamp.strftime('%Y.%m.%d_%H-%M-%S')}\n\n")

            f.write("=== Dataset Information ===\n")
            for model_name, model_results in results.items():
                f.write(f"\n{model_name} Model:\n")
                f.write(f"Total samples: {len(model_results)}\n")
                f.write(f"Training samples: {len(model_results[model_results['Dataset'] == 'Training'])}\n")
                f.write(f"Testing samples: {len(model_results[model_results['Dataset'] == 'Testing'])}\n")

            f.write("\n=== Configuration ===\n")
            f.write(json.dumps(config, indent=2, default=self._numpy_json_handler))

            f.write("\n\n=== Performance Metrics ===\n")
            for model_name, model_metrics in metrics.items():
                f.write(f"\n{model_name} Model:\n")
                f.write(json.dumps(model_metrics, indent=2, default=self._numpy_json_handler))
                f.write("\n")

        return info_path