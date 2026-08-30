from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.model import CatsDogsCNN


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Cats vs Dogs CNN"


# =========================================================
# Already Completed Training Results
# =========================================================

PARAMS = {
    "model": "CatsDogsCNN",
    "image_size": 128,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "device": "cpu",
}


METRICS = {
    "train_loss": 0.2771437168538612,
    "train_accuracy": 0.8853885388538854,
    "validation_loss": 0.24395336513519286,
    "validation_accuracy": 0.8896,
    "best_validation_accuracy": 0.8896,
    "test_loss": 0.25943834552764894,
    "test_accuracy": 0.892,
    "test_precision": 0.8804347826086957,
    "test_recall": 0.9072,
    "test_f1": 0.8936170212765957,
}


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("MLflow Model Logging")
    print("=" * 60)

    # -----------------------------------------------------
    # Verify model exists
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"Model found: {MODEL_PATH}")
    print(
        f"Model size: "
        f"{MODEL_PATH.stat().st_size:,} bytes"
    )

    # -----------------------------------------------------
    # Configure MLflow
    # -----------------------------------------------------

    if not MLFLOW_DB.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {MLFLOW_DB}"
        )

    # IMPORTANT:
    # Use SQLite tracking backend.
    # Do NOT use MLFLOW_DB.as_uri() because that creates
    # a file:// URI instead of a SQLite tracking URI.

    tracking_uri = (
        "sqlite:///"
        + MLFLOW_DB.as_posix()
    )

    mlflow.set_tracking_uri(tracking_uri)

    print(
        f"MLflow tracking URI: {tracking_uri}"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow experiment: {EXPERIMENT_NAME}"
    )

    # -----------------------------------------------------
    # Create model architecture
    # -----------------------------------------------------

    model = CatsDogsCNN(
        num_classes=2
    )

    # -----------------------------------------------------
    # Load already-trained model
    # -----------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print(
                "Loaded model_state_dict from checkpoint."
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

            print(
                "Loaded state_dict from checkpoint."
            )

        else:

            # Assume the dictionary itself is a state_dict
            model.load_state_dict(
                checkpoint
            )

            print(
                "Loaded checkpoint as state_dict."
            )

    else:

        # In case the complete model was saved
        model = checkpoint

        print(
            "Loaded complete PyTorch model."
        )

    model.eval()

    print("Model loaded successfully.")
    print("Training will NOT be repeated.")

    # -----------------------------------------------------
    # Create MLflow input example
    # -----------------------------------------------------

    # Must match the model input:
    # Batch size = 1
    # RGB channels = 3
    # Image size = 128 x 128

    example_input = torch.randn(
        1,
        3,
        128,
        128
    )

    # -----------------------------------------------------
    # Start new MLflow run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="cats-dogs-cnn-recovered"
    ) as run:

        print()
        print(
            f"MLflow Run ID: {run.info.run_id}"
        )

        # -------------------------------------------------
        # Log parameters
        # -------------------------------------------------

        mlflow.log_params(
            PARAMS
        )

        print("Parameters logged.")

        # -------------------------------------------------
        # Log metrics
        # -------------------------------------------------

        mlflow.log_metrics(
            METRICS
        )

        print("Metrics logged.")

        # -------------------------------------------------
        # Log training plots
        # -------------------------------------------------

        plots = [
            "accuracy_curve.png",
            "loss_curve.png",
            "confusion_matrix.png",
        ]

        for plot in plots:

            plot_path = ARTIFACT_DIR / plot

            if plot_path.exists():

                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="plots"
                )

                print(
                    f"Logged artifact: {plot}"
                )

            else:

                print(
                    f"Warning: artifact not found: "
                    f"{plot}"
                )

        # -------------------------------------------------
        # Log PyTorch model
        # -------------------------------------------------

        # MLflow 3.15.2 may use PT2 serialization by
        # default. We explicitly use pickle serialization
        # and provide an input example.

        mlflow.pytorch.log_model(
            model,
            name="cats_dogs_cnn",
            input_example=example_input,
            serialization_format="pickle"
        )

        print(
            "PyTorch model logged successfully."
        )

        # -------------------------------------------------
        # Log original .pt model file
        # -------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model_files"
        )

        print(
            "best_model.pt logged as artifact."
        )

        # -------------------------------------------------
        # Print final run information
        # -------------------------------------------------

        print()
        print("=" * 60)
        print("MLflow logging completed successfully.")
        print("=" * 60)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Experiment: {EXPERIMENT_NAME}"
        )

        print(
            f"Test Accuracy: "
            f"{METRICS['test_accuracy']:.4f}"
        )

        print(
            f"Test F1 Score: "
            f"{METRICS['test_f1']:.4f}"
        )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()