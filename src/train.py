from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import mlflow
import mlflow.pytorch

from src.data_loader import (
    create_dataloaders,
    IMAGE_SIZE,
    BATCH_SIZE
)

from src.model import CatsDogsCNN


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"

NUM_EPOCHS = 10
LEARNING_RATE = 0.001

CLASS_NAMES = ["Cat", "Dog"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MLflow Configuration
# =========================================================

# MLflow 3.x uses a database backend.
# We already created mlflow.db earlier.

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Cats vs Dogs CNN"
)


# =========================================================
# CPU Optimization
# =========================================================

if DEVICE.type == "cpu":
    torch.set_num_threads(4)


# =========================================================
# Previous Training Result
# =========================================================

# This is the best validation accuracy obtained during the
# already-completed 10-epoch training run.
#
# We DO NOT retrain just to recover this value.

PREVIOUS_BEST_VALIDATION_ACCURACY = 0.8896


# =========================================================
# Training Function
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss /
        total
    )

    epoch_accuracy = (
        correct /
        total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# =========================================================
# Evaluation Function
# =========================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            total_samples += (
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    test_loss = (
        total_loss /
        total_samples
    )

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    test_precision = precision_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    test_recall = recall_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    test_f1 = f1_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    return (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        report,
        matrix
    )


# =========================================================
# Create Training Plots
# =========================================================

def create_training_plots(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies
):

    # -----------------------------------------------------
    # Loss curve
    # -----------------------------------------------------

    loss_path = (
        ARTIFACTS_DIR /
        "loss_curve.png"
    )

    if train_losses and validation_losses:

        epochs = range(
            1,
            len(train_losses) + 1
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            epochs,
            train_losses,
            label="Training Loss"
        )

        plt.plot(
            epochs,
            validation_losses,
            label="Validation Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(
            "Cats vs Dogs - Loss Curve"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            loss_path
        )

        plt.close()


    # -----------------------------------------------------
    # Accuracy curve
    # -----------------------------------------------------

    accuracy_path = (
        ARTIFACTS_DIR /
        "accuracy_curve.png"
    )

    if train_accuracies and validation_accuracies:

        epochs = range(
            1,
            len(train_accuracies) + 1
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            epochs,
            train_accuracies,
            label="Training Accuracy"
        )

        plt.plot(
            epochs,
            validation_accuracies,
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title(
            "Cats vs Dogs - Accuracy Curve"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            accuracy_path
        )

        plt.close()

    return (
        loss_path,
        accuracy_path
    )


# =========================================================
# Confusion Matrix Plot
# =========================================================

def create_confusion_matrix_plot(
    matrix
):

    confusion_matrix_path = (
        ARTIFACTS_DIR /
        "confusion_matrix.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(matrix)

    plt.title(
        "Cats vs Dogs - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path
    )

    plt.close()

    return confusion_matrix_path


# =========================================================
# Load Existing Best Model
# =========================================================

def load_best_model():

    if not BEST_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nBest model was not found.\n"
            f"Expected location:\n"
            f"{BEST_MODEL_PATH}\n\n"
            "Training has intentionally NOT been started.\n"
            "Please train the model first."
        )

    model = CatsDogsCNN(
        num_classes=2
    )

    model = model.to(DEVICE)

    state_dict = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# =========================================================
# Main Pipeline
# =========================================================

def main():

    print("=" * 60)
    print("Cats vs Dogs CNN - MLflow Evaluation")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Image Size: {IMAGE_SIZE}"
    )

    print(
        f"Batch Size: {BATCH_SIZE}"
    )

    print(
        f"Configured Epochs: {NUM_EPOCHS}"
    )

    print()
    print(
        "Existing trained model detected."
    )

    print(
        "Training will NOT be repeated."
    )

    print()


    # =====================================================
    # Load Data
    # =====================================================

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print(
        f"Training images: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )


    # =====================================================
    # Load Existing Best Model
    # =====================================================

    print()
    print(
        "Loading existing best_model.pt..."
    )

    model = load_best_model()

    print(
        "Best model loaded successfully."
    )


    # =====================================================
    # Evaluate Existing Model
    # =====================================================

    print()
    print(
        "Evaluating existing model on test dataset..."
    )

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        report,
        matrix
    ) = evaluate_model(
        model,
        test_loader
    )


    # =====================================================
    # Print Results
    # =====================================================

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Loss:       {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:   {test_accuracy:.4f}"
    )

    print(
        f"Precision:       {test_precision:.4f}"
    )

    print(
        f"Recall:          {test_recall:.4f}"
    )

    print(
        f"F1 Score:        {test_f1:.4f}"
    )

    print()
    print(
        "Classification Report:"
    )

    print(report)

    print()
    print(
        "Confusion Matrix:"
    )

    print(matrix)

    print()
    print(
        "Best validation accuracy:"
    )

    print(
        f"{PREVIOUS_BEST_VALIDATION_ACCURACY:.4f}"
    )


    # =====================================================
    # Start MLflow Run
    # =====================================================

    print()
    print(
        "Starting MLflow run..."
    )

    with mlflow.start_run(
        run_name="CatsDogsCNN_10_Epochs_Final"
    ):

        # -------------------------------------------------
        # Log Parameters
        # -------------------------------------------------

        mlflow.log_params(
            {
                "model": "CatsDogsCNN",
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "device": str(DEVICE),
                "training_reused": True,
                "model_source": "models/best_model.pt"
            }
        )


        # -------------------------------------------------
        # Log Metrics
        # -------------------------------------------------

        mlflow.log_metrics(
            {
                "best_validation_accuracy":
                    PREVIOUS_BEST_VALIDATION_ACCURACY,

                "test_loss":
                    test_loss,

                "test_accuracy":
                    test_accuracy,

                "test_precision":
                    test_precision,

                "test_recall":
                    test_recall,

                "test_f1":
                    test_f1
            }
        )


        # -------------------------------------------------
        # Log Existing Evaluation Artifacts
        # -------------------------------------------------

        loss_path = (
            ARTIFACTS_DIR /
            "loss_curve.png"
        )

        accuracy_path = (
            ARTIFACTS_DIR /
            "accuracy_curve.png"
        )


        if loss_path.exists():

            mlflow.log_artifact(
                loss_path,
                artifact_path="plots"
            )

            print(
                "Logged loss_curve.png"
            )


        if accuracy_path.exists():

            mlflow.log_artifact(
                accuracy_path,
                artifact_path="plots"
            )

            print(
                "Logged accuracy_curve.png"
            )


        # -------------------------------------------------
        # Create and Log Current Confusion Matrix
        # -------------------------------------------------

        confusion_matrix_path = (
            create_confusion_matrix_plot(
                matrix
            )
        )

        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="plots"
        )

        print(
            "Logged confusion_matrix.png"
        )


        # -------------------------------------------------
        # MLflow Model Logging
        # -------------------------------------------------
        #
        # IMPORTANT:
        #
        # MLflow 3.15.2 may select PT2 serialization by
        # default. PT2 requires an input_example.
        #
        # We explicitly use pickle serialization here.
        # This avoids the PT2 input_example error and is
        # appropriate for our PyTorch state-dict model.
        #

        input_example = torch.randn(
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():

            example_output = model(
                input_example
            )

        try:

            mlflow.pytorch.log_model(
                model,
                name="cats_dogs_cnn",
                serialization_format="pickle",
                input_example=input_example.cpu()
            )

        except TypeError:

            # Compatibility fallback for MLflow versions
            # where the serialization_format parameter is
            # handled differently.

            mlflow.pytorch.log_model(
                model,
                name="cats_dogs_cnn",
                input_example=input_example.cpu()
            )


        # -------------------------------------------------
        # Log Model File as Additional Artifact
        # -------------------------------------------------

        mlflow.log_artifact(
            BEST_MODEL_PATH,
            artifact_path="model_files"
        )


        # -------------------------------------------------
        # Print MLflow Run Information
        # -------------------------------------------------

        active_run = mlflow.active_run()

        if active_run is not None:

            print()
            print(
                "MLflow Run ID:"
            )

            print(
                active_run.info.run_id
            )

            print()
            print(
                "MLflow Experiment:"
            )

            print(
                "Cats vs Dogs CNN"
            )


    # =====================================================
    # Final Information
    # =====================================================

    print()
    print("=" * 60)
    print("MLflow logging completed successfully.")
    print("=" * 60)

    print()
    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()
    print(
        "Artifacts:"
    )

    print(
        ARTIFACTS_DIR
    )

    print()
    print(
        "MLflow database:"
    )

    print(
        MLFLOW_DB
    )

    print()
    print(
        "Training was NOT repeated."
    )

    print(
        "Existing best_model.pt was reused."
    )

    print()
    print("=" * 60)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()