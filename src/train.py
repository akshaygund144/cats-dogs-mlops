from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.data_loader import create_dataloaders
from src.model import CatsDogsCNN


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Training Configuration
# =========================================================

EPOCHS = 10

LEARNING_RATE = 0.001

CLASS_NAMES = ["Cat", "Dog"]


# =========================================================
# CPU Optimization
# =========================================================

if DEVICE.type == "cpu":
    torch.set_num_threads(4)


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

        optimizer.zero_grad(set_to_none=True)

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
        len(loader.dataset)
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# =========================================================
# Evaluation Function
# =========================================================

def evaluate(
    model,
    loader,
    criterion,
    return_predictions=False
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    actuals = []
    predictions_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

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

            if return_predictions:

                actuals.extend(
                    labels.cpu().numpy()
                )

                predictions_list.extend(
                    predictions.cpu().numpy()
                )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = (
        correct / total
    )

    if return_predictions:

        return (
            epoch_loss,
            epoch_accuracy,
            actuals,
            predictions_list
        )

    return (
        epoch_loss,
        epoch_accuracy
    )


# =========================================================
# Save Training Curves
# =========================================================

def save_training_curves(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies
):

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # -----------------------------------------------------
    # Loss Curve
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

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
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR / "loss_curve.png"
    )

    plt.close()

    # -----------------------------------------------------
    # Accuracy Curve
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

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
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR / "accuracy_curve.png"
    )

    plt.close()


# =========================================================
# Save Confusion Matrix
# =========================================================

def save_confusion_matrix(
    actuals,
    predictions
):

    cm = confusion_matrix(
        actuals,
        predictions
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR / "confusion_matrix.png"
    )

    plt.close()

    return cm


# =========================================================
# Main Training Pipeline
# =========================================================

def main():

    print("=" * 60)
    print("Cats vs Dogs CNN Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cpu":
        print("CPU threads:", torch.get_num_threads())

    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # -----------------------------------------------------
    # Load Data
    # -----------------------------------------------------

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print(
        f"Training samples: {len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: {len(validation_loader.dataset)}"
    )

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    # -----------------------------------------------------
    # Create Model
    # -----------------------------------------------------

    model = CatsDogsCNN(
        num_classes=2
    ).to(DEVICE)

    print("\nModel created successfully.")

    # -----------------------------------------------------
    # Loss Function
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Learning Rate Scheduler
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # -----------------------------------------------------
    # Training History
    # -----------------------------------------------------

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    best_validation_accuracy = 0.0

    best_model_path = (
        MODEL_DIR / "best_model.pt"
    )

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    print("\nStarting training...\n")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        (
            validation_loss,
            validation_accuracy
        ) = evaluate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_accuracy
        )

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            validation_accuracy
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"  -> Saved best model "
                f"(Val Acc: {validation_accuracy:.4f})"
            )

    # =====================================================
    # Load Best Model
    # =====================================================

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE
        )
    )

    # =====================================================
    # Final Test Evaluation
    # =====================================================

    (
        test_loss,
        test_accuracy,
        test_actuals,
        test_predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        return_predictions=True
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    precision = precision_score(
        test_actuals,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        test_actuals,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        test_actuals,
        test_predictions,
        zero_division=0
    )

    # =====================================================
    # Final Results
    # =====================================================

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Loss:       {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:   {test_accuracy:.4f}"
    )

    print(
        f"Precision:       {precision:.4f}"
    )

    print(
        f"Recall:          {recall:.4f}"
    )

    print(
        f"F1 Score:        {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            test_actuals,
            test_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # =====================================================
    # Confusion Matrix
    # =====================================================

    cm = save_confusion_matrix(
        test_actuals,
        test_predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # =====================================================
    # Save Training Curves
    # =====================================================

    save_training_curves(
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies
    )

    # =====================================================
    # Final Information
    # =====================================================

    print("\nBest validation accuracy:")
    print(
        f"{best_validation_accuracy:.4f}"
    )

    print("\nArtifacts saved to:")
    print(ARTIFACT_DIR)

    print("\nBest model saved to:")
    print(best_model_path)

    print("\nTraining completed successfully.")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()