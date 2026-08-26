from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PetImages"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42

CLASS_NAMES = ["Cat", "Dog"]


# ---------------------------------------------------------
# Validate an image
# ---------------------------------------------------------

def is_valid_image(image_path):
    """
    Check whether an image can be opened and verified.
    """
    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


# ---------------------------------------------------------
# Collect valid images
# ---------------------------------------------------------

def collect_images():
    """
    Collect valid Cat and Dog image paths.
    """

    image_paths = []
    labels = []

    for class_name in CLASS_NAMES:

        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            if is_valid_image(image_path):

                image_paths.append(image_path)
                labels.append(class_name)

            else:

                print(f"Skipping invalid image: {image_path}")

    return image_paths, labels


# ---------------------------------------------------------
# Split dataset
# ---------------------------------------------------------

def split_dataset(image_paths, labels):
    """
    Create stratified 80/10/10 train/validation/test split.
    """

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    validation_paths, test_paths, validation_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            stratify=temp_labels,
            random_state=RANDOM_STATE
        )
    )

    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels
    )


# ---------------------------------------------------------
# Process and save images
# ---------------------------------------------------------

def process_images(image_paths, labels, split_name):
    """
    Convert images to RGB, resize to 224x224,
    and save them in the processed dataset.
    """

    split_dir = PROCESSED_DIR / split_name

    for class_name in CLASS_NAMES:
        (split_dir / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    for image_path, label in zip(image_paths, labels):

        destination_dir = split_dir / label

        destination_path = destination_dir / image_path.name

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")
                image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                image.save(
                    destination_path,
                    format="JPEG",
                    quality=95
                )

        except Exception as error:

            print(
                f"Failed processing {image_path}: {error}"
            )


# ---------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------

def main():

    print("Starting dataset preprocessing...")

    image_paths, labels = collect_images()

    print(f"Valid images found: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError("No valid images found.")

    (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels
    ) = split_dataset(image_paths, labels)

    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(validation_paths)}")
    print(f"Test images: {len(test_paths)}")

    # Remove previous processed dataset if it exists
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    process_images(
        train_paths,
        train_labels,
        "train"
    )

    process_images(
        validation_paths,
        validation_labels,
        "validation"
    )

    process_images(
        test_paths,
        test_labels,
        "test"
    )

    print("\nPreprocessing completed successfully.")

    print(f"Processed dataset location: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()