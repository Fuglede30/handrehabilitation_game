import os
import shutil
import csv
import random
from typing import List, Tuple
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset_gpu_mixed"  # Dataset with dimension-based classes
OUTPUT_DIR = PROJECT_ROOT / "yolo_results"
DATASET_YAML = OUTPUT_DIR / "dataset.yaml"
RUN_NAME = "train_class_gpu_6"

# Training parameters
MODEL_NAME = "yolo12s"
EPOCHS = 200  # Increased for better convergence
IMGSZ = 1024
BATCH_SIZE = 8   # CRITICAL: was 1, now 16 for stable gradients
DEVICE = 0  # GPU device (0 for first GPU, -1 for CPU)
LR = 0.003  # Learning rate
SEED = 42  # Random seed for reproducibility
WRKS = 4  # Number of workers for data loading

# In-training quality degradation to simulate distant/unclear objects
USE_DISTANCE_SIM_AUG = True
DISTANCE_SIM_PROB = 0.2
DISTANCE_SIM_BATCH_STRIDE = 8
DISTANCE_SIM_MAX_IMAGES = 1
DISTANCE_SIM_STOP_EPOCH_FRAC = 0.6
DOWNSCALE_MIN = 0.25
DOWNSCALE_MAX = 0.7
GAUSSIAN_NOISE_STD = 0.03

# Dimension classes (must match preprocessing)
DIMENSION_CLASSES = {
    0: "2x10",
    1: "2x2",
    2: "2x3",
    3: "2x4",
    4: "2x6",
    5: "2x8",
    6: "3x3",
    7: "4x4",
    8: "4x6",
    9: "4x8",
    10: "6x6"
}


def create_dataset_yaml():
    """Create YOLO dataset.yaml configuration file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use dimension classes directly (sorted by class index)
    class_names = [DIMENSION_CLASSES[i] for i in sorted(DIMENSION_CLASSES.keys())]
    
    dataset_config = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),  # Number of classes (11 plate types)
        "names": class_names  # Class names (dimensions: 2x10, 2x2, 2x3, 2x4, 2x6, 2x8, 3x3, 4x4, 4x6, 4x8, 6x6)
    }
    
    with open(DATASET_YAML, "w") as f:
        yaml.dump(dataset_config, f)
    
    print(f"Created dataset.yaml: {DATASET_YAML}")
    return DATASET_YAML


def _read_loss_columns(csv_path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Read epochs and train/val loss columns from results.csv."""
    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return epochs, train_losses, val_losses

        # Prefer box_loss; fall back to total loss if present
        train_key = "train/box_loss" if "train/box_loss" in reader.fieldnames else "train/loss"
        val_key = "val/box_loss" if "val/box_loss" in reader.fieldnames else "val/loss"

        if train_key not in reader.fieldnames or val_key not in reader.fieldnames:
            return epochs, train_losses, val_losses

        for row in reader:
            try:
                epochs.append(int(float(row.get("epoch", "0"))))
                train_losses.append(float(row[train_key]))
                val_losses.append(float(row[val_key]))
            except (TypeError, ValueError):
                continue

    return epochs, train_losses, val_losses


def plot_loss_curves(csv_path: Path, output_path: Path) -> None:
    """Plot train/val loss curves from results.csv."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipping loss plot.")
        return

    epochs, train_losses, val_losses = _read_loss_columns(csv_path)
    if not epochs or not train_losses or not val_losses:
        print(f"Warning: could not find loss columns in {csv_path}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {output_path}")


def _distance_sim_callback(trainer) -> None:
    """Apply low-resolution + noise augmentation on training batches."""
    if not USE_DISTANCE_SIM_AUG:
        return

    stop_epoch = int(EPOCHS * DISTANCE_SIM_STOP_EPOCH_FRAC)
    if int(getattr(trainer, "epoch", 0)) >= stop_epoch:
        return

    batch_i = int(getattr(trainer, "batch_i", 0))
    if DISTANCE_SIM_BATCH_STRIDE > 1 and (batch_i % DISTANCE_SIM_BATCH_STRIDE != 0):
        return
    if random.random() > DISTANCE_SIM_PROB:
        return

    batch = getattr(trainer, "batch", None)
    if not isinstance(batch, dict):
        return

    images = batch.get("img")
    if images is None or not torch.is_tensor(images) or images.ndim != 4:
        return

    n, c, h, w = images.shape
    if h < 8 or w < 8:
        return

    k = min(max(1, DISTANCE_SIM_MAX_IMAGES), n)
    if k < n:
        indices = torch.randint(0, n, (k,), device=images.device)
        selected = images[indices]
    else:
        indices = None
        selected = images

    factor = random.uniform(DOWNSCALE_MIN, DOWNSCALE_MAX)
    small_h = max(8, int(h * factor))
    small_w = max(8, int(w * factor))

    low_res = F.interpolate(selected, size=(small_h, small_w), mode="bilinear", align_corners=False)
    restored = F.interpolate(low_res, size=(h, w), mode="nearest")
    noise = torch.randn_like(restored) * GAUSSIAN_NOISE_STD
    augmented = (restored + noise).clamp(0.0, 1.0)

    if indices is None:
        images.copy_(augmented)
    else:
        images[indices] = augmented


def train_yolo():
    """Train YOLO model."""
    print(f"Starting YOLO training...")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMGSZ}")
    print()
    
    # Load model
    model = YOLO(f"{MODEL_NAME}.pt")
    if USE_DISTANCE_SIM_AUG:
        model.add_callback("on_train_batch_start", _distance_sim_callback)
    
    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        lr0=LR,          
        workers=WRKS,
        seed=SEED,
        multi_scale=False,

        # AUGMENTATION (important for your case)
        mosaic=0.5,         # reduce (was 1.0 default)
        scale=0.8,          # YOLO symmetric scaling (~0.2 to ~1.8)
        translate=0.1,      # small movement
        fliplr=0.5,         # horizontal flip
        erasing=0.2,

        #change color intensity and lightning to simulate different background lightining and camera settings
        hsv_h=0.05,
        hsv_s=0.3,
        hsv_v=0.2,

        # training stability, finetuning of learning rate
        cos_lr=True,
    )

    
    # Persist per-epoch metrics for plotting (train/val loss in results.csv)
    run_dir = Path(getattr(results, "save_dir", OUTPUT_DIR / RUN_NAME))
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        saved_metrics = OUTPUT_DIR / f"{RUN_NAME}_results.csv"
        shutil.copyfile(results_csv, saved_metrics)
        print(f"Saved per-epoch metrics to: {saved_metrics}")

        plot_path = OUTPUT_DIR / f"{RUN_NAME}_loss.png"
        plot_loss_curves(saved_metrics, plot_path)
    else:
        print(f"Warning: results.csv not found at {results_csv}")

    return results


def main():
    """Main training function."""
    # Check dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        print("Please run preprocess_gpu_3.py first to create the dataset.")
        return

    # Ensure labels folders exist (YOLO expects labels beside images)
    for split in ["train", "val", "test"]:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.exists():
            print(f"Error: Labels folder missing: {labels_dir}")
            print("Please run preprocess_gpu_3.py to generate YOLO .txt labels.")
            return
    
    train_count = len(list((DATASET_DIR / "train" / "images").glob("*")))
    val_count = len(list((DATASET_DIR / "val" / "images").glob("*")))
    test_count = len(list((DATASET_DIR / "test" / "images").glob("*")))
    train_label_count = len(list((DATASET_DIR / "train" / "labels").glob("*.txt")))
    val_label_count = len(list((DATASET_DIR / "val" / "labels").glob("*.txt")))
    test_label_count = len(list((DATASET_DIR / "test" / "labels").glob("*.txt")))
    
    print(f"Dataset found:")
    print(f"  Train: {train_count} images")
    print(f"  Train labels: {train_label_count} txt")
    print(f"  Val: {val_count} images")
    print(f"  Val labels: {val_label_count} txt")
    print(f"  Test: {test_count} images")
    print(f"  Test labels: {test_label_count} txt")
    print()
    
    # Create YAML config
    create_dataset_yaml()
    print()
    
    # Train model
    try:
        train_yolo()
        print()
        print("Training complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Best model: {OUTPUT_DIR / RUN_NAME / 'weights' / 'best.pt'}")
        print(f"Last model: {OUTPUT_DIR / RUN_NAME / 'weights' / 'last.pt'}")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
