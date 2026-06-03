"""Preprocessing script to build YOLO dataset splits without augmentation."""
import xml.etree.ElementTree as ET
import random
import shutil
import math
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR_1 = PROJECT_ROOT / "raw_data_2" / "images"
ANNOTATIONS_DIR_1 = PROJECT_ROOT / "raw_data_2" / "annotations"
IMAGES_DIR_2 = PROJECT_ROOT / "images"
ANNOTATIONS_DIR_2 = PROJECT_ROOT / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "dataset_gpu_mixed"

# Configuration
RANDOM_SEED = 42
PROGRESS_EVERY = 10
TOTAL_IMAGES = 3000  # Total train images desired
BACKGROUND_RATIO = 0.10  # 10% of train images should have no objects
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Part number to dimension mapping (from XML labels)
PLATE_PART_NUMBERS = {
    "11212": "3x3",
    "3020": "2x4",
    "3021": "2x3",
    "3022": "2x2",
    "3031": "4x4",
    "3032": "4x6",
    "3034": "2x8",
    "3035": "4x8",
    "3795": "2x6",
    "3832": "2x10",
    "3958": "6x6"
}

# Dimension classes mapping (must match training order: sorted dimensions)
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
def create_dataset():
    """Create YOLO dataset with equal distribution from both sources."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting dataset creation...")
    print(
        f"Config -> TOTAL_IMAGES={TOTAL_IMAGES}, TRAIN={TRAIN_RATIO:.2f}, "
        f"VAL={VAL_RATIO:.2f}, TEST={TEST_RATIO:.2f}, BACKGROUND_RATIO={BACKGROUND_RATIO:.2f}"
    )
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED)
    
    # Collect object/background candidates from both sources
    object_images_1, background_images_1 = _collect_images(IMAGES_DIR_1, ANNOTATIONS_DIR_1)
    object_images_2, background_images_2 = _collect_images(IMAGES_DIR_2, ANNOTATIONS_DIR_2)
    print(
        f"Source 1 ({IMAGES_DIR_1.name}): {len(object_images_1)} object, {len(background_images_1)} background"
    )
    print(
        f"Source 2 ({IMAGES_DIR_2.name}): {len(object_images_2)} object, {len(background_images_2)} background"
    )

    available_objects = len(object_images_1) + len(object_images_2)
    available_backgrounds = len(background_images_1) + len(background_images_2)
    print(f"Available total: {available_objects} object, {available_backgrounds} background")

    desired_total = TOTAL_IMAGES
    for _ in range(3):
        train_count = math.ceil(desired_total * TRAIN_RATIO)
        val_count = math.ceil(desired_total * VAL_RATIO)
        test_count = desired_total - train_count - val_count

        target_train_background = min(
            int(round(train_count * BACKGROUND_RATIO)),
            train_count,
            available_backgrounds,
        )

        required_objects = desired_total - target_train_background
        if required_objects <= available_objects:
            break

        desired_total = available_objects + target_train_background

    train_count = math.ceil(desired_total * TRAIN_RATIO)
    val_count = math.ceil(desired_total * VAL_RATIO)
    test_count = desired_total - train_count - val_count

    target_train_background = min(
        int(round(train_count * BACKGROUND_RATIO)),
        train_count,
        available_backgrounds,
    )
    target_train_objects = train_count - target_train_background
    total_objects_needed = target_train_objects + val_count + test_count
    print(
        f"Planned split sizes -> train={train_count}, val={val_count}, test={test_count}"
    )
    print(
        f"Train targets -> objects={target_train_objects}, backgrounds={target_train_background}"
    )

    selected_objects = _balanced_sample(
        object_images_1,
        object_images_2,
        total_objects_needed,
    )
    selected_backgrounds = _balanced_sample(
        background_images_1,
        background_images_2,
        target_train_background,
    )
    print(
        f"Selected -> objects={len(selected_objects)}, backgrounds={len(selected_backgrounds)}"
    )

    random.shuffle(selected_objects)
    random.shuffle(selected_backgrounds)

    train_object_images = selected_objects[:target_train_objects]
    remaining_objects = selected_objects[target_train_objects:]

    val_images = remaining_objects[:val_count]
    test_images = remaining_objects[val_count:val_count + test_count]
    train_images = train_object_images + selected_backgrounds
    random.shuffle(train_images)
    print("Copying images and labels...")
    
    # Copy files
    for split, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        total_in_split = len(images)
        print(f"  {split}: {total_in_split} files")
        for index, (img_path, ann_path) in enumerate(images, start=1):
            _copy_image_and_label(img_path, ann_path, split)
            if index % PROGRESS_EVERY == 0 or index == total_in_split:
                print(f"    {split}: {index}/{total_in_split}")
    
    actual_train_background = sum(1 for _, ann_path in train_images if ann_path is None)
    print(f"Dataset created: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    print(
        f"Train backgrounds: {actual_train_background}/{len(train_images)} "
        f"({(actual_train_background / len(train_images) * 100) if train_images else 0:.1f}%)"
    )

    print("\nFinal split composition:")
    _print_split_stats("train", train_images)
    _print_split_stats("val", val_images)
    _print_split_stats("test", test_images)


def _print_split_stats(split_name, split_images):
    """Print split composition including background and source counts."""
    total = len(split_images)
    background_count = sum(1 for _, ann_path in split_images if ann_path is None)
    source1_count = sum(1 for img_path, _ in split_images if img_path.parent == IMAGES_DIR_1)
    source2_count = sum(1 for img_path, _ in split_images if img_path.parent == IMAGES_DIR_2)
    print(
        f"  {split_name}: total={total}, background={background_count}, "
        f"from_set_1={source1_count}, from_set_2={source2_count}"
    )


def _collect_images(images_dir, annotations_dir):
    """Collect object images (valid labels) and background images (no valid labels)."""
    annotation_map = {ann_file.stem: ann_file for ann_file in annotations_dir.glob("*.xml")}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    object_images = []
    background_images = []

    for img_file in images_dir.iterdir():
        if not img_file.is_file() or img_file.suffix.lower() not in image_extensions:
            continue

        ann_file = annotation_map.get(img_file.stem)
        if ann_file is not None and _has_valid_annotations(ann_file):
            object_images.append((img_file, ann_file))
        else:
            background_images.append((img_file, None))

    return object_images, background_images


def _balanced_sample(source_1, source_2, target_count):
    """Sample approximately equally from two sources, filling remainder from either source."""
    if target_count <= 0:
        return []

    source_1 = list(source_1)
    source_2 = list(source_2)

    half = target_count // 2
    take_1 = min(half, len(source_1))
    take_2 = min(half, len(source_2))

    sampled_1 = random.sample(source_1, take_1) if take_1 > 0 else []
    sampled_2 = random.sample(source_2, take_2) if take_2 > 0 else []

    selected = sampled_1 + sampled_2
    remaining_needed = target_count - len(selected)

    if remaining_needed > 0:
        remaining_pool = [
            item for item in (source_1 + source_2)
            if item not in selected
        ]
        extra_take = min(remaining_needed, len(remaining_pool))
        if extra_take > 0:
            selected.extend(random.sample(remaining_pool, extra_take))

    return selected


def _has_valid_annotations(ann_path):
    """Check if annotation has valid plate parts."""
    try:
        tree = ET.parse(ann_path)
        for obj in tree.findall("object"):
            part_num = obj.find("name").text
            if part_num in PLATE_PART_NUMBERS:
                return True
    except:
        pass
    return False


def _copy_image_and_label(img_path, ann_path, split):
    """Copy image and generate YOLO label file."""
    shutil.copy(img_path, OUTPUT_DIR / split / "images" / img_path.name)

    label_file = OUTPUT_DIR / split / "labels" / (img_path.stem + ".txt")

    # Background image: create empty YOLO label file
    if ann_path is None:
        label_file.write_text("")
        return

    image = Image.open(img_path)
    width, height = image.size

    tree = ET.parse(ann_path)
    yolo_lines = []

    for obj in tree.findall("object"):
        part_num = obj.find("name").text
        if part_num not in PLATE_PART_NUMBERS:
            continue

        bndbox = obj.find("bndbox")
        x1, y1 = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
        x2, y2 = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)

        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        w, h = (x2 - x1) / width, (y2 - y1) / height

        class_id = {v: k for k, v in DIMENSION_CLASSES.items()}[PLATE_PART_NUMBERS[part_num]]
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_file.write_text("\n".join(yolo_lines))


if __name__ == "__main__":
    create_dataset()
