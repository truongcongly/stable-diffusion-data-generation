import argparse
import csv
import random
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ("real", "synthetic")


def parse_args():
    parser = argparse.ArgumentParser(description="Split raw real/synthetic images into train/val/test folders.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Folder containing real and synthetic folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Processed dataset folder.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/metadata/dataset_split_manifest.csv"),
        help="CSV file describing every copied image and split.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/metadata/dataset_summary.txt"),
        help="Text file containing split statistics.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Use all images instead of balancing classes to the smallest class count.",
    )
    return parser.parse_args()


def list_images(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_split(files, class_name, split_name, output_dir):
    target_dir = output_dir / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for source in tqdm(files, desc=f"Copying {split_name}/{class_name}"):
        target = target_dir / source.name
        shutil.copy2(source, target)
        manifest_rows.append(
            {
                "source_path": source.as_posix(),
                "target_path": target.as_posix(),
                "split": split_name,
                "label": class_name,
                "filename": source.name,
            }
        )

    return manifest_rows


def validate_ratios(train_ratio, val_ratio):
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train-ratio must be > 0 and val-ratio must be >= 0.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1 so the test split is not empty.")


def build_splits(files, train_ratio, val_ratio):
    train_end = int(len(files) * train_ratio)
    val_end = train_end + int(len(files) * val_ratio)

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_path", "target_path", "split", "label", "filename"]

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_path, raw_counts, split_counts, balanced_count, balanced):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Dataset Split Summary",
        "=====================",
        "",
        f"Balanced classes: {balanced}",
        f"Images per class used after balancing: {balanced_count if balanced else 'all available'}",
        "",
        "Raw image counts:",
    ]

    for class_name in CLASSES:
        lines.append(f"- {class_name}: {raw_counts[class_name]}")

    lines.extend(["", "Split counts:"])

    for split_name in ("train", "val", "test"):
        lines.append(f"- {split_name}:")
        for class_name in CLASSES:
            lines.append(f"  {class_name}: {split_counts[split_name][class_name]}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)
    random.seed(args.seed)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    class_files = {}
    raw_counts = {}

    for class_name in CLASSES:
        source_dir = args.raw_dir / class_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing folder: {source_dir}")

        files = list_images(source_dir)
        if not files:
            raise ValueError(f"No images found in {source_dir}")

        random.shuffle(files)
        class_files[class_name] = files
        raw_counts[class_name] = len(files)

    balanced = not args.no_balance
    balanced_count = min(raw_counts.values())

    if balanced:
        for class_name in CLASSES:
            class_files[class_name] = class_files[class_name][:balanced_count]

    manifest_rows = []
    split_counts = {split: {class_name: 0 for class_name in CLASSES} for split in ("train", "val", "test")}

    for class_name, files in class_files.items():
        splits = build_splits(files, args.train_ratio, args.val_ratio)
        for split_name, split_files in splits.items():
            rows = copy_split(split_files, class_name, split_name, args.output_dir)
            manifest_rows.extend(rows)
            split_counts[split_name][class_name] = len(split_files)

    write_manifest(args.manifest_path, manifest_rows)
    write_summary(args.summary_path, raw_counts, split_counts, balanced_count, balanced)

    print(f"Prepared dataset at {args.output_dir}")
    print(f"Saved manifest to {args.manifest_path}")
    print(f"Saved summary to {args.summary_path}")


if __name__ == "__main__":
    main()
