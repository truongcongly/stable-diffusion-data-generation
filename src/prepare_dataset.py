import argparse
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
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return parser.parse_args()


def list_images(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_split(files, class_name, split_name, output_dir):
    target_dir = output_dir / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for source in tqdm(files, desc=f"Copying {split_name}/{class_name}"):
        shutil.copy2(source, target_dir / source.name)


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    for class_name in CLASSES:
        source_dir = args.raw_dir / class_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing folder: {source_dir}")

        files = list_images(source_dir)
        if not files:
            raise ValueError(f"No images found in {source_dir}")

        random.shuffle(files)
        train_end = int(len(files) * args.train_ratio)
        val_end = train_end + int(len(files) * args.val_ratio)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, split_files in splits.items():
            copy_split(split_files, class_name, split_name, args.output_dir)

    print(f"Prepared dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
