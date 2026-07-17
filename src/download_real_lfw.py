import argparse
import csv
import re
from pathlib import Path

from PIL import Image
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = Path("data/raw/real")
DEFAULT_METADATA_PATH = Path("data/metadata/real_lfw.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Download real face images from the LFW dataset.")
    parser.add_argument("--count", type=int, default=200, help="Maximum number of real images to save.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="CSV file for real image metadata.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting image index. If omitted, the next available index is used.",
    )
    parser.add_argument(
        "--dataset-offset",
        type=int,
        default=0,
        help="Offset into the LFW dataset before selecting images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
    return parser.parse_args()


def find_next_index(output_dir):
    pattern = re.compile(r"real_(\d+)\.jpg$")
    existing_indices = []

    for path in output_dir.glob("real_*.jpg"):
        match = pattern.match(path.name)
        if match:
            existing_indices.append(int(match.group(1)))

    if not existing_indices:
        return 1
    return max(existing_indices) + 1


def append_metadata(metadata_path, rows, replace=False):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metadata_path.exists() and not replace

    fieldnames = [
        "image_path",
        "label",
        "source_dataset",
        "lfw_index",
        "person_name",
        "width",
        "height",
    ]

    mode = "a" if file_exists else "w"
    with metadata_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def to_uint8_image(image_array):
    if image_array.max() <= 1.0:
        image_array = image_array * 255.0
    image_array = image_array.clip(0, 255).astype("uint8")
    return Image.fromarray(image_array)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_index = args.start_index or find_next_index(args.output_dir)

    dataset = fetch_lfw_people(color=True, resize=1.0, funneled=True)
    images = dataset.images[args.dataset_offset : args.dataset_offset + args.count]
    target_indices = dataset.target[args.dataset_offset : args.dataset_offset + args.count]
    target_names = dataset.target_names
    metadata_rows = []

    for offset, (image_array, target_index) in enumerate(
        tqdm(zip(images, target_indices), total=len(images), desc="Saving real face images")
    ):
        image_index = start_index + offset
        output_path = args.output_dir / f"real_{image_index:04d}.jpg"

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        image = to_uint8_image(image_array)
        image.save(output_path)

        metadata_rows.append(
            {
                "image_path": output_path.as_posix(),
                "label": "real",
                "source_dataset": "LFW",
                "lfw_index": args.dataset_offset + offset,
                "person_name": target_names[target_index],
                "width": image.width,
                "height": image.height,
            }
        )

    if metadata_rows:
        replace_metadata = args.overwrite and start_index == 1 and args.dataset_offset == 0
        append_metadata(args.metadata_path, metadata_rows, replace=replace_metadata)

    print(f"Saved {len(metadata_rows)} real images to {args.output_dir}")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
