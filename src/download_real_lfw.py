import argparse
from pathlib import Path

from PIL import Image
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = Path("data/raw/real")


def parse_args():
    parser = argparse.ArgumentParser(description="Download real face images from the LFW dataset.")
    parser.add_argument("--count", type=int, default=200, help="Maximum number of real images to save.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_lfw_people(color=True, resize=1.0, funneled=True)
    images = dataset.images[: args.count]

    for index, image_array in enumerate(tqdm(images, desc="Saving real face images"), start=1):
        image = Image.fromarray(image_array.astype("uint8"))
        image.save(args.output_dir / f"real_{index:04d}.jpg")

    print(f"Saved {len(images)} real images to {args.output_dir}")


if __name__ == "__main__":
    main()
