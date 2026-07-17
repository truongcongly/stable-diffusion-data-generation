import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Move low-quality synthetic images out of the training folder.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--rejected-dir", type=Path, default=Path("data/rejected/synthetic"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/synthetic_quality_report.csv"))
    parser.add_argument("--require-face", action="store_true", help="Reject images when OpenCV cannot detect a face.")
    parser.add_argument("--dry-run", action="store_true", help="Only report rejected images without moving them.")
    return parser.parse_args()


def is_black_or_blank(image):
    image_array = np.asarray(image.convert("RGB"))
    brightness = image_array.mean()
    contrast = image_array.std()
    return brightness < 8 or contrast < 5


def has_detectable_face(image):
    image_array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    cascades = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_profileface.xml",
    ]

    for cascade_path in cascades:
        classifier = cv2.CascadeClassifier(cascade_path)
        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
        if len(faces) > 0:
            return True
    return False


def quality_status(path, require_face):
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        return "read_error", str(exc)

    if is_black_or_blank(image):
        return "reject", "black_or_blank"
    if require_face and not has_detectable_face(image):
        return "reject", "no_face_detected"
    return "keep", "ok"


def write_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "status", "reason", "new_path"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.rejected_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg"))

    rows = []
    rejected_count = 0
    kept_count = 0

    for image_path in tqdm(image_paths, desc="Checking synthetic images"):
        status, reason = quality_status(image_path, args.require_face)
        new_path = ""

        if status == "reject":
            rejected_count += 1
            new_path = args.rejected_dir / image_path.name
            if not args.dry_run:
                shutil.move(str(image_path), str(new_path))
        else:
            kept_count += 1

        rows.append(
            {
                "image_path": image_path.as_posix(),
                "status": status,
                "reason": reason,
                "new_path": new_path.as_posix() if new_path else "",
            }
        )

    write_report(args.report_path, rows)
    print(f"Kept images: {kept_count}")
    print(f"Rejected images: {rejected_count}")
    print(f"Saved report to {args.report_path}")
    if args.dry_run:
        print("Dry run only: no files were moved.")


if __name__ == "__main__":
    main()
