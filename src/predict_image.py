import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict real vs synthetic for external images.")
    parser.add_argument("--image", type=Path, help="Path to a single image.")
    parser.add_argument("--image-dir", type=Path, help="Folder containing images to predict.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV file for predictions.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def build_model(num_classes, pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def list_images(image_path, image_dir):
    if image_path and image_dir:
        raise ValueError("Use either --image or --image-dir, not both.")

    if image_path:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]

    if image_dir:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {image_dir}")
        return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    raise ValueError("Provide --image or --image-dir.")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    pretrained = checkpoint.get("pretrained", False)

    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def predict_one(model, classes, transform, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    scores = {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}
    predicted_label = max(scores, key=scores.get)
    confidence = scores[predicted_label]

    return {
        "image_path": image_path.as_posix(),
        "predicted_label": predicted_label,
        "confidence": confidence,
        **{f"prob_{class_name}": scores[class_name] for class_name in classes},
    }


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if device_arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_csv(output_path, rows):
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    image_paths = list_images(args.image, args.image_dir)

    if not image_paths:
        raise ValueError("No images found for prediction.")

    model, classes = load_model(args.model_path, device)
    transform = build_transform()

    rows = []
    print(f"Using {device}")
    print(f"Model: {args.model_path}")

    for image_path in image_paths:
        try:
            row = predict_one(model, classes, transform, image_path, device)
        except RuntimeError as exc:
            if device != "cuda":
                raise
            print(f"CUDA prediction failed: {exc}")
            print("Falling back to CPU for prediction.")
            device = "cpu"
            model, classes = load_model(args.model_path, device)
            row = predict_one(model, classes, transform, image_path, device)

        rows.append(row)
        print(
            f"{row['image_path']} -> "
            f"{row['predicted_label']} ({row['confidence'] * 100:.2f}%)"
        )

    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
