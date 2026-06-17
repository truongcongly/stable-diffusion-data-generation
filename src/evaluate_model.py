import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the real vs synthetic classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    (args.output_dir / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
