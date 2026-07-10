import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
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


def build_model(num_classes, pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def write_predictions(output_path, rows):
    fieldnames = [
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "correct",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_summary(output_path, accuracy, y_true, y_pred, classes, model_path, data_dir):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    summary = {
        "model_path": model_path.as_posix(),
        "data_dir": data_dir.as_posix(),
        "classes": classes,
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_confusion_matrix(matrix, classes, output_path, normalized=False):
    plt.figure(figsize=(6, 5))
    fmt = ".2f" if normalized else "d"
    sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]
    pretrained = checkpoint.get("pretrained", False)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    prediction_rows = []
    sample_paths = [path for path, _ in dataset.samples]
    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            predictions = predictions.cpu().tolist()
            confidences = confidences.cpu().tolist()
            labels = labels.cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels)

            for label_index, prediction_index, confidence in zip(labels, predictions, confidences):
                true_label = dataset.classes[label_index]
                predicted_label = classes[prediction_index]
                prediction_rows.append(
                    {
                        "image_path": Path(sample_paths[sample_index]).as_posix(),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "correct": true_label == predicted_label,
                    }
                )
                sample_index += 1

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)
    normalized_matrix = confusion_matrix(y_true, y_pred, normalize="true")

    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    (args.output_dir / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )
    write_metrics_summary(
        args.output_dir / "metrics_summary.json",
        accuracy,
        y_true,
        y_pred,
        classes,
        args.model_path,
        args.data_dir,
    )
    write_predictions(args.output_dir / "predictions.csv", prediction_rows)

    plot_confusion_matrix(matrix, classes, args.output_dir / "confusion_matrix.png", normalized=False)
    plot_confusion_matrix(
        normalized_matrix,
        classes,
        args.output_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
