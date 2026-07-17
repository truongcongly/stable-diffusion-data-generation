import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


SUPPORTED_MODELS = ["resnet18", "mobilenet_v2", "efficientnet_b0"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare multiple image classifiers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_comparison"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/model_comparison"))
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_loaders(data_dir, batch_size, num_workers):
    train_transform, eval_transform = build_transforms()
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(model_name, num_classes, pretrained):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_loss_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_items += images.size(0)

    return total_loss / total_items, total_correct / total_items


def evaluate_test_metrics(model, loader, device):
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
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_precision, macro_recall, macro_f1


def save_checkpoint(path, model, model_name, classes, class_to_idx, best_val_accuracy, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "val_accuracy": best_val_accuracy,
            "history": history,
            "model_name": model_name,
            "pretrained": args.pretrained,
            "image_size": 224,
        },
        path,
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Model Comparison",
        "",
        "| Model | Best val accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {best_val_accuracy:.4f} | {test_accuracy:.4f} | "
            "{macro_precision:.4f} | {macro_recall:.4f} | {macro_f1:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_one_model(model_name, args, train_dataset, train_loader, val_loader, test_loader, device):
    model = build_model(model_name, len(train_dataset.classes), args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_accuracy = 0.0
    history = []
    checkpoint_path = args.model_dir / f"{model_name}_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_items = 0

        for images, labels in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (predictions == labels).sum().item()
            running_items += images.size(0)

        train_loss = running_loss / running_items
        train_accuracy = running_correct / running_items
        val_loss, val_accuracy = evaluate_loss_accuracy(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"{model_name} epoch {epoch}: "
            f"train_acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path,
                model,
                model_name,
                train_dataset.classes,
                train_dataset.class_to_idx,
                best_val_accuracy,
                history,
                args,
            )

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    accuracy, macro_precision, macro_recall, macro_f1 = evaluate_test_metrics(model, test_loader, device)
    history_path = args.output_dir / f"{model_name}_history.csv"
    write_csv(history_path, history)

    return {
        "model": model_name,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "checkpoint_path": checkpoint_path.as_posix(),
        "history_path": history_path.as_posix(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    train_dataset, _val_dataset, _test_dataset, train_loader, val_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")

    rows = []
    for model_name in args.models:
        print(f"\nTraining {model_name}")
        rows.append(train_one_model(model_name, args, train_dataset, train_loader, val_loader, test_loader, device))

    comparison_csv = args.output_dir / "model_comparison.csv"
    comparison_md = args.output_dir / "model_comparison.md"
    config_path = args.output_dir / "model_comparison_config.json"

    write_csv(comparison_csv, rows)
    write_markdown(comparison_md, rows)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Saved comparison CSV to {comparison_csv}")
    print(f"Saved comparison report to {comparison_md}")


if __name__ == "__main__":
    main()
