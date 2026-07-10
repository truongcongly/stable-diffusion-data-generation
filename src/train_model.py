import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier for real vs synthetic faces.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed dataset folder.")
    parser.add_argument("--output-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--final-output-path", type=Path, default=Path("models/resnet18_real_vs_synthetic_final.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("results/training_history.csv"))
    parser.add_argument("--config-path", type=Path, default=Path("results/training_config.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ResNet18 weights.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, num_workers):
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

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, train_loader, val_loader


def build_model(num_classes, pretrained):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
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


def write_history(history_path, history):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_config(config_path, args, train_dataset, device):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": args.data_dir.as_posix(),
        "output_path": args.output_path.as_posix(),
        "final_output_path": args.final_output_path.as_posix(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pretrained": args.pretrained,
        "device": device,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_size": len(train_dataset),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_checkpoint(path, model, optimizer, epoch, classes, train_dataset, best_val_accuracy, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "class_to_idx": train_dataset.class_to_idx,
            "val_accuracy": best_val_accuracy,
            "history": history,
            "model_name": "resnet18",
            "pretrained": args.pretrained,
            "image_size": 224,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    train_dataset, train_loader, val_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(train_dataset.classes), pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_config(args.config_path, args, train_dataset, device)
    history = []

    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation batches: {len(val_loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_items = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
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
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        write_history(args.history_path, history)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                args.output_path,
                model,
                optimizer,
                epoch,
                train_dataset.classes,
                train_dataset,
                best_val_accuracy,
                history,
                args,
            )
            print(f"Saved best model to {args.output_path}")

    save_checkpoint(
        args.final_output_path,
        model,
        optimizer,
        args.epochs,
        train_dataset.classes,
        train_dataset,
        best_val_accuracy,
        history,
        args,
    )
    print(f"Saved final model to {args.final_output_path}")
    print(f"Saved training history to {args.history_path}")
    print(f"Saved training config to {args.config_path}")


if __name__ == "__main__":
    main()
