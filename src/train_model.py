import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier for real vs synthetic faces.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed dataset folder.")
    parser.add_argument("--output-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def build_loaders(data_dir, batch_size):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, train_loader, val_loader


def build_model(num_classes):
    model = models.resnet18(weights=None)
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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    train_dataset, train_loader, val_loader = build_loaders(args.data_dir, args.batch_size)
    model = build_model(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

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

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "val_accuracy": val_accuracy,
                },
                args.output_path,
            )
            print(f"Saved best model to {args.output_path}")


if __name__ == "__main__":
    main()
