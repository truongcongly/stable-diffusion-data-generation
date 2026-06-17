import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio demo for real vs synthetic classification.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    return parser.parse_args()


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def predict(image):
        image = Image.fromarray(image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

        return {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Input image"),
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="Real vs Synthetic Face Classifier",
    )
    demo.launch()


if __name__ == "__main__":
    main()
