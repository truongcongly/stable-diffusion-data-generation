import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio demo for real vs synthetic classification.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/metrics_summary.json"))
    parser.add_argument("--report-path", type=Path, default=Path("results/classification_report.txt"))
    parser.add_argument("--confusion-matrix-path", type=Path, default=Path("results/confusion_matrix.png"))
    return parser.parse_args()


def build_model(num_classes, pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_text(path, fallback):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def load_metrics(path):
    if not path.exists():
        return {
            "accuracy": None,
            "macro_f1": None,
            "num_samples": None,
        }

    return json.loads(path.read_text(encoding="utf-8"))


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]
    pretrained = checkpoint.get("pretrained", False)
    val_accuracy = checkpoint.get("val_accuracy")

    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metrics = load_metrics(args.metrics_path)
    report_text = load_text(args.report_path, "Evaluation report not found. Run src/evaluate_model.py first.")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def predict(image):
        if image is None:
            return {}, "No image uploaded.", "Please upload a face image before running prediction."

        image = Image.fromarray(image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

        scores = {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}
        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]
        summary = f"Prediction: {predicted_label}\nConfidence: {confidence * 100:.2f}%"

        if predicted_label == "synthetic":
            note = "The classifier predicts this image is AI-generated/synthetic."
        else:
            note = "The classifier predicts this image is real."

        return scores, summary, note

    model_info = (
        f"Model path: {args.model_path}\n"
        f"Device: {device}\n"
        f"Classes: {', '.join(classes)}\n"
        f"Validation accuracy in checkpoint: {format_percent(val_accuracy)}\n"
        f"Test accuracy: {format_percent(metrics.get('accuracy'))}\n"
        f"Macro F1: {format_percent(metrics.get('macro_f1'))}\n"
        f"Test samples: {metrics.get('num_samples', 'N/A')}"
    )

    with gr.Blocks(title="Real vs Synthetic Face Classifier") as demo:
        gr.Markdown("# Real vs Synthetic Face Classifier")
        gr.Markdown(
            "Upload a face image to classify it as real or synthetic. "
            "Stable Diffusion is used for data generation; this app runs the trained ResNet18 classifier."
        )

        with gr.Tabs():
            with gr.Tab("Predict Image"):
                with gr.Row():
                    image_input = gr.Image(type="numpy", label="Input image")
                    with gr.Column():
                        label_output = gr.Label(num_top_classes=2, label="Class probabilities")
                        summary_output = gr.Textbox(label="Prediction summary", lines=2)
                        note_output = gr.Textbox(label="Interpretation", lines=2)
                        predict_button = gr.Button("Predict", variant="primary")

                predict_button.click(
                    fn=predict,
                    inputs=image_input,
                    outputs=[label_output, summary_output, note_output],
                )

            with gr.Tab("Model Results"):
                gr.Textbox(value=model_info, label="Model info", lines=8)
                gr.Textbox(value=report_text, label="Classification report", lines=12)
                if args.confusion_matrix_path.exists():
                    gr.Image(value=args.confusion_matrix_path.as_posix(), label="Confusion matrix")
                else:
                    gr.Markdown("Confusion matrix not found. Run `python src\\evaluate_model.py` first.")

    demo.launch()


if __name__ == "__main__":
    main()
