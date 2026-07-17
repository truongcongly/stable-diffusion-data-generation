import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch import nn
from torchvision import models, transforms

DEFAULT_GENERATION_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, painting, blurry, distorted face, extra eyes, "
    "bad anatomy, low quality, watermark, text"
)

generation_pipe = None
generation_device = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio demo for real vs synthetic classification.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/metrics_summary.json"))
    parser.add_argument("--report-path", type=Path, default=Path("results/classification_report.txt"))
    parser.add_argument("--confusion-matrix-path", type=Path, default=Path("results/confusion_matrix.png"))
    parser.add_argument("--generated-output-dir", type=Path, default=Path("data/raw/synthetic/app_generated"))
    parser.add_argument("--gradcam-output-dir", type=Path, default=Path("results/gradcam/app"))
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--generation-model-id", default=DEFAULT_GENERATION_MODEL)
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


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def normalize_uploaded_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def get_confidence_warning(confidence, threshold):
    if confidence >= threshold:
        return "Confidence is high enough for a normal demo interpretation."
    return (
        f"Warning: confidence is below {threshold * 100:.0f}%. "
        "The model is not very certain, so this image should be checked manually."
    )


def compute_gradcam(model, image_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(_module, _input, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        outputs = model(image_tensor)
        score = outputs[:, target_class].sum()

        model.zero_grad()
        score.backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().tolist()
        return cam, probabilities
    finally:
        forward_handle.remove()
        backward_handle.remove()


def create_gradcam_overlay(image, cam):
    image_rgb = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * image_rgb + 0.45 * heatmap_color)
    return Image.fromarray(heatmap_color), Image.fromarray(overlay)


def load_generation_pipeline(model_id):
    global generation_pipe, generation_device

    if generation_pipe is not None:
        return generation_pipe, generation_device

    if torch.cuda.is_available():
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            generation_pipe = pipe
            generation_device = "cuda"
            return generation_pipe, generation_device
        except Exception:
            torch.cuda.empty_cache()

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    generation_pipe = pipe
    generation_device = "cpu"
    return generation_pipe, generation_device


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

    transform = build_transform()

    def classify_pil_image(pil_image):
        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

        scores = {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}
        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]
        return scores, predicted_label, confidence

    def predict(image):
        if image is None:
            return {}, "No image uploaded.", "Please upload a face image before running prediction."

        pil_image = normalize_uploaded_image(image)
        scores, predicted_label, confidence = classify_pil_image(pil_image)
        summary = f"Prediction: {predicted_label}\nConfidence: {confidence * 100:.2f}%"

        if predicted_label == "synthetic":
            note = "The classifier predicts this image is AI-generated/synthetic."
        else:
            note = "The classifier predicts this image is real."

        note = f"{note}\n{get_confidence_warning(confidence, args.confidence_threshold)}"
        return scores, summary, note

    def explain_with_gradcam(image):
        if image is None:
            return None, None, {}, "No image uploaded.", "Please upload an image before running Grad-CAM."

        pil_image = normalize_uploaded_image(image)
        tensor = transform(pil_image).unsqueeze(0).to(device)

        try:
            with torch.enable_grad():
                raw_outputs = model(tensor)
                target_class = int(raw_outputs.argmax(dim=1).item())
                cam, probabilities = compute_gradcam(model, tensor, target_class)
        except RuntimeError as exc:
            return None, None, {}, "Grad-CAM failed.", f"Runtime error: {exc}"

        heatmap, overlay = create_gradcam_overlay(pil_image, cam)
        predicted_label = classes[target_class]
        confidence = probabilities[target_class]
        scores = {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}

        args.gradcam_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = args.gradcam_output_dir / f"gradcam_{timestamp}_heatmap.png"
        overlay_path = args.gradcam_output_dir / f"gradcam_{timestamp}_overlay.png"
        heatmap.save(heatmap_path)
        overlay.save(overlay_path)

        summary = f"Prediction: {predicted_label}\nConfidence: {confidence * 100:.2f}%"
        note = (
            "Vung mau nong tren anh overlay la khu vuc model tap trung de dua ra du doan. "
            f"Saved overlay: {overlay_path}"
        )
        return heatmap, overlay, scores, summary, note

    def generate_synthetic_image(prompt, negative_prompt, seed, steps, guidance_scale, width, height):
        prompt = (prompt or "").strip()
        negative_prompt = (negative_prompt or "").strip()
        if not prompt:
            return None, "Please enter a prompt before generating an image."

        try:
            pipe, pipe_device = load_generation_pipeline(args.generation_model_id)
            generator = torch.Generator(device=pipe_device).manual_seed(int(seed))

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, f"Generation failed: {exc}"

        args.generated_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.generated_output_dir / f"app_synthetic_{timestamp}_seed{int(seed)}.png"
        image.save(output_path)

        info = (
            f"Saved image to: {output_path}\n"
            f"Device: {pipe_device}\n"
            f"Model: {args.generation_model_id}\n"
            f"Seed: {int(seed)}"
        )
        return image, info

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

            with gr.Tab("Grad-CAM Explain"):
                with gr.Row():
                    gradcam_input = gr.Image(type="numpy", label="Input image")
                    with gr.Column():
                        gradcam_heatmap = gr.Image(type="pil", label="Heatmap")
                        gradcam_overlay = gr.Image(type="pil", label="Grad-CAM overlay")
                with gr.Row():
                    gradcam_scores = gr.Label(num_top_classes=2, label="Class probabilities")
                    with gr.Column():
                        gradcam_summary = gr.Textbox(label="Prediction summary", lines=2)
                        gradcam_note = gr.Textbox(label="Interpretation", lines=3)
                        gradcam_button = gr.Button("Generate Grad-CAM", variant="primary")

                gradcam_button.click(
                    fn=explain_with_gradcam,
                    inputs=gradcam_input,
                    outputs=[gradcam_heatmap, gradcam_overlay, gradcam_scores, gradcam_summary, gradcam_note],
                )

            with gr.Tab("Generate Synthetic Image"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            value=(
                                "realistic portrait photo of a human face, natural lighting, "
                                "neutral background, high detail, sharp focus"
                            ),
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative prompt",
                            lines=3,
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )
                        with gr.Row():
                            seed_input = gr.Number(label="Seed", value=42, precision=0)
                            steps_input = gr.Slider(label="Inference steps", minimum=10, maximum=50, value=25, step=1)
                        with gr.Row():
                            guidance_input = gr.Slider(label="Guidance scale", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
                            width_input = gr.Dropdown(label="Width", choices=[256, 384, 512], value=384)
                            height_input = gr.Dropdown(label="Height", choices=[256, 384, 512], value=384)
                        generate_button = gr.Button("Generate Image", variant="primary")
                    with gr.Column():
                        generated_image = gr.Image(type="pil", label="Generated synthetic image")
                        generation_info = gr.Textbox(label="Generation info", lines=5)

                generate_button.click(
                    fn=generate_synthetic_image,
                    inputs=[
                        prompt_input,
                        negative_prompt_input,
                        seed_input,
                        steps_input,
                        guidance_input,
                        width_input,
                        height_input,
                    ],
                    outputs=[generated_image, generation_info],
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
