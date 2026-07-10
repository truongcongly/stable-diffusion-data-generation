import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for real vs synthetic classifier.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gradcam"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if device_arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    pretrained = checkpoint.get("pretrained", False)

    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


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
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

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


def save_visualizations(image, cam, output_dir, image_stem):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * image_rgb + 0.45 * heatmap_color)

    heatmap_path = output_dir / f"{image_stem}_heatmap.png"
    overlay_path = output_dir / f"{image_stem}_overlay.png"

    Image.fromarray(heatmap_color).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)
    return heatmap_path, overlay_path


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = resolve_device(args.device)
    model, classes = load_model(args.model_path, device)
    transform = build_transform()
    image, image_tensor = preprocess_image(args.image, transform, device)

    try:
        with torch.enable_grad():
            raw_outputs = model(image_tensor)
            target_class = int(raw_outputs.argmax(dim=1).item())
            cam, probabilities = compute_gradcam(model, image_tensor, target_class)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"CUDA Grad-CAM failed: {exc}")
        print("Falling back to CPU.")
        device = "cpu"
        model, classes = load_model(args.model_path, device)
        image, image_tensor = preprocess_image(args.image, transform, device)
        with torch.enable_grad():
            raw_outputs = model(image_tensor)
            target_class = int(raw_outputs.argmax(dim=1).item())
            cam, probabilities = compute_gradcam(model, image_tensor, target_class)

    predicted_label = classes[target_class]
    confidence = probabilities[target_class]
    heatmap_path, overlay_path = save_visualizations(image, cam, args.output_dir, args.image.stem)

    result = {
        "image_path": args.image.as_posix(),
        "model_path": args.model_path.as_posix(),
        "device": device,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_probabilities": {class_name: float(prob) for class_name, prob in zip(classes, probabilities)},
        "heatmap_path": heatmap_path.as_posix(),
        "overlay_path": overlay_path.as_posix(),
    }
    result_path = args.output_dir / f"{args.image.stem}_gradcam.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Prediction: {predicted_label} ({confidence * 100:.2f}%)")
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved overlay to {overlay_path}")
    print(f"Saved metadata to {result_path}")


if __name__ == "__main__":
    main()
