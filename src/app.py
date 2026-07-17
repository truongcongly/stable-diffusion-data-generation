import argparse
import csv
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
    "hoạt hình, anime, tranh vẽ, minh họa, phong cách nghệ thuật, màu quá rực, ánh sáng neon, "
    "mặt méo, thừa mắt, sai giải phẫu, watermark, chữ, logo, hoa văn trừu tượng, mosaic, lưới, "
    "nhiều người, mặt nạ, mũ bảo hiểm, mannequin, vẽ mặt"
)

generation_pipe = None
generation_device = None

FONT_CSS = """
.gradio-container {
    font-size: 17px !important;
}
.gradio-container label,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container button {
    font-size: 16px !important;
}
.gradio-container .tab-nav button {
    font-size: 16px !important;
}
.gradio-container h1 {
    font-size: 34px !important;
}
.gradio-container h2 {
    font-size: 26px !important;
}
.gradio-container h3 {
    font-size: 22px !important;
}
"""



def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio demo for real vs synthetic classification.")
    parser.add_argument("--model-path", type=Path, default=Path("models/resnet18_real_vs_synthetic.pth"))
    parser.add_argument("--metrics-path", type=Path, default=Path("results/metrics_summary.json"))
    parser.add_argument("--report-path", type=Path, default=Path("results/classification_report.txt"))
    parser.add_argument("--confusion-matrix-path", type=Path, default=Path("results/confusion_matrix.png"))
    parser.add_argument("--generated-output-dir", type=Path, default=Path("data/raw/synthetic/app_generated"))
    parser.add_argument("--gradcam-output-dir", type=Path, default=Path("results/gradcam/app"))
    parser.add_argument("--demo-report-dir", type=Path, default=Path("results/demo_reports"))
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


def count_images(folder):
    if not folder.exists():
        return 0
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in image_extensions)


def latest_images(folder, limit=5):
    if not folder.exists():
        return []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [path for path in folder.rglob("*") if path.suffix.lower() in image_extensions]
    image_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return image_paths[:limit]


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
        return (
            "Độ tin cậy đủ cao để demo thông thường, nhưng ảnh ngoài dataset hoặc ảnh webcam "
            "vẫn có thể bị dự đoán sai do khác phân phối dữ liệu huấn luyện."
        )
    return (
        f"Cảnh báo: độ tin cậy thấp hơn {threshold * 100:.0f}%. "
        "Model chưa thật sự chắc chắn, nên cần kiểm tra thủ công."
    )


def get_external_image_warning():
    return (
        "Lưu ý ảnh ngoài dữ liệu: nếu ảnh đến từ webcam, internet hoặc không nằm trong tập "
        "train/test đã chuẩn bị, hãy xem kết quả chỉ mang tính tham khảo. Ánh sáng, chất lượng "
        "camera, cách crop, nền ảnh và góc mặt đều có thể làm model dự đoán sai."
    )


PROMPT_TRANSLATIONS = {
    "ảnh khuôn mặt chân thực độ phân giải thấp, phong cách giống dataset LFW, crop sát khuôn mặt": (
        "low resolution realistic face photo of one human, LFW dataset style, cropped face"
    ),
    "người trẻ tuổi": "young adult",
    "người trung niên": "middle-aged adult",
    "người lớn tuổi": "elderly person",
    "thiếu niên": "teenager",
    "nam": "male",
    "nữ": "female",
    "trung tính": "gender-neutral",
    "ánh sáng tự nhiên": "natural lighting",
    "ánh sáng studio": "studio lighting",
    "ánh sáng trong nhà dịu": "soft indoor lighting",
    "ánh sáng yếu": "low light",
    "chân dung nhìn thẳng": "front-facing portrait",
    "góc nghiêng ba phần tư": "three-quarter view",
    "góc nghiêng bên": "side profile",
    "cận cảnh khuôn mặt": "close-up face",
    "nền trung tính": "neutral background",
    "nền văn phòng": "office background",
    "nền ngoài trời": "outdoor background",
    "nền studio đơn giản": "plain studio background",
    "kết cấu da chân thực": "realistic skin texture",
    "phong cách ảnh hộ chiếu": "passport photo style",
    "phong cách chân dung đời thường": "casual portrait style",
    "ảnh chân dung chuyên nghiệp": "professional headshot",
    "ảnh chụp bằng camera thông thường": "ordinary camera photo",
    "khuôn mặt rõ ràng": "face clearly visible",
    "hoạt hình": "cartoon",
    "tranh vẽ": "painting",
    "minh họa": "illustration",
    "phong cách nghệ thuật": "artwork",
    "màu quá rực": "oversaturated",
    "ánh sáng neon": "neon colors",
    "mặt méo": "distorted face",
    "thừa mắt": "extra eyes",
    "sai giải phẫu": "bad anatomy",
    "chữ": "text",
    "hoa văn trừu tượng": "abstract pattern",
    "lưới": "grid",
    "nhiều người": "multiple people",
    "mặt nạ": "mask",
    "mũ bảo hiểm": "helmet",
    "vẽ mặt": "face paint",
}


def translate_prompt_to_english(text):
    translated = text or ""
    for vietnamese_text, english_text in PROMPT_TRANSLATIONS.items():
        translated = translated.replace(vietnamese_text, english_text)
    return translated


def build_prompt_from_options(age, gender, lighting, face_view, background, detail):
    parts = [
        "ảnh khuôn mặt chân thực độ phân giải thấp, phong cách giống dataset LFW, crop sát khuôn mặt",
        age,
        gender,
        lighting,
        face_view,
        background,
        detail,
        "ảnh chụp bằng camera thông thường",
        "khuôn mặt rõ ràng",
    ]
    return ", ".join(part for part in parts if part)


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
    report_text = load_text(args.report_path, "Chưa tìm thấy báo cáo đánh giá. Hãy chạy src/evaluate_model.py trước.")

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
            return {}, "Chưa có ảnh.", "Vui lòng upload ảnh khuôn mặt trước khi dự đoán."

        pil_image = normalize_uploaded_image(image)
        scores, predicted_label, confidence = classify_pil_image(pil_image)
        summary = f"Dự đoán: {predicted_label}\nĐộ tin cậy: {confidence * 100:.2f}%"

        if predicted_label == "synthetic":
            note = "Model dự đoán ảnh này là ảnh AI-generated/synthetic."
        else:
            note = "Model dự đoán ảnh này là ảnh real/thật."

        note = (
            f"{note}\n"
            f"{get_confidence_warning(confidence, args.confidence_threshold)}\n"
            f"{get_external_image_warning()}"
        )
        return scores, summary, note

    def explain_with_gradcam(image):
        if image is None:
            return None, None, {}, "Chưa có ảnh.", "Vui lòng upload ảnh trước khi chạy Grad-CAM."

        pil_image = normalize_uploaded_image(image)
        tensor = transform(pil_image).unsqueeze(0).to(device)

        try:
            with torch.enable_grad():
                raw_outputs = model(tensor)
                target_class = int(raw_outputs.argmax(dim=1).item())
                cam, probabilities = compute_gradcam(model, tensor, target_class)
        except RuntimeError as exc:
            return None, None, {}, "Grad-CAM thất bại.", f"Lỗi runtime: {exc}"

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

        summary = f"Dự đoán: {predicted_label}\nĐộ tin cậy: {confidence * 100:.2f}%"
        note = (
            "Vùng màu nóng trên ảnh overlay là khu vực model tập trung để đưa ra dự đoán. "
            f"Đã lưu ảnh overlay tại: {overlay_path}\n"
            f"{get_confidence_warning(confidence, args.confidence_threshold)}\n"
            f"{get_external_image_warning()}"
        )
        return heatmap, overlay, scores, summary, note

    def generate_synthetic_image(prompt, negative_prompt, seed, steps, guidance_scale, width, height):
        prompt = (prompt or "").strip()
        negative_prompt = (negative_prompt or "").strip()
        if not prompt:
            return None, "Vui lòng nhập prompt trước khi tạo ảnh."

        try:
            generation_prompt = translate_prompt_to_english(prompt)
            generation_negative_prompt = translate_prompt_to_english(negative_prompt)
            pipe, pipe_device = load_generation_pipeline(args.generation_model_id)
            generator = torch.Generator(device=pipe_device).manual_seed(int(seed))

            image = pipe(
                prompt=generation_prompt,
                negative_prompt=generation_negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, f"Tạo ảnh thất bại: {exc}"

        args.generated_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.generated_output_dir / f"app_synthetic_{timestamp}_seed{int(seed)}.png"
        image.save(output_path)

        info = (
            f"Đã lưu ảnh tại: {output_path}\n"
            f"Thiết bị: {pipe_device}\n"
            f"Model tạo ảnh: {args.generation_model_id}\n"
            f"Seed: {int(seed)}\n"
            f"Prompt gửi vào Stable Diffusion: {generation_prompt}"
        )
        return image, info

    def create_demo_report():
        args.demo_report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = args.demo_report_dir / f"demo_report_{timestamp}.md"
        csv_path = args.demo_report_dir / f"demo_report_{timestamp}.csv"

        raw_real_count = count_images(Path("data/raw/real"))
        raw_synthetic_count = count_images(Path("data/raw/synthetic"))
        train_real_count = count_images(Path("data/processed/train/real"))
        train_synthetic_count = count_images(Path("data/processed/train/synthetic"))
        val_real_count = count_images(Path("data/processed/val/real"))
        val_synthetic_count = count_images(Path("data/processed/val/synthetic"))
        test_real_count = count_images(Path("data/processed/test/real"))
        test_synthetic_count = count_images(Path("data/processed/test/synthetic"))

        current_metrics = load_metrics(args.metrics_path)
        gradcam_images = latest_images(args.gradcam_output_dir, limit=5)
        generated_images = latest_images(args.generated_output_dir, limit=5)

        rows = [
            {"metric": "raw_real_images", "value": raw_real_count},
            {"metric": "raw_synthetic_images", "value": raw_synthetic_count},
            {"metric": "train_real_images", "value": train_real_count},
            {"metric": "train_synthetic_images", "value": train_synthetic_count},
            {"metric": "val_real_images", "value": val_real_count},
            {"metric": "val_synthetic_images", "value": val_synthetic_count},
            {"metric": "test_real_images", "value": test_real_count},
            {"metric": "test_synthetic_images", "value": test_synthetic_count},
            {"metric": "accuracy", "value": current_metrics.get("accuracy", "N/A")},
            {"metric": "macro_precision", "value": current_metrics.get("macro_precision", "N/A")},
            {"metric": "macro_recall", "value": current_metrics.get("macro_recall", "N/A")},
            {"metric": "macro_f1", "value": current_metrics.get("macro_f1", "N/A")},
            {"metric": "confusion_matrix", "value": args.confusion_matrix_path.as_posix()},
            {"metric": "gradcam_examples", "value": "; ".join(path.as_posix() for path in gradcam_images)},
            {"metric": "generated_examples", "value": "; ".join(path.as_posix() for path in generated_images)},
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

        markdown_lines = [
            "# Báo cáo demo - Phân loại ảnh real và synthetic",
            "",
            f"Thời điểm tạo báo cáo: {timestamp}",
            "",
            "## Dữ liệu",
            "",
            f"- Ảnh real gốc: {raw_real_count}",
            f"- Ảnh synthetic gốc: {raw_synthetic_count}",
            f"- Train real/synthetic: {train_real_count}/{train_synthetic_count}",
            f"- Validation real/synthetic: {val_real_count}/{val_synthetic_count}",
            f"- Test real/synthetic: {test_real_count}/{test_synthetic_count}",
            "",
            "## Chỉ số đánh giá model",
            "",
            f"- Accuracy: {format_percent(current_metrics.get('accuracy'))}",
            f"- Macro precision: {format_percent(current_metrics.get('macro_precision'))}",
            f"- Macro recall: {format_percent(current_metrics.get('macro_recall'))}",
            f"- Macro F1: {format_percent(current_metrics.get('macro_f1'))}",
            "",
            "## Ma trận nhầm lẫn",
            "",
            f"- Đường dẫn: {args.confusion_matrix_path.as_posix()}",
            "",
            "## Ví dụ Grad-CAM",
            "",
        ]
        markdown_lines.extend(f"- {path.as_posix()}" for path in gradcam_images)
        markdown_lines.extend(["", "## Ví dụ ảnh synthetic đã tạo", ""])
        markdown_lines.extend(f"- {path.as_posix()}" for path in generated_images)
        markdown_lines.extend(["", "## Báo cáo phân loại", "", "```text", report_text, "```"])

        report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        preview = "\n".join(markdown_lines[:30])
        return preview, report_path.as_posix(), csv_path.as_posix()

    model_info = (
        f"Đường dẫn model: {args.model_path}\n"
        f"Thiết bị chạy: {device}\n"
        f"Nhãn phân loại: {', '.join(classes)}\n"
        f"Validation accuracy trong checkpoint: {format_percent(val_accuracy)}\n"
        f"Test accuracy: {format_percent(metrics.get('accuracy'))}\n"
        f"Macro F1: {format_percent(metrics.get('macro_f1'))}\n"
        f"Số ảnh test: {metrics.get('num_samples', 'N/A')}"
    )
    with gr.Blocks(title="Phân loại ảnh real và synthetic", css=FONT_CSS) as demo:
        gr.Markdown("# Phân loại ảnh real và synthetic")
        gr.Markdown(
            "Dự đoán ảnh khuôn mặt real/synthetic, giải thích bằng Grad-CAM, "
            "tạo ảnh synthetic và xuất báo cáo demo."
        )

        with gr.Tabs():
            with gr.Tab("Dự đoán ảnh"):
                with gr.Row():
                    image_input = gr.Image(type="numpy", label="Ảnh đầu vào")
                    with gr.Column():
                        label_output = gr.Label(num_top_classes=2, label="Xác suất từng lớp")
                        summary_output = gr.Textbox(label="Tóm tắt dự đoán", lines=2)
                        note_output = gr.Textbox(label="Nhận xét", lines=4)
                        predict_button = gr.Button("Dự đoán", variant="primary")

                predict_button.click(
                    fn=predict,
                    inputs=image_input,
                    outputs=[label_output, summary_output, note_output],
                )

            with gr.Tab("Giải thích Grad-CAM"):
                with gr.Row():
                    gradcam_input = gr.Image(type="numpy", label="Ảnh đầu vào")
                    with gr.Column():
                        gradcam_heatmap = gr.Image(type="pil", label="Heatmap")
                        gradcam_overlay = gr.Image(type="pil", label="Grad-CAM overlay")
                with gr.Row():
                    gradcam_scores = gr.Label(num_top_classes=2, label="Xác suất từng lớp")
                    with gr.Column():
                        gradcam_summary = gr.Textbox(label="Tóm tắt dự đoán", lines=2)
                        gradcam_note = gr.Textbox(label="Nhận xét", lines=5)
                        gradcam_button = gr.Button("Tạo Grad-CAM", variant="primary")

                gradcam_button.click(
                    fn=explain_with_gradcam,
                    inputs=gradcam_input,
                    outputs=[gradcam_heatmap, gradcam_overlay, gradcam_scores, gradcam_summary, gradcam_note],
                )

            with gr.Tab("Tạo ảnh synthetic"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Bộ tạo prompt")
                        with gr.Row():
                            age_input = gr.Dropdown(
                                label="Độ tuổi",
                                choices=[
                                    "người trẻ tuổi",
                                    "người trung niên",
                                    "người lớn tuổi",
                                    "thiếu niên",
                                ],
                                value="người trẻ tuổi",
                            )
                            gender_input = gr.Dropdown(
                                label="Giới tính",
                                choices=[
                                    "nam",
                                    "nữ",
                                    "trung tính",
                                ],
                                value="trung tính",
                            )
                        with gr.Row():
                            lighting_input = gr.Dropdown(
                                label="Ánh sáng",
                                choices=[
                                    "ánh sáng tự nhiên",
                                    "ánh sáng studio",
                                    "ánh sáng trong nhà dịu",
                                    "ánh sáng yếu",
                                ],
                                value="ánh sáng tự nhiên",
                            )
                            view_input = gr.Dropdown(
                                label="Góc mặt",
                                choices=[
                                    "chân dung nhìn thẳng",
                                    "góc nghiêng ba phần tư",
                                    "góc nghiêng bên",
                                    "cận cảnh khuôn mặt",
                                ],
                                value="chân dung nhìn thẳng",
                            )
                        with gr.Row():
                            background_input = gr.Dropdown(
                                label="Nền ảnh",
                                choices=[
                                    "nền trung tính",
                                    "nền văn phòng",
                                    "nền ngoài trời",
                                    "nền studio đơn giản",
                                ],
                                value="nền trung tính",
                            )
                            detail_input = gr.Dropdown(
                                label="Chi tiết phong cách",
                                choices=[
                                    "kết cấu da chân thực",
                                    "phong cách ảnh hộ chiếu",
                                    "phong cách chân dung đời thường",
                                    "ảnh chân dung chuyên nghiệp",
                                ],
                                value="kết cấu da chân thực",
                            )
                        build_prompt_button = gr.Button("Tạo prompt")
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            value=(
                                "ảnh khuôn mặt chân thực độ phân giải thấp, phong cách giống dataset LFW, "
                                "crop sát khuôn mặt, ánh sáng tự nhiên, nền trung tính, khuôn mặt rõ ràng"
                            ),
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative prompt / Prompt loại trừ",
                            lines=3,
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )
                        with gr.Row():
                            seed_input = gr.Number(label="Seed", value=42, precision=0)
                            steps_input = gr.Slider(
                                label="Số bước sinh ảnh", minimum=10, maximum=50, value=25, step=1
                            )
                        with gr.Row():
                            guidance_input = gr.Slider(
                                label="Mức bám prompt", minimum=1.0, maximum=15.0, value=7.5, step=0.5
                            )
                            width_input = gr.Dropdown(label="Chiều rộng", choices=[256, 384, 512], value=384)
                            height_input = gr.Dropdown(label="Chiều cao", choices=[256, 384, 512], value=384)
                        generate_button = gr.Button("Tạo ảnh", variant="primary")
                    with gr.Column():
                        generated_image = gr.Image(type="pil", label="Ảnh synthetic đã tạo")
                        generation_info = gr.Textbox(label="Thông tin tạo ảnh", lines=5)

                build_prompt_button.click(
                    fn=build_prompt_from_options,
                    inputs=[
                        age_input,
                        gender_input,
                        lighting_input,
                        view_input,
                        background_input,
                        detail_input,
                    ],
                    outputs=prompt_input,
                )

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

            with gr.Tab("Báo cáo demo tự động"):
                with gr.Column():
                    gr.Markdown(
                        "Tạo báo cáo Markdown và CSV gồm số lượng dữ liệu, chỉ số đánh giá, "
                        "đường dẫn ma trận nhầm lẫn, ví dụ Grad-CAM và ví dụ ảnh synthetic đã tạo."
                    )
                    report_button = gr.Button("Tạo báo cáo demo", variant="primary")
                    report_preview = gr.Textbox(label="Xem trước báo cáo", lines=18)
                    with gr.Row():
                        report_md_file = gr.File(label="Báo cáo Markdown")
                        report_csv_file = gr.File(label="Báo cáo CSV")

                report_button.click(
                    fn=create_demo_report,
                    inputs=[],
                    outputs=[report_preview, report_md_file, report_csv_file],
                )

            with gr.Tab("Kết quả model"):
                with gr.Row():
                    with gr.Column():
                        gr.Textbox(value=model_info, label="Thông tin model", lines=8)
                        gr.Textbox(value=report_text, label="Báo cáo phân loại", lines=12)
                    with gr.Column():
                        if args.confusion_matrix_path.exists():
                            gr.Image(value=args.confusion_matrix_path.as_posix(), label="Ma trận nhầm lẫn")
                        else:
                            gr.Markdown("Chưa tìm thấy ma trận nhầm lẫn. Hãy chạy `python src\\evaluate_model.py` trước.")

    demo.launch()

if __name__ == "__main__":
    main()
