# Stable Diffusion Data Generation

Dự án dùng Stable Diffusion để sinh ảnh khuôn mặt synthetic, kết hợp với ảnh khuôn mặt thật để huấn luyện mô hình phân loại `real` và `synthetic`.

Mục tiêu chính:

- Sinh ảnh khuôn mặt synthetic từ prompt.
- Tải/chuẩn bị ảnh khuôn mặt thật từ LFW.
- Tạo dataset train/validation/test.
- Huấn luyện ResNet18 classifier.
- Đánh giá model bằng accuracy, precision, recall, F1-score và confusion matrix.
- Demo dự đoán, Grad-CAM và tạo ảnh synthetic bằng Gradio.

## Pipeline

```text
Prompt
-> Stable Diffusion sinh ảnh synthetic
-> Kết hợp với ảnh real
-> Chia train/validation/test
-> Train ResNet18 classifier
-> Evaluate model
-> Demo bằng Gradio
```

Lưu ý: Stable Diffusion chỉ được dùng như công cụ sinh dữ liệu. Model được huấn luyện để phân loại là ResNet18, không phải Stable Diffusion.

## Cấu trúc chính

```text
src/
  generate_synthetic.py     Sinh ảnh synthetic
  download_real_lfw.py      Tải ảnh real từ LFW
  prepare_dataset.py        Chia dataset train/val/test
  train_model.py            Huấn luyện ResNet18
  evaluate_model.py         Đánh giá model
  predict_image.py          Dự đoán ảnh hoặc thư mục ảnh
  gradcam.py                Tạo Grad-CAM
  prompt_analysis.py        Phân tích prompt
  app.py                    Gradio demo app

data/
  raw/                      Dữ liệu gốc
  processed/                Dữ liệu đã chia train/val/test
  metadata/                 Metadata CSV

models/                     Checkpoint model
results/                    Metrics, report, confusion matrix, Grad-CAM
docs/                       Báo cáo và checklist demo
```

## Cài đặt

Yêu cầu khuyến nghị:

- Python 3.11
- NVIDIA GPU và CUDA nếu muốn sinh ảnh nhanh
- Internet cho lần đầu tải model/dataset

Tạo hoặc kích hoạt virtual environment, sau đó cài thư viện:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu cần cài PyTorch CUDA 12.1:

```cmd
D:\project\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra môi trường:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Kiểm tra GPU:

```cmd
D:\project\.venv\Scripts\python.exe src\check_gpu.py
```

## Chạy pipeline

Sinh ảnh synthetic:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 200
```

Tải ảnh real:

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200
```

Chuẩn bị dataset:

```cmd
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
```

Train model:

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

Đánh giá model:

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Chạy demo app:

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

Sau đó mở URL Gradio, thường là:

```text
http://127.0.0.1:7860
```

## Dự đoán và giải thích model

Dự đoán một ảnh:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png
```

Dự đoán cả thư mục:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\real
```

Tạo Grad-CAM:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

Phân tích prompt:

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

## Output quan trọng

```text
models\resnet18_real_vs_synthetic.pth
results\classification_report.txt
results\metrics_summary.json
results\predictions.csv
results\confusion_matrix.png
results\gradcam\
results\prompt_analysis\
```

## Demo nhanh

Nếu dữ liệu và model đã có sẵn, chỉ cần chạy:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\app.py
```

Tài liệu bổ sung:

- `docs\report.md`
- `docs\demo_checklist.md`
