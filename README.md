# Stable Diffusion cho sinh dữ liệu ảnh khuôn mặt synthetic

Dự án này sử dụng Stable Diffusion để sinh ảnh khuôn mặt synthetic cho bài toán Computer Vision: phát hiện một ảnh khuôn mặt là ảnh thật hay ảnh do AI sinh ra.

## Điểm nổi bật

- Pipeline sinh dữ liệu synthetic end-to-end bằng Stable Diffusion.
- Classifier ResNet18 phân loại ảnh `real` và `synthetic`.
- Evaluation với accuracy, precision, recall, F1-score, confusion matrix và confidence từng ảnh.
- Phân tích prompt bằng NLP.
- Dự đoán ảnh ngoài dataset.
- Giải thích model bằng Grad-CAM.
- Demo tương tác bằng Gradio.

Tài liệu:

- [Báo cáo dự án](docs/report.md)
- [Checklist demo](docs/demo_checklist.md)

## Tổng quan dự án

Ý tưởng chính là dùng một mô hình text-to-image pretrained làm công cụ sinh dữ liệu. Stable Diffusion sinh ảnh khuôn mặt synthetic từ prompt văn bản. Các ảnh này được kết hợp với ảnh khuôn mặt thật để tạo dataset có nhãn, sau đó train classifier.

Pipeline:

```text
Prompt văn bản
-> Stable Diffusion sinh ảnh khuôn mặt synthetic
-> Kết hợp ảnh real và synthetic thành dataset
-> Train ResNet18 classifier
-> Model dự đoán real hoặc synthetic/AI-generated
-> Demo kết quả bằng Gradio app
```

## Bài toán

Ảnh khuôn mặt do AI sinh ra ngày càng chân thực. Vì vậy, dự án tập trung vào việc xây dựng pipeline sinh dữ liệu synthetic và train model phân biệt ảnh thật với ảnh AI.

Bài toán classification:

```text
Input: một ảnh khuôn mặt
Output: real hoặc synthetic
```

## Phạm vi dự án

Dự án không fine-tune Stable Diffusion. Stable Diffusion chỉ được dùng như mô hình pretrained để sinh dữ liệu ảnh.

Model được train là classifier:

```text
Stable Diffusion: dùng làm pretrained data generator
ResNet18: model được train để phân loại real/synthetic
```

## Mục tiêu

- Sinh ảnh khuôn mặt synthetic từ prompt.
- Thu thập ảnh khuôn mặt thật.
- Xây dựng dataset real/synthetic.
- Train ResNet18 classifier.
- Evaluate bằng accuracy, precision, recall, F1-score và confusion matrix.
- Demo bằng Gradio app.
- Thêm NLP prompt analysis.
- Thêm Grad-CAM explainability.

## Cách giải thích khi demo

Điểm quan trọng:

```text
Stable Diffusion dùng để sinh dữ liệu ảnh.
Model được train là classifier, không phải Stable Diffusion.
```

Câu giải thích gợi ý:

```text
Dự án dùng Stable Diffusion pretrained để sinh ảnh khuôn mặt synthetic từ prompt văn bản. Sau đó, ảnh synthetic được kết hợp với ảnh real để tạo dataset. Một ResNet18 classifier được train để phân biệt ảnh thật và ảnh AI-generated.
```

## Cài đặt

### Checklist yêu cầu

Trước khi chạy dự án, cần có:

- Python 3.11
- Git
- NVIDIA GPU driver
- PyTorch CUDA 12.1
- Internet cho lần đầu tải model Stable Diffusion và LFW dataset
- Visual Studio Code, khuyến nghị nhưng không bắt buộc

Kiểm tra Python:

```cmd
D:\project\.venv\Scripts\python.exe --version
```

Kết quả mong muốn:

```text
Python 3.11.x
```

Kiểm tra PyTorch và CUDA:

```cmd
D:\project\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Kết quả mong muốn:

```text
2.x.x+cu121
True
```

Kích hoạt virtual environment:

```cmd
cd /d D:\project\stable-diffusion-data-generation
..\.venv\Scripts\activate
```

Nếu không muốn activate, chạy trực tiếp bằng Python trong `.venv`:

```cmd
D:\project\.venv\Scripts\python.exe
```

Cài PyTorch CUDA:

```cmd
D:\project\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Cài thư viện còn lại:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Kiểm tra GPU

```cmd
D:\project\.venv\Scripts\python.exe src\check_gpu.py
```

Kết quả mong muốn:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Giai đoạn 2: Kiểm tra môi trường

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Script này kiểm tra:

- Phiên bản Python
- Các package cần thiết
- PyTorch CUDA
- Tên GPU và VRAM
- Các thư mục project
- Số ảnh real/synthetic
- Model đã train và các output evaluation

Nếu `CUDA available` là `False`, kiểm tra driver NVIDIA, PyTorch CUDA build và thử chạy lại trong Command Prompt thường.

## Sinh thử một ảnh

```cmd
D:\project\.venv\Scripts\python.exe src\generate_one_image.py
```

Output:

```text
data\raw\synthetic\test_face.png
```

## Giai đoạn 3: Sinh ảnh synthetic

Chạy test nhỏ:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 5
```

Ảnh được lưu tại:

```text
data\raw\synthetic
```

Metadata được lưu tại:

```text
data\metadata\synthetic_prompts.csv
```

Metadata gồm:

```text
image_path
label
prompt_id
prompt
negative_prompt
seed
width
height
steps
guidance_scale
model_id
```

Sinh dataset synthetic chính:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 200
```

## Giai đoạn 4: Thu thập ảnh real

Chạy test nhỏ:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 5
```

Ảnh real được lưu tại:

```text
data\raw\real
```

Metadata được lưu tại:

```text
data\metadata\real_lfw.csv
```

Tạo dataset real 200 ảnh:

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200
```

Nếu tải LFW lỗi do mạng, có thể tự đặt ảnh khuôn mặt thật vào:

```text
data\raw\real
```

## Giai đoạn 5: Chuẩn bị dataset

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
```

Input:

```text
data\raw\real
data\raw\synthetic
```

Output:

```text
data\processed\train\real
data\processed\train\synthetic
data\processed\val\real
data\processed\val\synthetic
data\processed\test\real
data\processed\test\synthetic
```

Script cũng lưu:

```text
data\metadata\dataset_split_manifest.csv
data\metadata\dataset_summary.txt
```

Mặc định script cân bằng dataset theo class có ít ảnh hơn để tránh model bị lệch class.

Tỷ lệ split mặc định:

```text
train: 70%
validation: 15%
test: 15%
```

## Giai đoạn 6: Train classifier

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

Input:

```text
data\processed\train
data\processed\val
```

Output:

```text
models\resnet18_real_vs_synthetic.pth
models\resnet18_real_vs_synthetic_final.pth
results\training_history.csv
results\training_config.json
```

Chạy test nhanh:

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
```

Giải thích khi demo:

```text
Ở giai đoạn này, ảnh synthetic và ảnh real được dùng để train ResNet18 classifier. Model học cách phân loại ảnh đầu vào là real hoặc synthetic.
```

## Giai đoạn 7: Evaluation

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Output:

```text
results\classification_report.txt
results\classification_report.json
results\metrics_summary.json
results\predictions.csv
results\confusion_matrix.png
results\confusion_matrix_normalized.png
```

Các chỉ số:

```text
accuracy
precision
recall
F1-score
support
```

## Giai đoạn 8: Gradio demo app

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\app.py
```

Mở URL do Gradio in ra, thường là:

```text
http://127.0.0.1:7860
```

App gồm:

```text
Predict Image:
- Upload ảnh
- Hiển thị xác suất real/synthetic
- Hiển thị nhãn dự đoán và confidence

Model Results:
- Hiển thị model path và device
- Hiển thị metrics
- Hiển thị classification report
- Hiển thị confusion matrix
```

## Giai đoạn 9: NLP prompt analysis

Chạy:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

Input:

```text
data\metadata\synthetic_prompts.csv
results\predictions.csv
```

Output:

```text
results\prompt_analysis\prompt_analysis.csv
results\prompt_analysis\prompt_analysis_report.txt
results\prompt_analysis\prompt_analysis_summary.json
results\prompt_analysis\prompt_keyword_frequency.png
results\prompt_analysis\prompt_length_distribution.png
results\prompt_analysis\confidence_by_prompt_id.png
```

Phần này phân tích:

```text
số từ trong prompt
số keyword
top keyword
tần suất keyword
phân bố độ dài prompt
confidence trung bình theo prompt ID
```

## Giai đoạn 10: Dự đoán ảnh ngoài dataset

Dự đoán một ảnh:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png
```

Dự đoán cả thư mục:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\real
```

Lưu output CSV:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\synthetic --output-csv results\external_predictions.csv
```

Nếu CUDA/cuDNN lỗi khi demo, dùng CPU:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

## Giai đoạn 11: Grad-CAM explainability

Grad-CAM giúp trực quan hóa vùng ảnh ảnh hưởng nhiều tới quyết định của classifier.

Chạy với ảnh synthetic:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png
```

Chạy với ảnh real:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\real\real_0007.jpg
```

Output:

```text
results\gradcam\<image_name>_heatmap.png
results\gradcam\<image_name>_overlay.png
results\gradcam\<image_name>_gradcam.json
```

Nếu CUDA/cuDNN lỗi khi demo:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

## Giai đoạn 12: Báo cáo và tài liệu portfolio

Tài liệu chính:

```text
docs\report.md
docs\demo_checklist.md
```

Trước khi nộp hoặc demo, nên chạy:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

Ảnh nên chụp cho slide:

```text
Gradio app prediction screen
confusion matrix
prompt keyword frequency chart
Grad-CAM overlay image
dataset summary
```

## Pipeline demo nhanh

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\app.py
```

Nếu còn thời gian:

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

## Đánh giá mô hình theo level

Giáo viên yêu cầu đánh giá full pipeline, vì vậy dự án được đánh giá theo nhiều level:

```text
Level 1: Kiểm tra môi trường, GPU, dữ liệu, checkpoint
Level 2: Đánh giá classifier trên test set bằng accuracy, precision, recall, F1-score
Level 3: Đánh giá từng ảnh bằng predictions.csv và confidence
Level 4: Đánh giá thực tế hơn bằng external prediction và Grad-CAM explainability
```

Full pipeline cần chứng minh:

```text
Prompt
-> Stable Diffusion sinh ảnh synthetic
-> Thu thập ảnh real
-> Chia dataset train/val/test
-> Train ResNet18 classifier
-> Evaluate model
-> Predict ảnh ngoài dataset
-> NLP prompt analysis
-> Grad-CAM explainability
-> Gradio demo app
```

Lệnh kiểm tra full pipeline rút gọn:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 1
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
D:\project\.venv\Scripts\python.exe src\app.py
```
