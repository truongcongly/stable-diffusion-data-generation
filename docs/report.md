# Stable Diffusion cho sinh dữ liệu ảnh khuôn mặt synthetic

## 1. Giới thiệu

Dự án này nghiên cứu cách sử dụng Stable Diffusion để sinh dữ liệu ảnh synthetic cho một bài toán Computer Vision. Bài toán được chọn là phân loại ảnh khuôn mặt thật và ảnh khuôn mặt do AI sinh ra.

Trong dự án, Stable Diffusion pretrained được dùng để sinh ảnh khuôn mặt synthetic từ prompt văn bản. Sau đó, ảnh synthetic được kết hợp với ảnh khuôn mặt thật để huấn luyện mô hình ResNet18 phân loại ảnh đầu vào là `real` hay `synthetic`.

## 2. Bài toán

Ảnh khuôn mặt do AI tạo ra ngày càng chân thực. Vì vậy, việc xây dựng mô hình có khả năng phân biệt ảnh thật và ảnh synthetic/AI-generated là một bài toán có tính thực tế.

Bài toán phân loại:

```text
Input: một ảnh khuôn mặt
Output: real hoặc synthetic
```

## 3. Phạm vi dự án

Dự án này không fine-tune Stable Diffusion. Stable Diffusion được dùng như một mô hình pretrained để sinh dữ liệu ảnh.

Mô hình được train trong dự án là classifier:

```text
Stable Diffusion: mô hình pretrained dùng để sinh dữ liệu
ResNet18: mô hình được train để phân loại real/synthetic
```

## 4. Pipeline

```text
Prompt văn bản
-> Stable Diffusion sinh ảnh khuôn mặt synthetic
-> Thu thập ảnh khuôn mặt thật từ LFW
-> Chia dữ liệu thành train/validation/test
-> Train ResNet18 classifier
-> Evaluate trên test set
-> Demo bằng Gradio app
-> Phân tích prompt bằng NLP
-> Giải thích model bằng Grad-CAM
```

## 5. Dataset

Dataset gồm hai lớp:

```text
real
synthetic
```

Ảnh `real` được lấy từ LFW dataset thông qua `scikit-learn`.

Ảnh `synthetic` được sinh bằng Stable Diffusion từ nhiều prompt khác nhau. Với mỗi ảnh synthetic, dự án lưu metadata gồm:

```text
image path
prompt
negative prompt
seed
image size
inference steps
guidance scale
model ID
```

Trong lần chạy hiện tại:

```text
real images: 200
synthetic images sau khi cân bằng: 200
```

Dataset được chia như sau:

```text
train: 140 real, 140 synthetic
validation: 30 real, 30 synthetic
test: 30 real, 30 synthetic
```

## 6. Train model

Classifier sử dụng kiến trúc ResNet18.

Cấu hình train:

```text
model: ResNet18
classes: real, synthetic
image size: 224x224
optimizer: Adam
loss: CrossEntropyLoss
default epochs: 5
default batch size: 16
```

Script training lưu:

```text
best checkpoint
final checkpoint
training history CSV
training config JSON
```

## 7. Đánh giá

Script evaluation tính các chỉ số:

```text
accuracy
precision
recall
F1-score
confusion matrix
confidence cho từng ảnh
```

Kết quả hiện tại trên test set:

```text
accuracy: 1.0000
macro precision: 1.0000
macro recall: 1.0000
macro F1-score: 1.0000
test samples: 60
```

Confusion matrix được lưu tại:

```text
results/confusion_matrix.png
results/confusion_matrix_normalized.png
```

Prediction từng ảnh được lưu tại:

```text
results/predictions.csv
```

### Đánh giá theo nhiều level

Để đáp ứng yêu cầu đánh giá mô hình đầy đủ hơn, dự án có thể được đánh giá theo 4 level:

#### Level 1: Kiểm tra môi trường và dữ liệu

Mục tiêu của level này là xác nhận project có thể chạy ổn định.

```text
Kiểm tra Python, PyTorch, CUDA, GPU
Kiểm tra số lượng ảnh real/synthetic
Kiểm tra model checkpoint và output evaluation
```

Script sử dụng:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

#### Level 2: Đánh giá classifier trên test set

Mục tiêu là đánh giá model trên tập test đã tách riêng.

Các metric:

```text
accuracy
precision
recall
F1-score
confusion matrix
```

Script sử dụng:

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

#### Level 3: Đánh giá từng ảnh và confidence

Mục tiêu là xem model dự đoán từng ảnh như thế nào, confidence bao nhiêu, ảnh nào đúng/sai.

Output:

```text
results/predictions.csv
```

File này chứa:

```text
image_path
true_label
predicted_label
confidence
correct
```

#### Level 4: Đánh giá ngoài dataset và giải thích mô hình

Mục tiêu là kiểm tra model thực tế hơn bằng ảnh ngoài dataset và giải thích vùng ảnh model chú ý.

Script sử dụng:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image path\to\image.jpg
D:\project\.venv\Scripts\python.exe src\gradcam.py --image path\to\image.jpg --device cpu
```

Level này giúp dự án không chỉ có kết quả số liệu, mà còn có khả năng demo thực tế và giải thích model.

## 7.1 Full pipeline evaluation

Dự án được đánh giá theo full pipeline, tức là kiểm tra toàn bộ luồng từ sinh dữ liệu đến demo:

```text
1. Kiểm tra môi trường
2. Sinh ảnh synthetic bằng Stable Diffusion
3. Thu thập ảnh real từ LFW
4. Chia dataset train/validation/test
5. Train ResNet18 classifier
6. Evaluate trên test set
7. Dự đoán ảnh ngoài dataset
8. Phân tích prompt bằng NLP
9. Giải thích model bằng Grad-CAM
10. Demo bằng Gradio app
```

Các script tương ứng:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 5
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 5
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
D:\project\.venv\Scripts\python.exe src\app.py
```

Khi demo, không nhất thiết phải chạy lại toàn bộ pipeline với số lượng lớn. Có thể chạy bản rút gọn như sau:

```text
generate_synthetic.py --count 1
train_model.py --epochs 1
evaluate_model.py
app.py
prompt_analysis.py
gradcam.py
```

Điều quan trọng là chứng minh pipeline có thể chạy end-to-end.

## 8. Phân tích prompt bằng NLP

Dự án có thêm phần phân tích ngôn ngữ tự nhiên cho các prompt dùng để sinh ảnh synthetic.

Phần phân tích prompt trích xuất:

```text
số từ trong prompt
số keyword trong prompt
top keywords
tần suất keyword
phân bố độ dài prompt
confidence trung bình theo prompt ID
```

Kết quả hiện tại:

```text
synthetic prompt rows analyzed: 205
unique prompts: 10
average prompt word count: 14.11
top keywords: natural, background, lighting, neutral, skin
```

Phần này giúp liên kết prompt văn bản với dữ liệu ảnh được sinh ra và hành vi của classifier.

## 9. Grad-CAM Explainability

Grad-CAM được dùng để trực quan hóa vùng ảnh có ảnh hưởng nhiều tới quyết định của classifier.

Ví dụ output:

```text
results/gradcam/synthetic_0014_overlay.png
results/gradcam/real_0007_overlay.png
```

Nhờ Grad-CAM, dự án không chỉ đưa ra nhãn `real/synthetic`, mà còn giúp quan sát vùng ảnh model tập trung khi dự đoán.

## 10. Demo app

Dự án có app Gradio gồm hai tab chính:

```text
Predict Image
Model Results
```

App cho phép upload ảnh và hiển thị:

```text
xác suất từng class
nhãn dự đoán
confidence
classification report
confusion matrix
```

Chạy app:

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

## 11. Hạn chế

Dự án hiện là một prototype end-to-end đã chạy được, nhưng vẫn có một số hạn chế:

- Dataset còn nhỏ.
- Ảnh synthetic được sinh từ một mô hình Stable Diffusion.
- Ảnh real chủ yếu lấy từ LFW.
- Accuracy cao trên dataset demo chưa đảm bảo model tổng quát tốt trên mọi ảnh AI ngoài thực tế.
- Classifier có thể học khác biệt đặc trưng của dataset thay vì artifact tổng quát của ảnh synthetic.

## 12. Hướng phát triển

Các hướng cải tiến:

- Thêm nhiều dataset ảnh real hơn.
- Sinh ảnh synthetic từ nhiều diffusion model khác nhau.
- Test trên dataset AI-generated/deepfake bên ngoài.
- Tăng độ đa dạng của prompt.
- Thử backbone mạnh hơn như EfficientNet hoặc Vision Transformer.
- Tích hợp Grad-CAM trực tiếp vào Gradio app.
- Thử LoRA hoặc DreamBooth như phần mở rộng nâng cao của Stable Diffusion.

## 13. Kết luận

Dự án đã xây dựng một pipeline end-to-end cho bài toán sinh dữ liệu ảnh synthetic và phát hiện ảnh AI-generated. Stable Diffusion được dùng để sinh ảnh khuôn mặt synthetic, còn ResNet18 classifier được train để phân biệt ảnh real và synthetic. Dự án cũng bổ sung NLP prompt analysis, external image prediction, Grad-CAM explainability và Gradio demo app.
