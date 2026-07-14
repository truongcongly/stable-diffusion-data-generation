# Checklist demo

## Trước khi demo

Chạy kiểm tra môi trường:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Kết quả mong muốn:

```text
Python: 3.11.x [OK]
CUDA available: True
trained model: OK
classification report: OK
confusion matrix: OK
```

## Thứ tự demo

1. Giải thích phạm vi dự án.

```text
Stable Diffusion dùng để sinh dữ liệu ảnh.
Model được train là ResNet18 classifier.
```

2. Demo sinh ảnh synthetic.

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 1
```

3. Mở file thống kê dataset.

```text
data\metadata\dataset_summary.txt
```

4. Giải thích lệnh train.

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

5. Chạy hoặc mở kết quả evaluation.

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

6. Chạy Gradio app.

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

7. Demo NLP prompt analysis.

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

8. Demo Grad-CAM.

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

## Đánh giá mô hình theo level

Khi giáo viên hỏi về đánh giá mô hình, trình bày theo 4 level:

### Level 1: Kiểm tra môi trường và dữ liệu

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Nói:

```text
Level này kiểm tra Python, CUDA, GPU, package, số lượng ảnh real/synthetic, checkpoint và output evaluation.
```

### Level 2: Đánh giá classifier trên test set

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Nói:

```text
Level này đánh giá bằng accuracy, precision, recall, F1-score và confusion matrix.
```

### Level 3: Đánh giá từng ảnh

Mở file:

```text
results\predictions.csv
```

Nói:

```text
File này lưu true label, predicted label, confidence và correct/incorrect cho từng ảnh test.
```

### Level 4: Đánh giá ngoài dataset và giải thích model

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

Nói:

```text
Level này kiểm tra khả năng dự đoán ảnh bất kỳ và dùng Grad-CAM để giải thích vùng ảnh model chú ý.
```

## Full pipeline cần trình bày

```text
Prompt
-> Stable Diffusion sinh ảnh synthetic
-> Thu thập ảnh real
-> Chia dataset train/validation/test
-> Train ResNet18 classifier
-> Evaluate model
-> Dự đoán ảnh ngoài dataset
-> NLP prompt analysis
-> Grad-CAM explainability
-> Gradio demo app
```

## Câu giải thích chính

```text
Dự án dùng Stable Diffusion để sinh ảnh khuôn mặt synthetic. Sau đó, ảnh synthetic được kết hợp với ảnh real để train ResNet18 classifier phát hiện ảnh thật và ảnh AI-generated. Dự án cũng có phân tích prompt bằng NLP và giải thích model bằng Grad-CAM.
```

## Hạn chế cần nói khi demo

```text
Accuracy hiện tại cao trên dataset demo, nhưng dataset còn nhỏ và cần test thêm trên dữ liệu đa dạng hơn trước khi kết luận khả năng tổng quát ngoài thực tế.
```
