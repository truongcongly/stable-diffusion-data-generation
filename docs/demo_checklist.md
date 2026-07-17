# Checklist demo

## TrÆ°á»›c khi demo

Cháº¡y kiá»ƒm tra mÃ´i trÆ°á»ng:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Káº¿t quáº£ mong muá»‘n:

```text
Python: 3.11.x [OK]
CUDA available: True
trained model: OK
classification report: OK
confusion matrix: OK
```

## Thá»© tá»± demo

1. Giáº£i thÃ­ch pháº¡m vi dá»± Ã¡n.

```text
Stable Diffusion dÃ¹ng Ä‘á»ƒ sinh dá»¯ liá»‡u áº£nh.
Model Ä‘Æ°á»£c train lÃ  ResNet18 classifier.
```

2. Demo sinh áº£nh synthetic.

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 1
```

3. Má»Ÿ file thá»‘ng kÃª dataset.

```text
data\metadata\dataset_summary.txt
```

4. Giáº£i thÃ­ch lá»‡nh train.

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

5. Cháº¡y hoáº·c má»Ÿ káº¿t quáº£ evaluation.

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

6. Cháº¡y Gradio app.

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

7. Demo NLP prompt analysis.

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

8. Demo Grad-CAM.

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

## ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh theo level

Khi giÃ¡o viÃªn há»i vá» Ä‘Ã¡nh giÃ¡ mÃ´ hÃ¬nh, trÃ¬nh bÃ y theo 4 level:

### Level 1: Kiá»ƒm tra mÃ´i trÆ°á»ng vÃ  dá»¯ liá»‡u

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

NÃ³i:

```text
Level nÃ y kiá»ƒm tra Python, CUDA, GPU, package, sá»‘ lÆ°á»£ng áº£nh real/synthetic, checkpoint vÃ  output evaluation.
```

### Level 2: ÄÃ¡nh giÃ¡ classifier trÃªn test set

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

NÃ³i:

```text
Level nÃ y Ä‘Ã¡nh giÃ¡ báº±ng accuracy, precision, recall, F1-score vÃ  confusion matrix.
```

### Level 3: ÄÃ¡nh giÃ¡ tá»«ng áº£nh

Má»Ÿ file:

```text
results\predictions.csv
```

NÃ³i:

```text
File nÃ y lÆ°u true label, predicted label, confidence vÃ  correct/incorrect cho tá»«ng áº£nh test.
```

### Level 4: ÄÃ¡nh giÃ¡ ngoÃ i dataset vÃ  giáº£i thÃ­ch model

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

NÃ³i:

```text
Level nÃ y kiá»ƒm tra kháº£ nÄƒng dá»± Ä‘oÃ¡n áº£nh báº¥t ká»³ vÃ  dÃ¹ng Grad-CAM Ä‘á»ƒ giáº£i thÃ­ch vÃ¹ng áº£nh model chÃº Ã½.
```

## Full pipeline cáº§n trÃ¬nh bÃ y

```text
Prompt
-> Stable Diffusion sinh áº£nh synthetic
-> Thu tháº­p áº£nh real
-> Chia dataset train/validation/test
-> Train ResNet18 classifier
-> Evaluate model
-> Dá»± Ä‘oÃ¡n áº£nh ngoÃ i dataset
-> NLP prompt analysis
-> Grad-CAM explainability
-> Gradio demo app
```

## CÃ¢u giáº£i thÃ­ch chÃ­nh

```text
Dá»± Ã¡n dÃ¹ng Stable Diffusion Ä‘á»ƒ sinh áº£nh khuÃ´n máº·t synthetic. Sau Ä‘Ã³, áº£nh synthetic Ä‘Æ°á»£c káº¿t há»£p vá»›i áº£nh real Ä‘á»ƒ train ResNet18 classifier phÃ¡t hiá»‡n áº£nh tháº­t vÃ  áº£nh AI-generated. Dá»± Ã¡n cÅ©ng cÃ³ phÃ¢n tÃ­ch prompt báº±ng NLP vÃ  giáº£i thÃ­ch model báº±ng Grad-CAM.
```

## Háº¡n cháº¿ cáº§n nÃ³i khi demo

```text
Accuracy hiá»‡n táº¡i cao trÃªn dataset demo, nhÆ°ng dataset cÃ²n nhá» vÃ  cáº§n test thÃªm trÃªn dá»¯ liá»‡u Ä‘a dáº¡ng hÆ¡n trÆ°á»›c khi káº¿t luáº­n kháº£ nÄƒng tá»•ng quÃ¡t ngoÃ i thá»±c táº¿.
```

