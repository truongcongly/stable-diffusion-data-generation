# Stable Diffusion cho sinh dá»¯ liá»‡u áº£nh khuÃ´n máº·t synthetic

Dá»± Ã¡n nÃ y sá»­ dá»¥ng Stable Diffusion Ä‘á»ƒ sinh áº£nh khuÃ´n máº·t synthetic cho bÃ i toÃ¡n Computer Vision: phÃ¡t hiá»‡n má»™t áº£nh khuÃ´n máº·t lÃ  áº£nh tháº­t hay áº£nh do AI sinh ra.

## Äiá»ƒm ná»•i báº­t

- Pipeline sinh dá»¯ liá»‡u synthetic end-to-end báº±ng Stable Diffusion.
- Classifier ResNet18 phÃ¢n loáº¡i áº£nh `real` vÃ  `synthetic`.
- Evaluation vá»›i accuracy, precision, recall, F1-score, confusion matrix vÃ  confidence tá»«ng áº£nh.
- PhÃ¢n tÃ­ch prompt báº±ng NLP.
- Dá»± Ä‘oÃ¡n áº£nh ngoÃ i dataset.
- Giáº£i thÃ­ch model báº±ng Grad-CAM.
- Demo tÆ°Æ¡ng tÃ¡c báº±ng Gradio.

TÃ i liá»‡u:

- [BÃ¡o cÃ¡o dá»± Ã¡n](docs/report.md)
- [Checklist demo](docs/demo_checklist.md)

## Tá»•ng quan dá»± Ã¡n

Ã tÆ°á»Ÿng chÃ­nh lÃ  dÃ¹ng má»™t mÃ´ hÃ¬nh text-to-image pretrained lÃ m cÃ´ng cá»¥ sinh dá»¯ liá»‡u. Stable Diffusion sinh áº£nh khuÃ´n máº·t synthetic tá»« prompt vÄƒn báº£n. CÃ¡c áº£nh nÃ y Ä‘Æ°á»£c káº¿t há»£p vá»›i áº£nh khuÃ´n máº·t tháº­t Ä‘á»ƒ táº¡o dataset cÃ³ nhÃ£n, sau Ä‘Ã³ train classifier.

Pipeline:

```text
Prompt vÄƒn báº£n
-> Stable Diffusion sinh áº£nh khuÃ´n máº·t synthetic
-> Káº¿t há»£p áº£nh real vÃ  synthetic thÃ nh dataset
-> Train ResNet18 classifier
-> Model dá»± Ä‘oÃ¡n real hoáº·c synthetic/AI-generated
-> Demo káº¿t quáº£ báº±ng Gradio app
```

## BÃ i toÃ¡n

áº¢nh khuÃ´n máº·t do AI sinh ra ngÃ y cÃ ng chÃ¢n thá»±c. VÃ¬ váº­y, dá»± Ã¡n táº­p trung vÃ o viá»‡c xÃ¢y dá»±ng pipeline sinh dá»¯ liá»‡u synthetic vÃ  train model phÃ¢n biá»‡t áº£nh tháº­t vá»›i áº£nh AI.

BÃ i toÃ¡n classification:

```text
Input: má»™t áº£nh khuÃ´n máº·t
Output: real hoáº·c synthetic
```

## Pháº¡m vi dá»± Ã¡n

Dá»± Ã¡n khÃ´ng fine-tune Stable Diffusion. Stable Diffusion chá»‰ Ä‘Æ°á»£c dÃ¹ng nhÆ° mÃ´ hÃ¬nh pretrained Ä‘á»ƒ sinh dá»¯ liá»‡u áº£nh.

Model Ä‘Æ°á»£c train lÃ  classifier:

```text
Stable Diffusion: dÃ¹ng lÃ m pretrained data generator
ResNet18: model Ä‘Æ°á»£c train Ä‘á»ƒ phÃ¢n loáº¡i real/synthetic
```

## Má»¥c tiÃªu

- Sinh áº£nh khuÃ´n máº·t synthetic tá»« prompt.
- Thu tháº­p áº£nh khuÃ´n máº·t tháº­t.
- XÃ¢y dá»±ng dataset real/synthetic.
- Train ResNet18 classifier.
- Evaluate báº±ng accuracy, precision, recall, F1-score vÃ  confusion matrix.
- Demo báº±ng Gradio app.
- ThÃªm NLP prompt analysis.
- ThÃªm Grad-CAM explainability.

## CÃ¡ch giáº£i thÃ­ch khi demo

Äiá»ƒm quan trá»ng:

```text
Stable Diffusion dÃ¹ng Ä‘á»ƒ sinh dá»¯ liá»‡u áº£nh.
Model Ä‘Æ°á»£c train lÃ  classifier, khÃ´ng pháº£i Stable Diffusion.
```

CÃ¢u giáº£i thÃ­ch gá»£i Ã½:

```text
Dá»± Ã¡n dÃ¹ng Stable Diffusion pretrained Ä‘á»ƒ sinh áº£nh khuÃ´n máº·t synthetic tá»« prompt vÄƒn báº£n. Sau Ä‘Ã³, áº£nh synthetic Ä‘Æ°á»£c káº¿t há»£p vá»›i áº£nh real Ä‘á»ƒ táº¡o dataset. Má»™t ResNet18 classifier Ä‘Æ°á»£c train Ä‘á»ƒ phÃ¢n biá»‡t áº£nh tháº­t vÃ  áº£nh AI-generated.
```

## CÃ i Ä‘áº·t

### Checklist yÃªu cáº§u

TrÆ°á»›c khi cháº¡y dá»± Ã¡n, cáº§n cÃ³:

- Python 3.11
- Git
- NVIDIA GPU driver
- PyTorch CUDA 12.1
- Internet cho láº§n Ä‘áº§u táº£i model Stable Diffusion vÃ  LFW dataset
- Visual Studio Code, khuyáº¿n nghá»‹ nhÆ°ng khÃ´ng báº¯t buá»™c

Kiá»ƒm tra Python:

```cmd
D:\project\.venv\Scripts\python.exe --version
```

Káº¿t quáº£ mong muá»‘n:

```text
Python 3.11.x
```

Kiá»ƒm tra PyTorch vÃ  CUDA:

```cmd
D:\project\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Káº¿t quáº£ mong muá»‘n:

```text
2.x.x+cu121
True
```

KÃ­ch hoáº¡t virtual environment:

```cmd
cd /d D:\project\stable-diffusion-data-generation
..\.venv\Scripts\activate
```

Náº¿u khÃ´ng muá»‘n activate, cháº¡y trá»±c tiáº¿p báº±ng Python trong `.venv`:

```cmd
D:\project\.venv\Scripts\python.exe
```

CÃ i PyTorch CUDA:

```cmd
D:\project\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CÃ i thÆ° viá»‡n cÃ²n láº¡i:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Kiá»ƒm tra GPU

```cmd
D:\project\.venv\Scripts\python.exe src\check_gpu.py
```

Káº¿t quáº£ mong muá»‘n:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Giai Ä‘oáº¡n 2: Kiá»ƒm tra mÃ´i trÆ°á»ng

Cháº¡y:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Script nÃ y kiá»ƒm tra:

- PhiÃªn báº£n Python
- CÃ¡c package cáº§n thiáº¿t
- PyTorch CUDA
- TÃªn GPU vÃ  VRAM
- CÃ¡c thÆ° má»¥c project
- Sá»‘ áº£nh real/synthetic
- Model Ä‘Ã£ train vÃ  cÃ¡c output evaluation

Náº¿u `CUDA available` lÃ  `False`, kiá»ƒm tra driver NVIDIA, PyTorch CUDA build vÃ  thá»­ cháº¡y láº¡i trong Command Prompt thÆ°á»ng.

## Sinh thá»­ má»™t áº£nh

```cmd
D:\project\.venv\Scripts\python.exe src\generate_one_image.py
```

Output:

```text
data\raw\synthetic\test_face.png
```

## Giai Ä‘oáº¡n 3: Sinh áº£nh synthetic

Cháº¡y test nhá»:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 5
```

áº¢nh Ä‘Æ°á»£c lÆ°u táº¡i:

```text
data\raw\synthetic
```

Metadata Ä‘Æ°á»£c lÆ°u táº¡i:

```text
data\metadata\synthetic_prompts.csv
```

Metadata gá»“m:

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

Sinh dataset synthetic chÃ­nh:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 200
```

## Giai Ä‘oáº¡n 4: Thu tháº­p áº£nh real

Cháº¡y test nhá»:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 5
```

áº¢nh real Ä‘Æ°á»£c lÆ°u táº¡i:

```text
data\raw\real
```

Metadata Ä‘Æ°á»£c lÆ°u táº¡i:

```text
data\metadata\real_lfw.csv
```

Táº¡o dataset real 200 áº£nh:

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200
```

Náº¿u táº£i LFW lá»—i do máº¡ng, cÃ³ thá»ƒ tá»± Ä‘áº·t áº£nh khuÃ´n máº·t tháº­t vÃ o:

```text
data\raw\real
```

## Giai Ä‘oáº¡n 5: Chuáº©n bá»‹ dataset

Cháº¡y:

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

Script cÅ©ng lÆ°u:

```text
data\metadata\dataset_split_manifest.csv
data\metadata\dataset_summary.txt
```

Máº·c Ä‘á»‹nh script cÃ¢n báº±ng dataset theo class cÃ³ Ã­t áº£nh hÆ¡n Ä‘á»ƒ trÃ¡nh model bá»‹ lá»‡ch class.

Tá»· lá»‡ split máº·c Ä‘á»‹nh:

```text
train: 70%
validation: 15%
test: 15%
```

## Giai Ä‘oáº¡n 6: Train classifier

Cháº¡y:

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

Cháº¡y test nhanh:

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
```

Giáº£i thÃ­ch khi demo:

```text
á»ž giai Ä‘oáº¡n nÃ y, áº£nh synthetic vÃ  áº£nh real Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ train ResNet18 classifier. Model há»c cÃ¡ch phÃ¢n loáº¡i áº£nh Ä‘áº§u vÃ o lÃ  real hoáº·c synthetic.
```

## Giai Ä‘oáº¡n 7: Evaluation

Cháº¡y:

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

CÃ¡c chá»‰ sá»‘:

```text
accuracy
precision
recall
F1-score
support
```

## Giai Ä‘oáº¡n 8: Gradio demo app

Cháº¡y:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\app.py
```

Má»Ÿ URL do Gradio in ra, thÆ°á»ng lÃ :

```text
http://127.0.0.1:7860
```

App gá»“m:

```text
Predict Image:
- Upload áº£nh
- Hiá»ƒn thá»‹ xÃ¡c suáº¥t real/synthetic
- Hiá»ƒn thá»‹ nhÃ£n dá»± Ä‘oÃ¡n vÃ  confidence

Model Results:
- Hiá»ƒn thá»‹ model path vÃ  device
- Hiá»ƒn thá»‹ metrics
- Hiá»ƒn thá»‹ classification report
- Hiá»ƒn thá»‹ confusion matrix
```

## Giai Ä‘oáº¡n 9: NLP prompt analysis

Cháº¡y:

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

Pháº§n nÃ y phÃ¢n tÃ­ch:

```text
sá»‘ tá»« trong prompt
sá»‘ keyword
top keyword
táº§n suáº¥t keyword
phÃ¢n bá»‘ Ä‘á»™ dÃ i prompt
confidence trung bÃ¬nh theo prompt ID
```

## Giai Ä‘oáº¡n 10: Dá»± Ä‘oÃ¡n áº£nh ngoÃ i dataset

Dá»± Ä‘oÃ¡n má»™t áº£nh:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png
```

Dá»± Ä‘oÃ¡n cáº£ thÆ° má»¥c:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\real
```

LÆ°u output CSV:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\synthetic --output-csv results\external_predictions.csv
```

Náº¿u CUDA/cuDNN lá»—i khi demo, dÃ¹ng CPU:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

## Giai Ä‘oáº¡n 11: Grad-CAM explainability

Grad-CAM giÃºp trá»±c quan hÃ³a vÃ¹ng áº£nh áº£nh hÆ°á»Ÿng nhiá»u tá»›i quyáº¿t Ä‘á»‹nh cá»§a classifier.

Cháº¡y vá»›i áº£nh synthetic:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png
```

Cháº¡y vá»›i áº£nh real:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\real\real_0007.jpg
```

Output:

```text
results\gradcam\<image_name>_heatmap.png
results\gradcam\<image_name>_overlay.png
results\gradcam\<image_name>_gradcam.json
```

Náº¿u CUDA/cuDNN lá»—i khi demo:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

## Giai Ä‘oáº¡n 12: BÃ¡o cÃ¡o vÃ  tÃ i liá»‡u portfolio

TÃ i liá»‡u chÃ­nh:

```text
docs\report.md
docs\demo_checklist.md
```

TrÆ°á»›c khi ná»™p hoáº·c demo, nÃªn cháº¡y:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

áº¢nh nÃªn chá»¥p cho slide:

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

Náº¿u cÃ²n thá»i gian:

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
```

## ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh theo level

GiÃ¡o viÃªn yÃªu cáº§u Ä‘Ã¡nh giÃ¡ full pipeline, vÃ¬ váº­y dá»± Ã¡n Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡ theo nhiá»u level:

```text
Level 1: Kiá»ƒm tra mÃ´i trÆ°á»ng, GPU, dá»¯ liá»‡u, checkpoint
Level 2: ÄÃ¡nh giÃ¡ classifier trÃªn test set báº±ng accuracy, precision, recall, F1-score
Level 3: ÄÃ¡nh giÃ¡ tá»«ng áº£nh báº±ng predictions.csv vÃ  confidence
Level 4: ÄÃ¡nh giÃ¡ thá»±c táº¿ hÆ¡n báº±ng external prediction vÃ  Grad-CAM explainability
```

Full pipeline cáº§n chá»©ng minh:

```text
Prompt
-> Stable Diffusion sinh áº£nh synthetic
-> Thu tháº­p áº£nh real
-> Chia dataset train/val/test
-> Train ResNet18 classifier
-> Evaluate model
-> Predict áº£nh ngoÃ i dataset
-> NLP prompt analysis
-> Grad-CAM explainability
-> Gradio demo app
```

Lá»‡nh kiá»ƒm tra full pipeline rÃºt gá»n:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 1
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
D:\project\.venv\Scripts\python.exe src\app.py
```

