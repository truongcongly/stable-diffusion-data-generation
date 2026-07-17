# Stable Diffusion cho sinh dá»¯ liá»‡u áº£nh khuÃ´n máº·t synthetic

## 1. Giá»›i thiá»‡u

Dá»± Ã¡n nÃ y nghiÃªn cá»©u cÃ¡ch sá»­ dá»¥ng Stable Diffusion Ä‘á»ƒ sinh dá»¯ liá»‡u áº£nh synthetic cho má»™t bÃ i toÃ¡n Computer Vision. BÃ i toÃ¡n Ä‘Æ°á»£c chá»n lÃ  phÃ¢n loáº¡i áº£nh khuÃ´n máº·t tháº­t vÃ  áº£nh khuÃ´n máº·t do AI sinh ra.

Trong dá»± Ã¡n, Stable Diffusion pretrained Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ sinh áº£nh khuÃ´n máº·t synthetic tá»« prompt vÄƒn báº£n. Sau Ä‘Ã³, áº£nh synthetic Ä‘Æ°á»£c káº¿t há»£p vá»›i áº£nh khuÃ´n máº·t tháº­t Ä‘á»ƒ huáº¥n luyá»‡n mÃ´ hÃ¬nh ResNet18 phÃ¢n loáº¡i áº£nh Ä‘áº§u vÃ o lÃ  `real` hay `synthetic`.

## 2. BÃ i toÃ¡n

áº¢nh khuÃ´n máº·t do AI táº¡o ra ngÃ y cÃ ng chÃ¢n thá»±c. VÃ¬ váº­y, viá»‡c xÃ¢y dá»±ng mÃ´ hÃ¬nh cÃ³ kháº£ nÄƒng phÃ¢n biá»‡t áº£nh tháº­t vÃ  áº£nh synthetic/AI-generated lÃ  má»™t bÃ i toÃ¡n cÃ³ tÃ­nh thá»±c táº¿.

BÃ i toÃ¡n phÃ¢n loáº¡i:

```text
Input: má»™t áº£nh khuÃ´n máº·t
Output: real hoáº·c synthetic
```

## 3. Pháº¡m vi dá»± Ã¡n

Dá»± Ã¡n nÃ y khÃ´ng fine-tune Stable Diffusion. Stable Diffusion Ä‘Æ°á»£c dÃ¹ng nhÆ° má»™t mÃ´ hÃ¬nh pretrained Ä‘á»ƒ sinh dá»¯ liá»‡u áº£nh.

MÃ´ hÃ¬nh Ä‘Æ°á»£c train trong dá»± Ã¡n lÃ  classifier:

```text
Stable Diffusion: mÃ´ hÃ¬nh pretrained dÃ¹ng Ä‘á»ƒ sinh dá»¯ liá»‡u
ResNet18: mÃ´ hÃ¬nh Ä‘Æ°á»£c train Ä‘á»ƒ phÃ¢n loáº¡i real/synthetic
```

## 4. Pipeline

```text
Prompt vÄƒn báº£n
-> Stable Diffusion sinh áº£nh khuÃ´n máº·t synthetic
-> Thu tháº­p áº£nh khuÃ´n máº·t tháº­t tá»« LFW
-> Chia dá»¯ liá»‡u thÃ nh train/validation/test
-> Train ResNet18 classifier
-> Evaluate trÃªn test set
-> Demo báº±ng Gradio app
-> PhÃ¢n tÃ­ch prompt báº±ng NLP
-> Giáº£i thÃ­ch model báº±ng Grad-CAM
```

## 5. Dataset

Dataset gá»“m hai lá»›p:

```text
real
synthetic
```

áº¢nh `real` Ä‘Æ°á»£c láº¥y tá»« LFW dataset thÃ´ng qua `scikit-learn`.

áº¢nh `synthetic` Ä‘Æ°á»£c sinh báº±ng Stable Diffusion tá»« nhiá»u prompt khÃ¡c nhau. Vá»›i má»—i áº£nh synthetic, dá»± Ã¡n lÆ°u metadata gá»“m:

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

Trong láº§n cháº¡y hiá»‡n táº¡i:

```text
real images: 200
synthetic images sau khi cÃ¢n báº±ng: 200
```

Dataset Ä‘Æ°á»£c chia nhÆ° sau:

```text
train: 140 real, 140 synthetic
validation: 30 real, 30 synthetic
test: 30 real, 30 synthetic
```

## 6. Train model

Classifier sá»­ dá»¥ng kiáº¿n trÃºc ResNet18.

Cáº¥u hÃ¬nh train:

```text
model: ResNet18
classes: real, synthetic
image size: 224x224
optimizer: Adam
loss: CrossEntropyLoss
default epochs: 5
default batch size: 16
```

Script training lÆ°u:

```text
best checkpoint
final checkpoint
training history CSV
training config JSON
```

## 7. ÄÃ¡nh giÃ¡

Script evaluation tÃ­nh cÃ¡c chá»‰ sá»‘:

```text
accuracy
precision
recall
F1-score
confusion matrix
confidence cho tá»«ng áº£nh
```

Káº¿t quáº£ hiá»‡n táº¡i trÃªn test set:

```text
accuracy: 0.9833
macro precision: 0.9839
macro recall: 0.9833
macro F1-score: 0.9833
test samples: 60
```

Sau khi sá»­a lá»—i lÆ°u áº£nh real, káº¿t quáº£ evaluation thá»±c táº¿ hÆ¡n: model Ä‘Ãºng 59/60 áº£nh test. CÃ³ 1 áº£nh synthetic bá»‹ dá»± Ä‘oÃ¡n nháº§m thÃ nh real vá»›i confidence khoáº£ng 80.77%.

Confusion matrix Ä‘Æ°á»£c lÆ°u táº¡i:

```text
results/confusion_matrix.png
results/confusion_matrix_normalized.png
```

Prediction tá»«ng áº£nh Ä‘Æ°á»£c lÆ°u táº¡i:

```text
results/predictions.csv
```

### ÄÃ¡nh giÃ¡ theo nhiá»u level

Äá»ƒ Ä‘Ã¡p á»©ng yÃªu cáº§u Ä‘Ã¡nh giÃ¡ mÃ´ hÃ¬nh Ä‘áº§y Ä‘á»§ hÆ¡n, dá»± Ã¡n cÃ³ thá»ƒ Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡ theo 4 level:

#### Level 1: Kiá»ƒm tra mÃ´i trÆ°á»ng vÃ  dá»¯ liá»‡u

Má»¥c tiÃªu cá»§a level nÃ y lÃ  xÃ¡c nháº­n project cÃ³ thá»ƒ cháº¡y á»•n Ä‘á»‹nh.

```text
Kiá»ƒm tra Python, PyTorch, CUDA, GPU
Kiá»ƒm tra sá»‘ lÆ°á»£ng áº£nh real/synthetic
Kiá»ƒm tra model checkpoint vÃ  output evaluation
```

Script sá»­ dá»¥ng:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

#### Level 2: ÄÃ¡nh giÃ¡ classifier trÃªn test set

Má»¥c tiÃªu lÃ  Ä‘Ã¡nh giÃ¡ model trÃªn táº­p test Ä‘Ã£ tÃ¡ch riÃªng.

CÃ¡c metric:

```text
accuracy
precision
recall
F1-score
confusion matrix
```

Script sá»­ dá»¥ng:

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

#### Level 3: ÄÃ¡nh giÃ¡ tá»«ng áº£nh vÃ  confidence

Má»¥c tiÃªu lÃ  xem model dá»± Ä‘oÃ¡n tá»«ng áº£nh nhÆ° tháº¿ nÃ o, confidence bao nhiÃªu, áº£nh nÃ o Ä‘Ãºng/sai.

Output:

```text
results/predictions.csv
```

File nÃ y chá»©a:

```text
image_path
true_label
predicted_label
confidence
correct
```

#### Level 4: ÄÃ¡nh giÃ¡ ngoÃ i dataset vÃ  giáº£i thÃ­ch mÃ´ hÃ¬nh

Má»¥c tiÃªu lÃ  kiá»ƒm tra model thá»±c táº¿ hÆ¡n báº±ng áº£nh ngoÃ i dataset vÃ  giáº£i thÃ­ch vÃ¹ng áº£nh model chÃº Ã½.

Script sá»­ dá»¥ng:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image path\to\image.jpg
D:\project\.venv\Scripts\python.exe src\gradcam.py --image path\to\image.jpg --device cpu
```

Level nÃ y giÃºp dá»± Ã¡n khÃ´ng chá»‰ cÃ³ káº¿t quáº£ sá»‘ liá»‡u, mÃ  cÃ²n cÃ³ kháº£ nÄƒng demo thá»±c táº¿ vÃ  giáº£i thÃ­ch model.

## 7.1 Full pipeline evaluation

Dá»± Ã¡n Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡ theo full pipeline, tá»©c lÃ  kiá»ƒm tra toÃ n bá»™ luá»“ng tá»« sinh dá»¯ liá»‡u Ä‘áº¿n demo:

```text
1. Kiá»ƒm tra mÃ´i trÆ°á»ng
2. Sinh áº£nh synthetic báº±ng Stable Diffusion
3. Thu tháº­p áº£nh real tá»« LFW
4. Chia dataset train/validation/test
5. Train ResNet18 classifier
6. Evaluate trÃªn test set
7. Dá»± Ä‘oÃ¡n áº£nh ngoÃ i dataset
8. PhÃ¢n tÃ­ch prompt báº±ng NLP
9. Giáº£i thÃ­ch model báº±ng Grad-CAM
10. Demo báº±ng Gradio app
```

CÃ¡c script tÆ°Æ¡ng á»©ng:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 5
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 5
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0010.png --device cpu
D:\project\.venv\Scripts\python.exe src\app.py
```

Khi demo, khÃ´ng nháº¥t thiáº¿t pháº£i cháº¡y láº¡i toÃ n bá»™ pipeline vá»›i sá»‘ lÆ°á»£ng lá»›n. CÃ³ thá»ƒ cháº¡y báº£n rÃºt gá»n nhÆ° sau:

```text
generate_synthetic.py --count 1
train_model.py --epochs 1
evaluate_model.py
app.py
prompt_analysis.py
gradcam.py
```

Äiá»u quan trá»ng lÃ  chá»©ng minh pipeline cÃ³ thá»ƒ cháº¡y end-to-end.

## 8. PhÃ¢n tÃ­ch prompt báº±ng NLP

Dá»± Ã¡n cÃ³ thÃªm pháº§n phÃ¢n tÃ­ch ngÃ´n ngá»¯ tá»± nhiÃªn cho cÃ¡c prompt dÃ¹ng Ä‘á»ƒ sinh áº£nh synthetic.

Pháº§n phÃ¢n tÃ­ch prompt trÃ­ch xuáº¥t:

```text
sá»‘ tá»« trong prompt
sá»‘ keyword trong prompt
top keywords
táº§n suáº¥t keyword
phÃ¢n bá»‘ Ä‘á»™ dÃ i prompt
confidence trung bÃ¬nh theo prompt ID
```

Káº¿t quáº£ hiá»‡n táº¡i:

```text
synthetic prompt rows analyzed: 206
unique prompts: 10
average prompt word count: 14.12
top keywords: natural, background, lighting, neutral, skin
```

Pháº§n nÃ y giÃºp liÃªn káº¿t prompt vÄƒn báº£n vá»›i dá»¯ liá»‡u áº£nh Ä‘Æ°á»£c sinh ra vÃ  hÃ nh vi cá»§a classifier.

## 9. Grad-CAM Explainability

Grad-CAM Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ trá»±c quan hÃ³a vÃ¹ng áº£nh cÃ³ áº£nh hÆ°á»Ÿng nhiá»u tá»›i quyáº¿t Ä‘á»‹nh cá»§a classifier.

VÃ­ dá»¥ output:

```text
results/gradcam/synthetic_0010_overlay.png
results/gradcam/real_0007_overlay.png
```

Nhá» Grad-CAM, dá»± Ã¡n khÃ´ng chá»‰ Ä‘Æ°a ra nhÃ£n `real/synthetic`, mÃ  cÃ²n giÃºp quan sÃ¡t vÃ¹ng áº£nh model táº­p trung khi dá»± Ä‘oÃ¡n.

## 10. Demo app

Dá»± Ã¡n cÃ³ app Gradio gá»“m hai tab chÃ­nh:

```text
Predict Image
Model Results
```

App cho phÃ©p upload áº£nh vÃ  hiá»ƒn thá»‹:

```text
xÃ¡c suáº¥t tá»«ng class
nhÃ£n dá»± Ä‘oÃ¡n
confidence
classification report
confusion matrix
```

Cháº¡y app:

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

## 11. Háº¡n cháº¿

Dá»± Ã¡n hiá»‡n lÃ  má»™t prototype end-to-end Ä‘Ã£ cháº¡y Ä‘Æ°á»£c, nhÆ°ng váº«n cÃ³ má»™t sá»‘ háº¡n cháº¿:

- Dataset cÃ²n nhá».
- áº¢nh synthetic Ä‘Æ°á»£c sinh tá»« má»™t mÃ´ hÃ¬nh Stable Diffusion.
- áº¢nh real chá»§ yáº¿u láº¥y tá»« LFW.
- Accuracy cao trÃªn dataset demo chÆ°a Ä‘áº£m báº£o model tá»•ng quÃ¡t tá»‘t trÃªn má»i áº£nh AI ngoÃ i thá»±c táº¿.
- Classifier cÃ³ thá»ƒ há»c khÃ¡c biá»‡t Ä‘áº·c trÆ°ng cá»§a dataset thay vÃ¬ artifact tá»•ng quÃ¡t cá»§a áº£nh synthetic.

## 12. HÆ°á»›ng phÃ¡t triá»ƒn

CÃ¡c hÆ°á»›ng cáº£i tiáº¿n:

- ThÃªm nhiá»u dataset áº£nh real hÆ¡n.
- Sinh áº£nh synthetic tá»« nhiá»u diffusion model khÃ¡c nhau.
- Test trÃªn dataset AI-generated/deepfake bÃªn ngoÃ i.
- TÄƒng Ä‘á»™ Ä‘a dáº¡ng cá»§a prompt.
- Thá»­ backbone máº¡nh hÆ¡n nhÆ° EfficientNet hoáº·c Vision Transformer.
- TÃ­ch há»£p Grad-CAM trá»±c tiáº¿p vÃ o Gradio app.
- Thá»­ LoRA hoáº·c DreamBooth nhÆ° pháº§n má»Ÿ rá»™ng nÃ¢ng cao cá»§a Stable Diffusion.

## 13. Káº¿t luáº­n

Dá»± Ã¡n Ä‘Ã£ xÃ¢y dá»±ng má»™t pipeline end-to-end cho bÃ i toÃ¡n sinh dá»¯ liá»‡u áº£nh synthetic vÃ  phÃ¡t hiá»‡n áº£nh AI-generated. Stable Diffusion Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ sinh áº£nh khuÃ´n máº·t synthetic, cÃ²n ResNet18 classifier Ä‘Æ°á»£c train Ä‘á»ƒ phÃ¢n biá»‡t áº£nh real vÃ  synthetic. Dá»± Ã¡n cÅ©ng bá»• sung NLP prompt analysis, external image prediction, Grad-CAM explainability vÃ  Gradio demo app.

