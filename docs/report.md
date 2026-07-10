# Stable Diffusion for Synthetic Face Data Generation

## 1. Introduction

This project studies how Stable Diffusion can be used to generate synthetic image data for a computer vision task. The selected downstream task is real vs synthetic face image classification.

The project uses a pretrained Stable Diffusion model to generate synthetic face images from text prompts. These generated images are combined with real face images, then used to train a ResNet18 classifier that predicts whether an input face image is real or AI-generated.

## 2. Problem Statement

AI-generated face images are becoming increasingly realistic. This creates a practical need for models that can distinguish real face images from synthetic or AI-generated images.

The classification task is:

```text
Input: a face image
Output: real or synthetic
```

## 3. Project Scope

This project does not fine-tune Stable Diffusion. Stable Diffusion is used as a pretrained data generation model through the Hugging Face `diffusers` library.

The trained model in this project is the downstream classifier:

```text
Stable Diffusion: pretrained data generator
ResNet18: trained real vs synthetic classifier
```

## 4. Pipeline

```text
Text prompt
-> Stable Diffusion generates synthetic face images
-> Real face images are collected from LFW
-> Real and synthetic images are split into train/validation/test sets
-> ResNet18 classifier is trained
-> Model is evaluated on the test set
-> Gradio app demonstrates prediction
-> NLP prompt analysis studies the generation prompts
-> Grad-CAM visualizes model attention
```

## 5. Dataset

The dataset contains two classes:

```text
real
synthetic
```

Real images are collected from the LFW dataset using `scikit-learn`.

Synthetic images are generated using Stable Diffusion with multiple prompt templates. The project records metadata for each generated image, including:

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

The dataset preparation step balances the classes based on the smaller class count. In the current run:

```text
real images: 200
synthetic images used after balancing: 200
```

The split is:

```text
train: 140 real, 140 synthetic
validation: 30 real, 30 synthetic
test: 30 real, 30 synthetic
```

## 6. Model Training

The classifier is based on ResNet18.

Training configuration:

```text
model: ResNet18
classes: real, synthetic
image size: 224x224
optimizer: Adam
loss: CrossEntropyLoss
default epochs: 5
default batch size: 16
```

The training script saves:

```text
best checkpoint
final checkpoint
training history CSV
training config JSON
```

## 7. Evaluation

The evaluation script computes:

```text
accuracy
precision
recall
F1-score
confusion matrix
per-image prediction confidence
```

Current test result:

```text
accuracy: 1.0000
macro precision: 1.0000
macro recall: 1.0000
macro F1-score: 1.0000
test samples: 60
```

The confusion matrix is saved at:

```text
results/confusion_matrix.png
results/confusion_matrix_normalized.png
```

The per-image predictions are saved at:

```text
results/predictions.csv
```

## 8. NLP Prompt Analysis

The project includes a lightweight NLP analysis stage for the text prompts used to generate synthetic images.

The prompt analysis extracts:

```text
prompt word count
prompt keyword count
top keywords
keyword frequency
prompt length distribution
average classifier confidence by prompt ID
```

Current prompt analysis summary:

```text
synthetic prompt rows analyzed: 205
unique prompts: 10
average prompt word count: 14.11
top keywords: natural, background, lighting, neutral, skin
```

This connects the text generation prompts with the generated image dataset and classifier behavior.

## 9. Grad-CAM Explainability

Grad-CAM is used to visualize image regions that influenced the classifier decision.

Example outputs:

```text
results/gradcam/synthetic_0014_overlay.png
results/gradcam/real_0007_overlay.png
```

This helps make the classifier more interpretable by showing where the model focuses when predicting real or synthetic.

## 10. Demo Application

The project includes a Gradio app with two main tabs:

```text
Predict Image
Model Results
```

The app allows users to upload a face image and view:

```text
class probabilities
predicted label
confidence score
classification report
confusion matrix
```

Run the app:

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

## 11. Limitations

The current project is a working end-to-end prototype, but it has limitations:

- The dataset is still small.
- Synthetic images are generated from one Stable Diffusion model.
- Real images mainly come from LFW.
- High accuracy on this demo dataset may not generalize to all real-world AI-generated face images.
- The classifier may learn dataset-specific differences rather than universal synthetic image artifacts.

## 12. Future Work

Possible improvements:

- Add more real face datasets.
- Generate synthetic images from multiple diffusion models.
- Test on external AI-generated face datasets.
- Add more prompt diversity.
- Fine-tune a classifier with stronger backbones such as EfficientNet or ViT.
- Improve Grad-CAM integration inside the Gradio app.
- Add LoRA or DreamBooth fine-tuning as an advanced Stable Diffusion extension.

## 13. Conclusion

This project demonstrates an end-to-end workflow for synthetic data generation and AI-generated image detection. Stable Diffusion is used to generate synthetic face data, while a ResNet18 classifier is trained to detect real vs synthetic images. The project also adds NLP prompt analysis, external image prediction, Grad-CAM explainability, and an interactive Gradio demo.
