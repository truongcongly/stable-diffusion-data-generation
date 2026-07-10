# Stable Diffusion for Synthetic Face Data Generation

This project uses Stable Diffusion to generate synthetic face images for a computer vision task: detecting whether a face image is real or AI-generated.

## Portfolio Highlights

- End-to-end synthetic data generation pipeline with Stable Diffusion.
- Real vs synthetic face image classifier using ResNet18.
- Evaluation with accuracy, precision, recall, F1-score, confusion matrix, and per-image confidence.
- NLP prompt analysis for generated synthetic image prompts.
- External image prediction workflow.
- Grad-CAM explainability for model predictions.
- Interactive Gradio demo app.

Project documents:

- [Project Report](docs/report.md)
- [Demo Checklist](docs/demo_checklist.md)

## Project Overview

The main idea is to use a pretrained text-to-image diffusion model as a data generation tool. Stable Diffusion generates synthetic face images from text prompts. These generated images are then combined with real face images to create a labeled dataset for training an image classifier.

Pipeline:

```text
Text prompt
-> Stable Diffusion generates synthetic face images
-> Real and synthetic images are combined into a dataset
-> A ResNet18 classifier is trained
-> The model predicts real vs synthetic/AI-generated faces
-> A Gradio app demonstrates the prediction result
```

## Problem Statement

AI-generated face images are increasingly realistic, which makes it useful to study whether a computer vision model can distinguish real face images from synthetic ones. This project focuses on building an end-to-end synthetic data generation and classification pipeline.

The classification task is:

```text
Input: a face image
Output: real or synthetic
```

## Project Scope

This project does not fine-tune Stable Diffusion. Instead, it uses a pretrained Stable Diffusion model from Hugging Face through the `diffusers` library to generate synthetic data.

The model trained in this project is the downstream classifier:

```text
Stable Diffusion: used as a pretrained data generator
ResNet18: trained as the real vs synthetic image classifier
```

## Objectives

- Generate synthetic face images from text prompts.
- Collect or download real face images.
- Build a real vs synthetic image dataset.
- Train a ResNet18 classifier.
- Evaluate the model with accuracy, precision, recall, F1-score, and a confusion matrix.
- Provide a Gradio demo app for uploading an image and predicting whether it is real or synthetic.

## Demo Explanation

When presenting this project, the key point is:

```text
Stable Diffusion is used to generate image data.
The trained model is the classifier, not Stable Diffusion.
```

Suggested explanation:

```text
This project uses a pretrained Stable Diffusion model to generate synthetic face images from text prompts. These generated images are combined with real face images to create a dataset. A ResNet18 classifier is then trained to distinguish real images from AI-generated images.
```

## Setup

### Requirements Checklist

Before running the project, make sure these are available:

- Python 3.11
- Git
- NVIDIA GPU driver
- PyTorch with CUDA 12.1
- Internet connection for the first Stable Diffusion model download and LFW dataset download
- Visual Studio Code, optional but recommended

Check Python:

```cmd
D:\project\.venv\Scripts\python.exe --version
```

Expected:

```text
Python 3.11.x
```

Check PyTorch and CUDA:

```cmd
D:\project\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```text
2.x.x+cu121
True
```

The virtual environment is located in the outer project folder:

```cmd
cd /d D:\project\stable-diffusion-data-generation
..\.venv\Scripts\activate
```

If activation is confusing, run commands with the virtual environment Python directly:

```cmd
D:\project\.venv\Scripts\python.exe
```

Install PyTorch with CUDA first:

```cmd
D:\project\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest of the dependencies:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Check GPU

```cmd
D:\project\.venv\Scripts\python.exe src\check_gpu.py
```

Expected result:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Stage 2: Environment Verification

After installing Python, PyTorch, and the project dependencies, run the full environment check:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

This checks:

- Python version
- Required Python packages
- PyTorch CUDA support
- GPU name and VRAM
- Project folders
- Number of real and synthetic images
- Trained model and evaluation outputs

Expected important lines:

```text
Python: 3.11.x [OK]
torch: ... [OK]
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

If `CUDA available` is `False`, check these items:

- Make sure the NVIDIA GPU driver is installed.
- Restart the computer after installing or updating the driver.
- Make sure PyTorch was installed with the CUDA 12.1 wheel:

```cmd
D:\project\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Run the check from a normal Command Prompt, not only from the IDE terminal:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

## Generate One Test Image

```cmd
D:\project\.venv\Scripts\python.exe src\generate_one_image.py
```

The first run downloads the Stable Diffusion model, so it can take a while.

Output:

```text
data\raw\synthetic\test_face.png
```

## Stage 3: Synthetic Image Generation

This stage creates synthetic face images with Stable Diffusion and records generation metadata for later analysis.

Run a small generation test first:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 5
```

The script saves images to:

```text
data\raw\synthetic
```

It also saves metadata to:

```text
data\metadata\synthetic_prompts.csv
```

The metadata file contains:

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

This metadata is important for the next NLP stage because it connects each generated image with the text prompt that created it.

To generate the final synthetic dataset:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 200
```

By default, the script uses the next available image index. This avoids overwriting existing files such as `synthetic_0001.png`.

If you intentionally want to start from a specific index:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 20 --start-index 1 --overwrite
```

## Stage 4: Real Image Dataset Collection

This stage prepares the real face image class. The current workflow uses the LFW dataset through `scikit-learn`.

Run a small test first:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 5
```

The script saves images to:

```text
data\raw\real
```

It also saves metadata to:

```text
data\metadata\real_lfw.csv
```

The metadata file contains:

```text
image_path
label
source_dataset
lfw_index
person_name
width
height
```

To create a real image dataset balanced with 200 synthetic images:

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200
```

By default, the script uses the next available image index. This avoids overwriting existing files such as `real_0001.jpg`.

If you intentionally want to start from a specific index:

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200 --start-index 1 --overwrite
```

If the LFW download fails because of network access, manually place real face images in:

```text
data\raw\real
```

Then continue with dataset preparation.

## Stage 5: Dataset Preparation

This stage converts the raw real/synthetic image folders into a machine learning dataset with train, validation, and test splits.

Run:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
```

Input folders:

```text
data\raw\real
data\raw\synthetic
```

Output folders:

```text
data\processed\train\real
data\processed\train\synthetic
data\processed\val\real
data\processed\val\synthetic
data\processed\test\real
data\processed\test\synthetic
```

The script also saves:

```text
data\metadata\dataset_split_manifest.csv
data\metadata\dataset_summary.txt
```

The manifest records each image path, label, and split. The summary records how many real and synthetic images are used in each split.

By default, the script balances the dataset to the smaller class count. For example, if there are 200 real images and 345 synthetic images, it uses 200 images from each class. This avoids training a biased classifier.

Default split ratios:

```text
train: 70%
validation: 15%
test: 15%
```

To use all available images without class balancing:

```cmd
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py --no-balance
```

To change split ratios:

```cmd
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py --train-ratio 0.8 --val-ratio 0.1
```

## Stage 6: Classifier Training

This stage trains the downstream computer vision model. Stable Diffusion is not trained in this project; it is only used to generate synthetic images. The trained model is a ResNet18 classifier for `real` vs `synthetic` face images.

Run:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

Input:

```text
data\processed\train
data\processed\val
```

Outputs:

```text
models\resnet18_real_vs_synthetic.pth
models\resnet18_real_vs_synthetic_final.pth
results\training_history.csv
results\training_config.json
```

The best model checkpoint is saved based on validation accuracy:

```text
models\resnet18_real_vs_synthetic.pth
```

The final checkpoint after the last epoch is saved as:

```text
models\resnet18_real_vs_synthetic_final.pth
```

The training history contains:

```text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
```

For a quick test run:

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 1 --batch-size 16
```

Optional: use ImageNet pretrained ResNet18 weights:

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16 --pretrained
```

Suggested demo explanation:

```text
In this stage, the generated synthetic images and real images are used to train a ResNet18 classifier. The model learns to classify each input face image as real or synthetic.
```

## Stage 7: Model Evaluation

This stage evaluates the trained classifier on the test split.

Run:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Input:

```text
data\processed\test
models\resnet18_real_vs_synthetic.pth
```

Outputs:

```text
results\classification_report.txt
results\classification_report.json
results\metrics_summary.json
results\predictions.csv
results\confusion_matrix.png
results\confusion_matrix_normalized.png
```

The report includes:

```text
accuracy
precision
recall
F1-score
support
```

The predictions CSV contains one row per test image:

```text
image_path
true_label
predicted_label
confidence
correct
```

This file is useful for error analysis and for future prompt/NLP analysis. For example, synthetic image predictions can later be connected back to the prompt metadata from Stage 3.

Suggested demo explanation:

```text
In this stage, the trained classifier is evaluated on unseen test images. The project reports standard classification metrics and a confusion matrix to show how well the model distinguishes real and synthetic face images.
```

## Stage 8: Interactive Demo App

This stage runs a Gradio app for live demonstration.

Run:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\app.py
```

Open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

The app includes:

```text
Predict Image tab:
- Upload a face image
- Show real/synthetic probabilities
- Show predicted label and confidence

Model Results tab:
- Show model path and device
- Show validation/test metrics
- Show classification report
- Show confusion matrix
```

Before running the app, make sure these files exist:

```text
models\resnet18_real_vs_synthetic.pth
results\metrics_summary.json
results\classification_report.txt
results\confusion_matrix.png
```

If result files are missing, run:

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Suggested demo flow:

```text
1. Open the app.
2. Upload a real image from data\processed\test\real.
3. Show the predicted label and confidence.
4. Upload a synthetic image from data\processed\test\synthetic.
5. Open the Model Results tab and explain the metrics/confusion matrix.
```

## Stage 9: NLP Prompt Analysis

This stage adds a lightweight natural language processing analysis for the prompts used to generate synthetic images.

Run:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

Inputs:

```text
data\metadata\synthetic_prompts.csv
results\predictions.csv
```

Outputs:

```text
results\prompt_analysis\prompt_analysis.csv
results\prompt_analysis\prompt_analysis_report.txt
results\prompt_analysis\prompt_analysis_summary.json
results\prompt_analysis\prompt_keyword_frequency.png
results\prompt_analysis\prompt_length_distribution.png
results\prompt_analysis\confidence_by_prompt_id.png
```

The analysis extracts simple NLP features:

```text
prompt word count
prompt keyword count
top prompt keywords
keyword frequency
prompt length distribution
average classifier confidence by prompt ID
```

This stage connects the text prompts to the generated images and model predictions. It helps answer questions such as:

```text
Which words appear most often in the generated image prompts?
Are some prompt types easier for the classifier to detect as synthetic?
How long are the prompts used to generate the synthetic dataset?
```

Only synthetic images that appear in the test split can be matched with rows from `results\predictions.csv`. The remaining generated images are still included in keyword and prompt-length analysis.

Suggested demo explanation:

```text
This project also includes a simple NLP analysis of the text prompts used to generate synthetic images. The goal is to connect language prompts with image generation and classifier behavior.
```

## Stage 10: External Image Prediction

This stage predicts real vs synthetic for images outside the training workflow. It is useful for a practical demo because you can test images manually selected from other folders.

Predict one image:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png
```

Example output:

```text
data/processed/test/synthetic/synthetic_0014.png -> synthetic (99.99%)
```

Predict all images in a folder:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\real
```

Save predictions to CSV:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\processed\test\synthetic --output-csv results\external_predictions.csv
```

If CUDA/cuDNN has a temporary memory issue during live demo, force CPU prediction:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

For a more realistic external test, create folders such as:

```text
data\external_test\real
data\external_test\synthetic
```

Then run:

```cmd
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\external_test\real --output-csv results\external_real_predictions.csv
D:\project\.venv\Scripts\python.exe src\predict_image.py --image-dir data\external_test\synthetic --output-csv results\external_synthetic_predictions.csv
```

Suggested demo explanation:

```text
Besides the fixed test set, the project can classify external images using the trained model. This makes the demo closer to a real-world use case.
```

## Stage 11: Grad-CAM Explainability

This stage adds explainable AI visualization with Grad-CAM. Grad-CAM highlights the image regions that most influenced the classifier prediction.

Generate Grad-CAM for a synthetic test image:

```cmd
cd /d D:\project\stable-diffusion-data-generation
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png
```

Generate Grad-CAM for a real test image:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\real\real_0007.jpg
```

Outputs:

```text
results\gradcam\<image_name>_heatmap.png
results\gradcam\<image_name>_overlay.png
results\gradcam\<image_name>_gradcam.json
```

If CUDA/cuDNN has a temporary memory issue during live demo, force CPU:

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

Suggested demo explanation:

```text
Grad-CAM helps explain the classifier decision by visualizing which image regions contributed most to the real/synthetic prediction. This makes the model behavior easier to inspect instead of only showing a label and confidence score.
```

## Stage 12: Final Report and Portfolio Documentation

This stage prepares the project for submission, presentation, or portfolio use.

Main documents:

```text
docs\report.md
docs\demo_checklist.md
```

The report covers:

```text
introduction
problem statement
project scope
pipeline
dataset
model training
evaluation
NLP prompt analysis
Grad-CAM explainability
demo application
limitations
future work
```

Before submission, run:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

Recommended screenshots for presentation:

```text
Gradio app prediction screen
confusion matrix
prompt keyword frequency chart
Grad-CAM overlay image
dataset summary
```

## Full Demo Pipeline

### 1. Generate Synthetic Images

Start with a small number first:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 20
```

For the final demo, generate more images:

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 200
```

Output:

```text
data\raw\synthetic
```

### 2. Download Real Face Images

```cmd
D:\project\.venv\Scripts\python.exe src\download_real_lfw.py --count 200
```

Output:

```text
data\raw\real
```

### 3. Prepare Train/Validation/Test Dataset

```cmd
D:\project\.venv\Scripts\python.exe src\prepare_dataset.py
```

Output:

```text
data\processed\train
data\processed\val
data\processed\test
```

### 4. Train the Classifier

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

Output:

```text
models\resnet18_real_vs_synthetic.pth
```

### 5. Evaluate the Model

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

Output:

```text
results\classification_report.txt
results\confusion_matrix.png
```

### 6. Run the Demo App

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

The app lets you upload an image and predicts whether it is `real` or `synthetic`.
