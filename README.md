# Stable Diffusion for Synthetic Face Data Generation

This project uses Stable Diffusion to generate synthetic face images for a computer vision task: detecting whether a face image is real or AI-generated.

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
