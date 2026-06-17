# Stable Diffusion for Data Generation

This project uses Stable Diffusion to generate synthetic image data for a computer vision task.

Initial task:

- Generate synthetic face images.
- Combine them with real face images.
- Train a classifier to distinguish `real` vs `synthetic`.

## Setup

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
