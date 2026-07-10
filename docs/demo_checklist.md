# Demo Checklist

## Before Demo

Run the environment check:

```cmd
D:\project\.venv\Scripts\python.exe src\check_environment.py
```

Expected:

```text
Python: 3.11.x [OK]
CUDA available: True
trained model: OK
classification report: OK
confusion matrix: OK
```

## Demo Flow

1. Explain the project scope.

```text
Stable Diffusion is used for data generation.
The trained model is the ResNet18 classifier.
```

2. Show synthetic generation.

```cmd
D:\project\.venv\Scripts\python.exe src\generate_synthetic.py --count 1
```

3. Show dataset split summary.

```text
data\metadata\dataset_summary.txt
```

4. Show training command.

```cmd
D:\project\.venv\Scripts\python.exe src\train_model.py --epochs 5 --batch-size 16
```

5. Show evaluation results.

```cmd
D:\project\.venv\Scripts\python.exe src\evaluate_model.py
```

6. Run the app.

```cmd
D:\project\.venv\Scripts\python.exe src\app.py
```

7. Show NLP prompt analysis.

```cmd
D:\project\.venv\Scripts\python.exe src\prompt_analysis.py
```

8. Show Grad-CAM.

```cmd
D:\project\.venv\Scripts\python.exe src\gradcam.py --image data\processed\test\synthetic\synthetic_0014.png --device cpu
```

## Key Explanation

```text
This project uses Stable Diffusion to generate synthetic face images. These images are combined with real face images to train a ResNet18 classifier for detecting real vs AI-generated faces. The project also includes NLP prompt analysis and Grad-CAM explainability.
```

## Important Limitation

```text
The current accuracy is high on the demo dataset, but the dataset is small and should be tested on more diverse real-world images before claiming general performance.
```
