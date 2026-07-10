import importlib
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "PIL",
    "cv2",
    "sklearn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "gradio",
]

PROJECT_PATHS = [
    Path("src"),
    Path("data/raw/real"),
    Path("data/raw/synthetic"),
    Path("data/processed"),
    Path("models"),
    Path("results"),
]


def status(ok):
    return "OK" if ok else "MISSING"


def check_python():
    version = sys.version_info
    expected = version.major == 3 and version.minor == 11
    print(f"Python: {platform.python_version()} [{status(expected)}]")
    if not expected:
        print("  Expected Python 3.11 for best PyTorch CUDA compatibility.")


def check_packages():
    print("\nPackages:")
    missing = []

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"  {package}: {version} [OK]")
        except Exception as exc:
            missing.append(package)
            print(f"  {package}: {exc.__class__.__name__} [MISSING]")

    if missing:
        print("\nInstall missing packages with:")
        print("  python -m pip install -r requirements.txt")


def check_torch_cuda():
    print("\nPyTorch CUDA:")
    try:
        import torch
    except Exception as exc:
        print(f"  torch import failed: {exc}")
        return

    print(f"  torch version: {torch.__version__}")
    print(f"  torch CUDA version: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")

    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM GB: {total_vram:.2f}")
        try:
            tensor = torch.rand(1, device="cuda")
            print(f"  CUDA tensor test: {tensor} [OK]")
        except Exception as exc:
            print(f"  CUDA tensor test failed: {exc}")
    else:
        print("  CUDA is not available. Stable Diffusion and training will run slowly on CPU.")
        print("  Check NVIDIA driver, PyTorch CUDA build, and whether the NVIDIA GPU is enabled.")


def check_project_paths():
    print("\nProject folders:")
    for path in PROJECT_PATHS:
        exists = path.exists()
        print(f"  {path}: {status(exists)}")


def count_images(folder):
    if not folder.exists():
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in extensions)


def check_dataset_outputs():
    print("\nDataset and outputs:")
    real_count = count_images(Path("data/raw/real"))
    synthetic_count = count_images(Path("data/raw/synthetic"))
    model_path = Path("models/resnet18_real_vs_synthetic.pth")
    report_path = Path("results/classification_report.txt")
    matrix_path = Path("results/confusion_matrix.png")

    print(f"  real images: {real_count}")
    print(f"  synthetic images: {synthetic_count}")
    print(f"  trained model: {status(model_path.exists())}")
    print(f"  classification report: {status(report_path.exists())}")
    print(f"  confusion matrix: {status(matrix_path.exists())}")

    if real_count == 0 or synthetic_count == 0:
        print("  Add/generate images before preparing the dataset and training.")


def main():
    print("Environment Check")
    print("=================")
    check_python()
    check_packages()
    check_torch_cuda()
    check_project_paths()
    check_dataset_outputs()


if __name__ == "__main__":
    main()
