import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = Path("data/raw/synthetic")
DEFAULT_METADATA_PATH = Path("data/metadata/synthetic_prompts.csv")

PROMPTS = [
    "low resolution color photograph of one real human face, LFW dataset style, cropped face, natural skin, ordinary lighting",
    "realistic low resolution face photo of one adult person, centered face crop, natural expression, plain background",
    "real candid headshot photo of one person, face fills most of the image, natural indoor lighting, LFW style",
    "realistic celebrity face crop photo, one person only, low resolution jpeg, natural face, simple background",
    "realistic passport-like face crop of one adult person, ordinary camera photo, natural colors, clear face",
    "realistic webcam face photo of one adult person, low resolution, centered face, neutral background",
    "realistic ID style face photograph of one person, cropped close to face, natural skin texture, plain background",
    "realistic face crop photo of one older adult, low resolution jpeg, natural wrinkles, ordinary lighting",
    "realistic face crop photo of one young adult, low resolution jpeg, natural expression, plain background",
    "realistic news photo face crop of one adult person, low resolution, natural lighting, face clearly visible",
]

AGES = [
    "young adult",
    "middle-aged adult",
    "elderly adult",
    "adult in their 30s",
    "adult in their 50s",
]

GENDERS = [
    "male",
    "female",
    "gender-neutral person",
]

FACE_ANGLES = [
    "front-facing face",
    "three-quarter face angle",
    "looking slightly left",
    "looking slightly right",
    "close-up face",
    "cropped headshot",
]

EXPRESSIONS = [
    "neutral expression",
    "slight smile",
    "wide smile",
    "serious expression",
    "surprised expression",
    "thoughtful expression",
    "relaxed expression",
    "laughing expression",
]

LIGHTING_STYLES = [
    "natural daylight",
    "soft indoor lighting",
    "office lighting",
    "ordinary room lighting",
    "news photo lighting",
]

BACKGROUNDS = [
    "neutral background",
    "plain background",
    "simple indoor background",
    "blurred background",
]

ACCESSORIES = [
    "",
    "wearing glasses",
    "without glasses",
    "casual clothing",
    "professional clothing",
]

NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, drawing, artwork, stylized, glamour, fashion, "
    "oversaturated, neon colors, colorful lighting, distorted face, extra eyes, bad anatomy, "
    "watermark, text, logo, abstract pattern, mosaic, grid, architecture, object, landscape, "
    "black image, blank image, duplicate face, multiple people, mask, helmet, mannequin, "
    "plastic skin, doll face, deformed, surreal, face paint, heavy makeup"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic face images with Stable Diffusion.")
    parser.add_argument("--count", type=int, default=20, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="CSV file for generated image metadata.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting image index. If omitted, the next available index is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--size", type=int, default=384, help="Image width and height.")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps per image.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Prompt guidance strength.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per image when output fails quality checks.")
    parser.add_argument("--require-face", action="store_true", help="Retry images when OpenCV cannot detect a face.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
    return parser.parse_args()


def load_pipeline():
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        print("Using CUDA GPU")
        return pipe, "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    print("Using CPU")
    return pipe, "cpu"


def find_next_index(output_dir):
    pattern = re.compile(r"synthetic_(\d+)\.png$")
    existing_indices = []

    for path in output_dir.glob("synthetic_*.png"):
        match = pattern.match(path.name)
        if match:
            existing_indices.append(int(match.group(1)))

    if not existing_indices:
        return 1
    return max(existing_indices) + 1


def append_metadata(metadata_path, rows):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metadata_path.exists()

    fieldnames = [
        "image_path",
        "label",
        "prompt_id",
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "model_id",
    ]

    with metadata_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def pick(items, index, multiplier):
    return items[(index * multiplier) % len(items)]


def build_prompt(index):
    base_prompt = PROMPTS[index % len(PROMPTS)]
    attributes = [
        pick(AGES, index, 3),
        pick(GENDERS, index, 5),
        pick(FACE_ANGLES, index, 7),
        pick(EXPRESSIONS, index, 11),
        pick(LIGHTING_STYLES, index, 13),
        pick(BACKGROUNDS, index, 17),
        pick(ACCESSORIES, index, 19),
        "realistic skin texture",
        "single person",
        "face clearly visible",
        "photo-realistic",
    ]
    return ", ".join(part for part in [base_prompt, *attributes] if part)


def is_black_or_blank(image):
    image_array = np.asarray(image.convert("RGB"))
    brightness = image_array.mean()
    contrast = image_array.std()
    return brightness < 8 or contrast < 5


def has_detectable_face(image):
    image_array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    image_height, image_width = gray.shape
    cascades = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_profileface.xml",
    ]

    for cascade_path in cascades:
        classifier = cv2.CascadeClassifier(cascade_path)
        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
        for x, y, width, height in faces:
            face_area_ratio = (width * height) / float(image_width * image_height)
            face_center_x = x + width / 2
            face_center_y = y + height / 2
            centered_x = image_width * 0.2 <= face_center_x <= image_width * 0.8
            centered_y = image_height * 0.15 <= face_center_y <= image_height * 0.85
            if face_area_ratio >= 0.08 and centered_x and centered_y:
                return True
    return False


def passes_quality_checks(image, require_face):
    if is_black_or_blank(image):
        return False, "black_or_blank"
    if require_face and not has_detectable_face(image):
        return False, "no_face_detected"
    return True, "ok"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_index = args.start_index or find_next_index(args.output_dir)

    pipe, device = load_pipeline()
    pipe.enable_attention_slicing()

    metadata_rows = []

    saved_count = 0
    skipped_count = 0

    for offset in tqdm(range(args.count), desc="Generating synthetic images"):
        image_index = start_index + offset
        prompt_id = offset % len(PROMPTS)
        prompt = build_prompt(image_index)
        output_path = args.output_dir / f"synthetic_{image_index:04d}.png"

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        image = None
        quality_status = "not_generated"
        seed = args.seed + offset

        for retry in range(args.max_retries + 1):
            seed = args.seed + offset + retry * 100000
            generator = torch.Generator(device=device).manual_seed(seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            quality_ok, quality_status = passes_quality_checks(image, args.require_face)
            if quality_ok:
                break

        if image is None or quality_status != "ok":
            skipped_count += 1
            print(f"Skipping synthetic_{image_index:04d}.png because quality check failed: {quality_status}")
            continue

        image.save(output_path)
        saved_count += 1
        metadata_rows.append(
            {
                "image_path": output_path.as_posix(),
                "label": "synthetic",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed": seed,
                "width": args.size,
                "height": args.size,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "model_id": MODEL_ID,
            }
        )

    if metadata_rows:
        append_metadata(args.metadata_path, metadata_rows)

    print(f"Saved {saved_count} images to {args.output_dir}")
    print(f"Skipped {skipped_count} low-quality images")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
