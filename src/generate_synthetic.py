import argparse
import csv
import re
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = Path("data/raw/synthetic")
DEFAULT_METADATA_PATH = Path("data/metadata/synthetic_prompts.csv")

PROMPTS = [
    "realistic portrait photo of an adult human face, natural lighting, neutral background, high detail, sharp focus",
    "realistic passport style portrait of a person, centered face, plain background, natural skin texture",
    "close up portrait photo of a human face, soft daylight, realistic eyes, detailed skin, neutral expression",
    "studio portrait photograph of a person, realistic face, balanced lighting, simple background",
    "realistic phone selfie portrait of a person, casual indoor lighting, natural expression, detailed face",
    "outdoor portrait photo of a human face, daylight, shallow depth of field, realistic skin texture",
    "professional headshot photo of a person, soft studio lighting, clean background, sharp facial details",
    "realistic face portrait of an older adult, natural wrinkles, neutral background, documentary photography",
    "realistic portrait of a young adult, natural lighting, slight smile, simple background, high detail",
    "low light realistic portrait of a person, soft shadows, natural face, photographic style",
]

NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, blurry, distorted face, extra eyes, "
    "bad anatomy, low quality, watermark, text, logo"
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_index = args.start_index or find_next_index(args.output_dir)

    pipe, device = load_pipeline()
    pipe.enable_attention_slicing()

    metadata_rows = []

    for offset in tqdm(range(args.count), desc="Generating synthetic images"):
        image_index = start_index + offset
        prompt_id = offset % len(PROMPTS)
        prompt = PROMPTS[prompt_id]
        output_path = args.output_dir / f"synthetic_{image_index:04d}.png"

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        generator = torch.Generator(device=device).manual_seed(args.seed + offset)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=args.size,
            height=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        image.save(output_path)
        metadata_rows.append(
            {
                "image_path": output_path.as_posix(),
                "label": "synthetic",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed": args.seed + offset,
                "width": args.size,
                "height": args.size,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "model_id": MODEL_ID,
            }
        )

    if metadata_rows:
        append_metadata(args.metadata_path, metadata_rows)

    print(f"Saved {len(metadata_rows)} images to {args.output_dir}")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
