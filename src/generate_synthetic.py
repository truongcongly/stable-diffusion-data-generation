import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = Path("data/raw/synthetic")

PROMPTS = [
    "realistic portrait photo of an adult human face, natural lighting, neutral background, high detail, sharp focus",
    "realistic passport style portrait of a person, centered face, plain background, natural skin texture",
    "close up portrait photo of a human face, soft daylight, realistic eyes, detailed skin, neutral expression",
    "studio portrait photograph of a person, realistic face, balanced lighting, simple background",
]

NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, blurry, distorted face, extra eyes, "
    "bad anatomy, low quality, watermark, text, logo"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic face images with Stable Diffusion.")
    parser.add_argument("--count", type=int, default=20, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting image index.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--size", type=int, default=384, help="Image width and height.")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps per image.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Prompt guidance strength.")
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe, device = load_pipeline()
    pipe.enable_attention_slicing()

    for offset in tqdm(range(args.count), desc="Generating synthetic images"):
        image_index = args.start_index + offset
        prompt = PROMPTS[offset % len(PROMPTS)]
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

        output_path = args.output_dir / f"synthetic_{image_index:04d}.png"
        image.save(output_path)

    print(f"Saved {args.count} images to {args.output_dir}")


if __name__ == "__main__":
    main()
