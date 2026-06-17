from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_PATH = Path("data/raw/synthetic/test_face.png")


def load_pipeline():
    if torch.cuda.is_available():
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")
            print("Using CUDA GPU")
            return pipe, "cuda"
        except Exception as exc:
            print("CUDA is available, but the model could not use the GPU.")
            print(f"CUDA error: {exc}")
            print("Falling back to CPU. This will be much slower.")
            torch.cuda.empty_cache()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cpu")
    print("Using CPU")
    return pipe, "cpu"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipe, device = load_pipeline()
    pipe.enable_attention_slicing()

    prompt = (
        "realistic portrait photo of a human face, natural lighting, "
        "neutral background, high detail, sharp focus"
    )
    negative_prompt = (
        "cartoon, anime, painting, blurry, distorted face, extra eyes, "
        "bad anatomy, low quality"
    )

    generator = torch.Generator(device=device).manual_seed(42)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=384,
        height=384,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
