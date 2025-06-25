import os
import runpod
import subprocess
import uuid
import logging
import shutil
from diffusers import DiffusionPipeline
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RunPod volume paths
RUNPOD_VOLUME_PATH = "/runpod-volume"
MODEL_DIR = os.path.join(RUNPOD_VOLUME_PATH, "flux", "checkpoints")
TEMP_DIR = os.path.join(RUNPOD_VOLUME_PATH, "flux", "temp")
OUTPUT_DIR = "/app/output"
CHECKPOINTS_DIR = "/app/checkpoints"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Flux.1 model details
FLUX_MODEL_PATH = os.path.join(
    CHECKPOINTS_DIR, "flux1-pruned-fp32.safetensors")
FLUX_RUNPOD_MODEL_PATH = os.path.join(
    MODEL_DIR, "flux1-pruned-fp32.safetensors")
CIVITAI_URL = "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=pruned&fp=fp32"


def download_model():
    """Download Flux.1 model if not present"""
    if os.path.exists(FLUX_MODEL_PATH):
        logger.info("Using model from /app/checkpoints")
        return FLUX_MODEL_PATH
    elif os.path.exists(FLUX_RUNPOD_MODEL_PATH):
        logger.info("Using model from /runpod-volume/checkpoints")
        return FLUX_RUNPOD_MODEL_PATH
    else:
        logger.info("Model not found, downloading from CivitAI...")
        api_token = os.getenv("CIVITAI_API_TOKEN")
        if not api_token:
            raise ValueError("CIVITAI_API_TOKEN environment variable not set")
        try:
            subprocess.run([
                "wget", "--header", f"Authorization: Bearer {api_token}",
                CIVITAI_URL, "-O", FLUX_MODEL_PATH
            ], check=True)
            logger.info("Model downloaded successfully")
            return FLUX_MODEL_PATH
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading model: {e}")
            raise


def run_flux(prompt, output_path):
    """Run Flux.1 inference to generate an image from a text prompt"""
    try:
        # Load model
        model_path = download_model()
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,  # Optimize for VRAM
            use_safetensors=True,
        )
        pipe.load_lora_weights(model_path)

        # Move to GPU with optimizations
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()  # Reduce VRAM usage
        else:
            logger.warning("GPU not available, using CPU")

        # Generate image
        logger.info(f"Generating image with prompt: {prompt}")
        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

        # Save image
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error running Flux.1: {str(e)}")
        return False


def handler(job):
    """RunPod handler function"""
    job_input = job["input"]

    # Get input prompt
    prompt = job_input.get("prompt")
    if not prompt:
        return {
            "error": "Missing required input: prompt must be provided"
        }

    # Create temp directory with UUID to avoid collisions
    job_id = job.get("id", str(uuid.uuid4()))
    unique_id = str(uuid.uuid4())
    work_dir = os.path.join(TEMP_DIR, f"job_{job_id}_{unique_id}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Set output path
        output_path = os.path.join(OUTPUT_DIR, f"output_image_{unique_id}.png")

        # Run Flux.1 inference
        if not run_flux(prompt, output_path):
            return {"error": "Flux.1 inference failed"}

        # Return success with output path
        return {
            "status": "success",
            "output_image_path": output_path
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        # Clean up
        try:
            shutil.rmtree(work_dir)
        except Exception as cleanup_error:
            logger.warning(
                f"Error cleaning up temporary files: {str(cleanup_error)}")


# Start the RunPod handler
runpod.serverless.start({"handler": handler})
