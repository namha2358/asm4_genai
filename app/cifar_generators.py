# app/cifar_generators.py — loads the trained diffusion model and EBM once and turns their samples into PNGs
import io
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from helper_lib.model import get_model                 # same architectures used in training
from helper_lib.generator import generate_samples      # class-activity sampling function

APP_DIR = Path(__file__).parent                        # locate weights relative to THIS file

# Diffusion model: UNet, EMA UNet and per-channel normalizer are all restored from one file
diffusion = get_model("Diffusion")
diffusion.load_state_dict(torch.load(APP_DIR / "diffusion_cifar10.pth", map_location="cpu", weights_only=True))
diffusion.eval()

# EBM: only the energy CNN was saved (the replay buffer is training-only)
ebm = get_model("EBM")
ebm.model.load_state_dict(torch.load(APP_DIR / "ebm_cifar10.pth", map_location="cpu", weights_only=True))
ebm.model.eval()


def _grid_to_png(grid: torch.Tensor, upscale: int = 4) -> bytes:
    """Convert a (3, H, W) grid in [0, 1] to PNG bytes, enlarged so the 32x32 images are easy to see."""
    image = TF.to_pil_image(grid)
    image = image.resize((image.width * upscale, image.height * upscale), Image.NEAREST)   # nearest = no blurring
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_diffusion_png(num_samples: int = 10, diffusion_steps: int = 50, seed: int | None = None) -> bytes:
    """Start from Gaussian noise and run reverse diffusion for diffusion_steps steps."""
    if seed is not None:
        torch.manual_seed(seed)                        # same seed -> same images
    grid = generate_samples(diffusion, device="cpu", num_samples=num_samples, diffusion_steps=diffusion_steps, show=False)
    return _grid_to_png(grid)


def generate_ebm_png(num_samples: int = 10, langevin_steps: int = 256, seed: int | None = None) -> bytes:
    """Start from uniform noise and run Langevin dynamics for langevin_steps steps to lower the energy."""
    if seed is not None:
        torch.manual_seed(seed)
    grid = generate_samples(ebm, device="cpu", num_samples=num_samples, langevin_steps=langevin_steps, show=False)
    return _grid_to_png(grid)