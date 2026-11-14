# helper_lib/generator.py — sample new images from a trained GAN, diffusion model or EBM and plot them on a grid
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .diffusion import DiffusionModel
from .energy import EBM


def generate_samples(model, device='cpu', num_samples=10, diffusion_steps=100, langevin_steps=256, show=True):
    """
    Generate num_samples images; plot the grid if show=True and return it.
    - DiffusionModel: start from Gaussian noise and run reverse diffusion for diffusion_steps steps
    - EBM: start from uniform noise and run Langevin dynamics for langevin_steps steps
    - GAN (or a bare Generator): run the generator on Gaussian noise vectors
    """
    if isinstance(model, DiffusionModel):
        model.to(device)
        images = model.generate(num_samples, diffusion_steps).cpu()           # already in [0, 1]
        title = f"{num_samples} diffusion samples ({diffusion_steps} steps)"
    elif isinstance(model, EBM):
        model.to(device)
        images = model.generate(num_samples, steps=langevin_steps).cpu()      # already in [0, 1]
        title = f"{num_samples} energy-model samples ({langevin_steps} Langevin steps)"
    else:
        gen = model.generator if hasattr(model, "generator") else model       # accept the GAN wrapper or a bare Generator
        gen.to(device)
        gen.eval()                                                            # eval mode: BatchNorm uses running statistics
        with torch.no_grad():
            noise = torch.randn(num_samples, gen.z_dim, device=device)        # points in the latent space
            images = (gen(noise).cpu() + 1) / 2                               # [-1, 1] -> [0, 1]
        title = f"{num_samples} generated digits"

    # Arrange the images in one grid, at most 10 per row
    grid = make_grid(images, nrow=min(num_samples, 10))

    if show:
        plt.imshow(grid.permute(1, 2, 0))        # imshow expects height x width x channels
        plt.axis("off")
        plt.title(title)
        plt.show()
    return grid