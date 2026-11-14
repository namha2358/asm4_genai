# helper_lib/data_loader.py — builds MNIST or CIFAR10 data loaders for the helper library models
import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, dataset="MNIST", scale="minus_one_one"):
    """
    Return a DataLoader over MNIST or CIFAR10.
    scale="minus_one_one": pixels in [-1, 1] (GAN and energy model, matching Tanh / Langevin clamping)
    scale="zero_one": pixels in [0, 1] (diffusion model, which normalizes per channel itself)
    """
    steps = [transforms.ToTensor()]                       # PIL image -> tensor (C, H, W), values in [0, 1]
    if scale == "minus_one_one":
        channels = 1 if dataset == "MNIST" else 3
        steps.append(transforms.Normalize((0.5,) * channels, (0.5,) * channels))   # (x - 0.5) / 0.5 -> [-1, 1]
    transform = transforms.Compose(steps)

    # Download on first use; train=True gives the training split
    if dataset == "MNIST":
        data = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    elif dataset == "CIFAR10":
        data = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: MNIST, CIFAR10")

    # Shuffle only the training split; drop_last keeps every batch full (the energy model's buffer expects that)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train, drop_last=train)