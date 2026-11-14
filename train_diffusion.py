# train_diffusion.py — trains the diffusion model on CIFAR10 through helper_lib and saves its weights
import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion, compute_normalizer
from helper_lib.generator import generate_samples

EPOCHS = 10                                  # full passes over the 50,000 CIFAR10 training images
BATCH_SIZE = 128
WEIGHTS_PATH = "app/diffusion_cifar10.pth"   # saved in app/ so the API can load it

torch.manual_seed(42)                        # fixed seed so reruns start from the same random weights

# Use the Mac GPU (MPS) when available, otherwise CUDA, otherwise CPU, as in the class notebooks
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR10 with pixels in [0, 1]; the model normalizes each color channel itself
train_loader = get_data_loader("data", batch_size=BATCH_SIZE, train=True, dataset="CIFAR10", scale="zero_one")

# Build the diffusion model (UNet + EMA copy) from the helper library
diffusion = get_model("Diffusion")

# Per-channel mean and std of the training images, stored inside the model and saved with the weights
mean, std = compute_normalizer(train_loader)
diffusion.set_normalizer(mean, std)
print("Per-channel mean:", mean, "| std:", std)

# L1 loss between predicted and true noise, AdamW optimizer, as in the class notebook
criterion = nn.L1Loss()
optimizer = optim.AdamW(diffusion.network.parameters(), lr=1e-3, weight_decay=1e-4)

train_diffusion(diffusion, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

# Save the whole model state: UNet, EMA UNet and normalizer
torch.save(diffusion.state_dict(), WEIGHTS_PATH)
print(f"Saved diffusion weights to {WEIGHTS_PATH}")

# Plot 10 generated images to check quality (close the plot window to end the script)
generate_samples(diffusion, device=device, num_samples=10, diffusion_steps=100)