# train_ebm.py — trains the energy-based model on CIFAR10 through helper_lib and saves its weights
import random
import numpy as np
import torch
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.generator import generate_samples

EPOCHS = 10                                  # same number of epochs as the class notebook
BATCH_SIZE = 128
WEIGHTS_PATH = "app/ebm_cifar10.pth"         # saved in app/ so the API can load it

# Fixed seeds: torch for weights and noise, random/numpy for the replay buffer's sampling
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR10 with pixels in [-1, 1], matching the [-1, 1] clamping in Langevin sampling
train_loader = get_data_loader("data", batch_size=BATCH_SIZE, train=True, dataset="CIFAR10", scale="minus_one_one")

# Build the EBM (energy CNN; alpha 0.1, 60 Langevin steps, step size 10, noise 0.005, as in the class notebook)
ebm = get_model("EBM")

# Adam with beta1 = 0, as in the class notebook
optimizer = optim.Adam(ebm.model.parameters(), lr=1e-4, betas=(0.0, 0.999))

train_ebm(ebm, train_loader, optimizer, device=device, epochs=EPOCHS)

# Save only the energy CNN; the replay buffer is only used during training
torch.save(ebm.model.state_dict(), WEIGHTS_PATH)
print(f"Saved EBM weights to {WEIGHTS_PATH}")

# Plot 10 generated images to check quality (close the plot window to end the script)
generate_samples(ebm, device=device, num_samples=10, langevin_steps=256)