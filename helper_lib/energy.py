# helper_lib/energy.py — energy-based model, adapted from Module_8_Practical_1_EnergyBasedMethods.py
import random
import numpy as np
import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)                      # swish activation, as in the class notebook


class EnergyModel(nn.Module):
    """CNN that maps a 32x32 image to one energy value: low for realistic images, high for fakes."""

    def __init__(self, num_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)   # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)             # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)             # 8x8 -> 4x4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)             # 4x4 -> 2x2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)                                                    # one energy per image

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def langevin_sample(energy_model, inp_imgs, steps, step_size, noise_std):
    """Langevin dynamics: gradient descent on the INPUT images (not the weights) to lower their energy."""
    energy_model.eval()
    for p in energy_model.parameters():
        p.requires_grad_(False)                      # weights are fixed while we change the images

    for _ in range(steps):
        # Add a little noise; no gradient tracking needed for this step
        with torch.no_grad():
            inp_imgs = (inp_imgs + torch.randn_like(inp_imgs) * noise_std).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)                # track gradients with respect to the pixels
        energy = energy_model(inp_imgs)
        # d(energy)/d(pixels) for every image in the batch
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)          # gradient clipping keeps sampling stable
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)   # step downhill in energy

    for p in energy_model.parameters():
        p.requires_grad_(True)                       # restore weight gradients for training
    return inp_imgs.detach()


class Buffer:
    """Replay buffer of past samples, so Langevin chains continue instead of restarting from noise."""

    def __init__(self, model, device, num_channels=3, image_size=32, batch_size=128):
        self.model = model
        self.device = device
        self.shape = (num_channels, image_size, image_size)
        self.batch_size = batch_size
        # Start with random images in [-1, 1]
        self.examples = [torch.rand((1, *self.shape), device=device) * 2 - 1 for _ in range(batch_size)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch_size, 0.05)                          # ~5% fresh random images
        new_rand_imgs = torch.rand((n_new, *self.shape), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)   # rest from buffer
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        new_imgs = langevin_sample(self.model, inp_imgs, steps, step_size, noise)

        # Newest samples go to the front; keep at most 8192
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class EBM(nn.Module):
    """Wraps the energy CNN with its replay buffer and the contrastive-divergence training step."""

    def __init__(self, num_channels=3, image_size=32, alpha=0.1, steps=60, step_size=10, noise=0.005):
        super().__init__()
        self.model = EnergyModel(num_channels)
        self.num_channels = num_channels
        self.image_size = image_size
        self.alpha = alpha            # weight of the regularization term
        self.steps = steps            # Langevin steps per training batch
        self.step_size = step_size
        self.noise = noise
        self.buffer = None            # created on the training device in init_buffer

    def init_buffer(self, device, batch_size=128):
        self.buffer = Buffer(self.model, device, self.num_channels, self.image_size, batch_size)

    def forward(self, x):
        return self.model(x)

    def train_step(self, real_imgs, optimizer):
        # Small noise on real images, as in the class notebook
        real_imgs = (real_imgs + torch.randn_like(real_imgs) * self.noise).clamp(-1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(steps=self.steps, step_size=self.step_size, noise=self.noise)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).detach()
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        # Contrastive divergence: lower energy on real images, raise it on generated ones
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())   # keeps energies near 0
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()
        return {"loss": loss.item(), "cdiv": cdiv_loss.item(), "real": real_out.mean().item(), "fake": fake_out.mean().item()}

    def generate(self, num_images, steps=256, step_size=10.0, noise_std=0.01):
        device = next(self.model.parameters()).device
        x = torch.rand((num_images, self.num_channels, self.image_size, self.image_size), device=device) * 2 - 1
        imgs = langevin_sample(self.model, x, steps, step_size, noise_std)
        return ((imgs + 1) / 2).clamp(0, 1)          # [-1, 1] -> [0, 1] pixels