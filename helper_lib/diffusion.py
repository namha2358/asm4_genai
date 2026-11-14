# helper_lib/diffusion.py — diffusion model (UNet + noise schedule), adapted from Module_8_Practical_2_DiffusionMethods.py
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Map diffusion times in [0, 1] to (noise_rates, signal_rates); same schedule the class notebook trains with."""
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Angles at which the signal rate equals its max (t = 0) and its min (t = 1)
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Interpolate the angle linearly in time, then signal = cos(angle), noise = sin(angle)
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)
    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    """Encodes the noise variance as sin/cos features at 16 frequencies -> 32 values per image."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) noise variances -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection (1x1 projection when channel counts differ)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if self.needs_projection else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)            # swish activation, as in the class notebook

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks (saved as skips for the UpBlocks), then 2x2 average pooling to halve the size."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    """Upsamples 2x, concatenates the matching skip from the DownBlock, then runs residual blocks."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise in a noisy image, given the image and its noise variance (class notebook architecture)."""

    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)      # image -> 32 feature channels
        self.embedding = SinusoidalEmbedding(num_frequencies=16)        # noise variance -> 32 channels

        # 32 image channels + 32 embedding channels = 64 going into the first DownBlock
        self.down1 = DownBlock(32, in_channels=64, block_depth=2)       # 32x32 -> 16x16
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)       # 16x16 -> 8x8
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)       # 8x8 -> 4x4 (bottleneck)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)          # 4x4 -> 8x8
        self.up2 = UpBlock(64, in_channels=96, block_depth=2)           # 8x8 -> 16x16
        self.up3 = UpBlock(32, in_channels=64, block_depth=2)           # 16x16 -> 32x32

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)         # back to image channels
        nn.init.zeros_(self.final.weight)                               # start by predicting zero noise

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)                     # (B, 1, 1, 32)
        # Spread the embedding over every pixel so it can be stacked with the image features
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    """Wraps the UNet with the noise schedule, an EMA copy for sampling, and per-channel normalization."""

    def __init__(self, image_size=32, num_channels=3, ema_decay=0.999):
        super().__init__()
        self.network = UNet(image_size, num_channels)
        self.ema_network = copy.deepcopy(self.network)   # slowly-updated copy, used to generate images
        self.ema_network.eval()
        self.ema_decay = ema_decay
        self.schedule_fn = offset_cosine_diffusion_schedule
        self.image_size = image_size
        self.num_channels = num_channels
        # Per-channel mean/std of the training images, stored as buffers so they are saved with the weights
        self.register_buffer("normalizer_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(mean.view(1, -1, 1, 1))
        self.normalizer_std.copy_(std.view(1, -1, 1, 1))

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)   # back to [0, 1] pixels

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Training uses the main network; sampling uses the EMA network
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)               # UNet predicts the added noise
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates   # solve for the clean image
        return pred_noises, pred_images

    @torch.no_grad()
    def update_ema(self):
        # EMA weights = decay * EMA weights + (1 - decay) * current weights
        for ema_p, p in zip(self.ema_network.parameters(), self.network.parameters()):
            ema_p.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std     # normalize to mean 0, std 1
        noises = torch.randn_like(images)                                  # Gaussian noise to add

        # A random diffusion time per image decides how much noise it gets
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)                                # compare predicted vs. true noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.update_ema()
        return loss.item()

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            # Current time goes from 1 (pure noise) down toward 0 (clean image)
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            # Re-noise the predicted clean image to the next (slightly less noisy) time
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps):
        device = self.normalizer_mean.device
        initial_noise = torch.randn((num_images, self.num_channels, self.image_size, self.image_size), device=device)
        return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))   # (N, C, H, W) in [0, 1]