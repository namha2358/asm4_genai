# helper_lib/trainer.py — training loops for the GAN (Assignment 3), diffusion model and energy model (Assignment 4)
import torch
from tqdm import tqdm


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Train a GAN wrapper (model.generator + model.discriminator).
    criterion: nn.BCEWithLogitsLoss() — applies the sigmoid to the discriminator's logit inside the loss
    optimizer: pair (opt_gen, opt_disc), one optimizer per network
    """
    gen, disc = model.generator, model.discriminator
    opt_gen, opt_disc = optimizer              # unpack the two optimizers
    model.to(device)                           # move both networks to the chosen device
    model.train()                              # training mode: BatchNorm uses batch statistics

    for epoch in range(epochs):
        loop = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real, _) in enumerate(loop):   # MNIST labels are ignored, as in the class notebook
            real = real.to(device)
            batch_size = real.size(0)

            # Targets for the BCE loss: 1 = real, 0 = fake
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            ## === Train Discriminator === ##
            noise = torch.randn(batch_size, model.z_dim, device=device)   # fresh noise vectors
            fake = gen(noise)                                             # generate a batch of fake images
            loss_real = criterion(disc(real), real_labels)                # push scores on real images toward 1
            loss_fake = criterion(disc(fake.detach()), fake_labels)       # push scores on fakes toward 0; detach = don't update gen here
            loss_disc = loss_real + loss_fake

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ## === Train Generator === ##
            loss_gen = criterion(disc(fake), real_labels)   # generator wants the discriminator to call its fakes real (1)

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Show the latest losses on the progress bar every 100 batches
            if batch_number % 100 == 0:
                loop.set_postfix({"D loss": f"{loss_disc.item():.4f}", "G loss": f"{loss_gen.item():.4f}"})

    return model



def compute_normalizer(data_loader, num_channels=3):
    """Per-channel mean and standard deviation of every pixel in the dataset (images in [0, 1])."""
    total, total_sq, count = torch.zeros(num_channels), torch.zeros(num_channels), 0
    for imgs, _ in data_loader:
        flat = imgs.transpose(0, 1).reshape(num_channels, -1)   # (C, all pixels in the batch)
        total += flat.sum(dim=1)
        total_sq += (flat ** 2).sum(dim=1)
        count += flat.shape[1]
    mean = total / count
    std = torch.sqrt(total_sq / count - mean ** 2)              # std = sqrt(E[x^2] - E[x]^2)
    return mean, std


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Train a DiffusionModel: each step adds noise at a random level and teaches the UNet to predict that noise.
    criterion: nn.L1Loss(), as in the class notebook
    """
    model.to(device)
    model.network.train()
    for epoch in range(epochs):
        losses = []
        loop = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (images, _) in enumerate(loop):          # CIFAR10 labels are ignored
            images = images.to(device)
            losses.append(model.train_step(images, optimizer, criterion))
            if batch_number % 100 == 0:
                loop.set_postfix({"loss": f"{sum(losses) / len(losses):.4f}"})
        print(f"Epoch {epoch+1}: average loss {sum(losses) / len(losses):.4f}")
    return model


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10):
    """
    Train an EBM with contrastive divergence: lower the energy of real images,
    raise the energy of images sampled from the model by Langevin dynamics.
    """
    model.to(device)
    model.init_buffer(device, batch_size=data_loader.batch_size)   # replay buffer on the training device
    for epoch in range(epochs):
        loop = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real_imgs, _) in enumerate(loop):        # CIFAR10 labels are ignored
            metrics = model.train_step(real_imgs.to(device), optimizer)
            if batch_number % 100 == 0:
                loop.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})
    return model