# sps_genai

FastAPI app with a bigram text generator, spaCy word embeddings, a CNN image classifier, a GAN that generates handwritten digits, and diffusion and energy-based models that generate CIFAR10-style images.

## Run with Docker

```bash
docker build -t asm4-genai .
docker run --rm -p 8000:80 asm4-genai
```

Docs: http://127.0.0.1:8000/docs

## Endpoints

- `POST /generate`: `{"start_word": "bigram", "length": 5}`
- `POST /embedding`: `{"word": "apple"}` returns a 300-dimension vector
- `POST /similarity`: `{"word1": "apple", "word2": "orange"}`
- `POST /classify`: upload an image file, returns the predicted CIFAR10 class
- `POST /generate_digits`: `{"num_samples": 10, "seed": 42}` returns a PNG grid of GAN-generated digits (`num_samples` 1-64, `seed` optional)
- `POST /generate_diffusion`: `{"num_samples": 10, "diffusion_steps": 50, "seed": 42}` returns a PNG grid of images from the diffusion model (`num_samples` 1-64, `diffusion_steps` 1-500, `seed` optional)
- `POST /generate_ebm`: `{"num_samples": 10, "langevin_steps": 256, "seed": 42}` returns a PNG grid of images sampled from the energy-based model (`num_samples` 1-64, `langevin_steps` 1-1000, `seed` optional)

## Examples

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'

curl -X POST http://127.0.0.1:8000/classify -F "file=@your_image.png"

curl -X POST http://127.0.0.1:8000/generate_digits \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10, "seed": 42}' \
  -o digits.png

curl -X POST http://127.0.0.1:8000/generate_diffusion \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10, "seed": 42}' \
  -o diffusion.png
```

## Assignment 2: CNN image classifier

`POST /classify` accepts an image upload and returns the predicted CIFAR10 class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) with probabilities.

Model: `app/cnn_model.py` (Conv16 -> Pool -> Conv32 -> Pool -> FC100 -> FC10, 64x64x3 input). Trained with `train_cnn.py` on CIFAR10 (resized to 64x64); weights saved in `app/cnn_cifar10.pth`.

## Assignment 3: GAN digit generator

`POST /generate_digits` returns a PNG grid of handwritten digits created by a GAN trained on MNIST. The same `seed` always returns the same digits.

Helper library (`helper_lib/`, Module 6 class activity):
- `model.py`: `Generator`, `Discriminator`, a `GAN` wrapper holding both, and `get_model("GAN")`
- `trainer.py`: `train_gan` (standard GAN training with binary cross-entropy)
- `generator.py`: `generate_samples` (samples noise, runs the generator, plots or returns the image grid)
- `data_loader.py`: `get_data_loader` (MNIST scaled to [-1, 1])

Architecture:
- Generator: noise (100) -> Linear to 7x7x128 -> ConvTranspose 128->64 (14x14) + BatchNorm + ReLU -> ConvTranspose 64->1 (28x28) + Tanh
- Discriminator: image (1x28x28) -> Conv 1->64 (14x14) + LeakyReLU(0.2) -> Conv 64->128 (7x7) + BatchNorm + LeakyReLU(0.2) -> Flatten -> Linear to 1 output

The discriminator's Linear layer outputs a logit; the sigmoid that turns it into the real/fake probability is applied inside `nn.BCEWithLogitsLoss` during training.

Trained with `train_gan.py` (10 epochs, Adam lr 2e-4, batch size 128); generator weights saved in `app/gan_mnist.pth`.

## Assignment 4: Diffusion and energy-based image generators

Both models are trained on CIFAR10 (32x32x3) and added to the helper library (Module 8 class activity):
- `helper_lib/diffusion.py`: UNet with sinusoidal noise embedding, offset cosine noise schedule, and `DiffusionModel` (training step, EMA network for sampling, reverse diffusion)
- `helper_lib/energy.py`: energy CNN, Langevin sampling (gradient descent on the input image with the weights frozen), replay buffer, and `EBM` (contrastive divergence training step)
- `get_model("Diffusion")` / `get_model("EBM")`, `train_diffusion` / `train_ebm`, and `generate_samples` (reverse diffusion for the diffusion model, Langevin dynamics for the EBM)

`POST /generate_diffusion` starts from Gaussian noise and runs reverse diffusion. `POST /generate_ebm` starts from uniform noise and runs Langevin dynamics to lower each image's energy. Both return a PNG grid, enlarged 4x for viewing; the same `seed` returns the same images.

Trained with `train_diffusion.py` (10 epochs, AdamW lr 1e-3, L1 loss on predicted noise) and `train_ebm.py` (10 epochs, Adam lr 1e-4, 60 Langevin steps per batch); weights saved in `app/diffusion_cifar10.pth` and `app/ebm_cifar10.pth`.