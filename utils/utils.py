import torch
import matplotlib.pyplot as plt
import os
from config import GENERATED_IMG_PATH, SAVE_MODEL_PATH, DEVICE  # Import DEVICE

def save_model(model, filename):
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, filename))

def load_model(model, filename):
    model.load_state_dict(torch.load(os.path.join(SAVE_MODEL_PATH, filename), map_location=DEVICE))

def generate_and_save_images(generator, epoch, latent_dim):
    os.makedirs(GENERATED_IMG_PATH, exist_ok=True)

    z = torch.randn(16, latent_dim).to(DEVICE)  # Use DEVICE from config
    with torch.no_grad():
        samples = generator(z).cpu()

    fig, axs = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.savefig(f"{GENERATED_IMG_PATH}/epoch_{epoch}.png")
    plt.close()
