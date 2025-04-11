import torch
import matplotlib.pyplot as plt
from models.generator import Generator
from config import LATENT_DIM, DEVICE, SAVE_MODEL_PATH

# Load trained generator
generator = Generator(LATENT_DIM).to(DEVICE)
generator.load_state_dict(torch.load(f"{SAVE_MODEL_PATH}/generator.pth"))

# Generate image
z = torch.randn(1, LATENT_DIM).to(DEVICE)
img = generator(z).cpu().detach().squeeze()

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
