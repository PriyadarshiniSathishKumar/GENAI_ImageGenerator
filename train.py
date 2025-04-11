import torch
import torch.optim as optim
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset_loader import get_dataloader
from utils.utils import save_model, generate_and_save_images
from config import *

# Initialize models
generator = Generator(LATENT_DIM).to(DEVICE)  # Ensure models are moved to DEVICE
discriminator = Discriminator().to(DEVICE)

# Loss & Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Load dataset
dataloader = get_dataloader()

# Training Loop
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        real = torch.ones(imgs.size(0), 1).to(DEVICE)
        fake = torch.zeros(imgs.size(0), 1).to(DEVICE)

        real_imgs = imgs.to(DEVICE)  # Move images to DEVICE
        
        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), LATENT_DIM).to(DEVICE)  # Use DEVICE
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), real)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    generate_and_save_images(generator, epoch, LATENT_DIM)

    # Save model every 10 epochs
    if epoch % 10 == 0:
        save_model(generator, "generator.pth")
        save_model(discriminator, "discriminator.pth")
