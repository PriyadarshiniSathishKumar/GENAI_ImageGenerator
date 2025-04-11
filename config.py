import torch

# Training Parameters
BATCH_SIZE = 128
IMG_SIZE = 64
LATENT_DIM = 100  # Noise vector size
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5  # Adam optimizer beta1 value

# Data Parameters
DATA_PATH = "./dataset/"
SAVE_MODEL_PATH = "./models/"
GENERATED_IMG_PATH = "./generated_images/"

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
