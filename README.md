# 🧠 GAN Image Generator with PyTorch

This project is a PyTorch-based implementation of a **Generative Adversarial Network (GAN)** trained on the **MNIST** dataset. The generator learns to create realistic handwritten digits, while the discriminator learns to distinguish between real and fake images.

## 📁 Project Structure

```
.
├── config.py                # Global config (hyperparameters, paths, device)
├── train.py                 # Main training script
├── generate.py              # Script to generate images from saved generator
├── models/
│   ├── generator.py         # Generator architecture
│   └── discriminator.py     # Discriminator architecture
├── utils/
│   ├── dataset_loader.py    # Loads and preprocesses MNIST data
│   └── utils.py             # Helper functions for saving models/images
├── generated_images/        # Output folder for generated images
├── models/                  # Saved models (.pth files)
└── dataset/                 # MNIST dataset download location
```


## ⚙️ Features

- Custom **Generator** and **Discriminator** networks.
- Image generation using **latent noise vector**.
- Real-time image saving during training.
- **Model checkpointing** for later inference.
- Device support for **GPU acceleration**.

---

## 🧪 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/gan-image-generator.git
   cd gan-image-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## 🚀 Training

To start training the GAN model:
```bash
python train.py
```

- Models will be saved in the `./models/` folder.
- Generated images will be saved in the `./generated_images/` folder after each epoch.

---

## 🖼️ Generate New Images

Once training is complete, generate images using the saved generator:

```bash
python generate.py
```

This script will load `generator.pth` and display a new image generated from random noise.

---

## 🧠 Model Details

- **Latent vector size**: `100`
- **Generator output**: 1 × 28 × 28 grayscale image
- **Discriminator output**: Probability [0, 1] of being a real image

---

## 📸 Sample Output
![epoch_49](https://github.com/user-attachments/assets/eee94c59-f74a-4205-95bc-71035965385c)
![epoch_0](https://github.com/user-attachments/assets/d5c26209-1c7f-40e3-903a-5dc23f4faf9f)

---

## ✨ Credits

Developed using **PyTorch** as part of a Generative AI project.  
Feel free to fork, contribute, or use it for your own creative experiments!

---

Let me know if you want it in a fancy Markdown format with badges, emoji headers, or collapsible sections!
