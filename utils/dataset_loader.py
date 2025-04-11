import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from config import DATA_PATH, BATCH_SIZE

def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize images between -1 and 1
    ])

    dataset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader
