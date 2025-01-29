import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pytesseract
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_files = os.listdir(low_res_dir)  # Both folders must have same filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.image_files[idx])
        high_res_path = os.path.join(self.high_res_dir, self.image_files[idx])

        low_res_image = Image.open(low_res_path).convert("L")  # Convert to grayscale
        high_res_image = Image.open(high_res_path).convert("L")  # Convert to grayscale

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image

# Residual Dense Block (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out

# Generator
class Generator(nn.Module):
    def __init__(self, in_channels=1, num_rrdb=5):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(32) for _ in range(num_rrdb)])
        self.final_conv = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feature = self.initial_conv(x)
        out = self.rrdb_blocks(initial_feature)
        out = self.final_conv(out)
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# Loss Functions
# Sobel Loss
class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, sr, hr):
        # Apply Sobel filters
        sobel_x = self.sobel_x.to(sr.device)
        sobel_y = self.sobel_y.to(sr.device)

        # Compute gradients (edges) in x and y directions
        grad_sr_x = F.conv2d(sr, sobel_x, padding=1)
        grad_sr_y = F.conv2d(sr, sobel_y, padding=1)
        grad_hr_x = F.conv2d(hr, sobel_x, padding=1)
        grad_hr_y = F.conv2d(hr, sobel_y, padding=1)

        # Compute Sobel loss as the L1 distance between gradients
        grad_loss = F.l1_loss(grad_sr_x, grad_hr_x) + F.l1_loss(grad_sr_y, grad_hr_y)
        return grad_loss

# Loss functions
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, sr, hr):
        return F.mse_loss(sr, hr)

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg_model.features[:18]  # First 18 layers of VGG19
        self.vgg.eval()

    def forward(self, sr, hr):
        # Ensure both sr and hr have 3 channels for VGG (replicate grayscale)
        sr = sr.repeat(1, 3, 1, 1)  # Replicate grayscale to RGB
        hr = hr.repeat(1, 3, 1, 1)  # Replicate grayscale to RGB
        
        # Extract features using VGG
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        # Calculate MSE loss between feature maps
        return F.mse_loss(sr_features, hr_features)

class SSIMLoss(nn.Module):
    def forward(self, sr, hr):
        sr = sr.detach().squeeze().cpu().numpy()  # Detach before converting to numpy
        hr = hr.detach().squeeze().cpu().numpy()  # Detach before converting to numpy
        return 1 - ssim(sr, hr, data_range=1.0)

import numpy as np
import pytesseract
from PIL import Image

class OCRLoss(nn.Module):
    def forward(self, sr, hr):
        # Convert tensor to NumPy (detach, move to CPU, remove batch and channel dimensions)
        sr_img = sr.detach().cpu().numpy()  # Shape: (batch, 1, H, W)
        hr_img = hr.detach().cpu().numpy()

        # Remove batch and channel dimensions
        sr_img = sr_img[0, 0, :, :]  # Now shape is (H, W)
        hr_img = hr_img[0, 0, :, :]

        # Convert to 8-bit format
        sr_img = (sr_img * 255).astype(np.uint8)
        hr_img = (hr_img * 255).astype(np.uint8)

        # Convert to PIL Image
        sr_pil = Image.fromarray(sr_img, mode="L")  # "L" mode = grayscale
        hr_pil = Image.fromarray(hr_img, mode="L")

        # Extract text using Tesseract
        sr_text = pytesseract.image_to_string(sr_pil)
        hr_text = pytesseract.image_to_string(hr_pil)

        # Compute OCR loss (L1 loss of text length difference)
        return F.l1_loss(
            torch.tensor([len(sr_text)], dtype=torch.float32, device=sr.device),
            torch.tensor([len(hr_text)], dtype=torch.float32, device=sr.device)
        )



# Training function
def train(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D, losses, device):
    generator.to(device)
    discriminator.to(device)

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, (low_res_image, high_res_image) in enumerate(dataloader):
            low_res_image = low_res_image.to(device)
            high_res_image = high_res_image.to(device)

            optimizer_G.zero_grad()
            sr_image = generator(low_res_image)

            content_loss = losses['content'](sr_image, high_res_image)
            perceptual_loss = losses['perceptual'](sr_image, high_res_image)
            sobel_loss = losses['sobel'](sr_image, high_res_image)
            ssim_loss = losses['ssim'](sr_image, high_res_image)
            ocr_loss = losses['ocr'](sr_image, high_res_image)

            g_loss = content_loss + perceptual_loss + sobel_loss + ssim_loss + ocr_loss
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_output = discriminator(high_res_image)
            fake_output = discriminator(sr_image.detach())
            d_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) + \
                     F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            optimizer_D.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        g_losses.append(epoch_g_loss / len(dataloader))
        d_losses.append(epoch_d_loss / len(dataloader))

    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    transform = transforms.ToTensor()
    dataset = ImageDataset("dataset/low_res", "dataset/high_res", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    losses = {
        'content': nn.MSELoss(),
        'perceptual': PerceptualLoss(models.vgg19(pretrained=True)),
        'sobel': SobelLoss(),
        'ssim': SSIMLoss(),
        'ocr': OCRLoss()
    }

    train(generator, discriminator, dataloader, 10, optimizer_G, optimizer_D, losses, device)

