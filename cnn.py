import rasterio
import numpy as np
import pandas as pd
import math
import random

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import torchvision.models as models

from pathlib import Path
import os, sys
from tqdm import tqdm

LULC_labels = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

def load_img(tif_path):
    # e.g. tif_path = "/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"

    # Read GeoTIFF and extract RGB bands (4=R, 3=G, 2=B)
    with rasterio.open(tif_path) as src:
        img = src.read()
        rgb = img[[3, 2, 1], :, :]  # shape: (3, H, W)

    # Normalize Sentinel-2 range and convert to PIL
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(np.transpose(rgb_uint8, (1, 2, 0)))

    # Preprocess for a lightweight CNN
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])      # Using the mean and std for ImageNet natural images
    ])

    img_tensor = preprocess(rgb_pil)  # shape: (3, 224, 224)
    return img_tensor


def batch_load_img(img_dir="/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS", 
                     mode="selected", 
                     samples_per_class=None,
                     df_path=None):
    valid_modes = ["all", "selected", "sample", "top"]
    assert mode in valid_modes, f"choose one valid mode from {valid_modes}"

    # Configure all tif file paths
    all_tif_files = []
    if mode == "all":
        for label in LULC_labels:
            class_dir = os.path.join(img_dir, label)
            tif_files = list(Path(class_dir).rglob('*.tif'))        # PosixPath format
            tif_files = [str(f) for f in tif_files]                 # conver path to str
            all_tif_files.extend(tif_files)
    elif mode == "selected":
        assert isinstance(df_path, str), "provide a str: df_path - that contains selected tif paths"
        df = pd.read_csv(df_path)
        for filename in df["Filename"]:
            tif_path = os.path.join(img_dir, filename)
            all_tif_files.append(tif_path)
    elif mode == "sample":
        assert isinstance(samples_per_class, int), "provide an int: samples_per_class"
        for label in LULC_labels:
            class_dir = os.path.join(img_dir, label)
            tif_files = list(Path(class_dir).rglob('*.tif'))[:samples_per_class]        # PosixPath format
            tif_files = [str(f) for f in tif_files]                 # conver path to str
            all_tif_files.extend(tif_files)
    elif mode == "top":
        assert isinstance(samples_per_class, int), "provide an int: samples_per_class"
        for label in LULC_labels:
            class_dir = os.path.join(img_dir, label)
            for idx in range(1, samples_per_class+1):
                tif_file = os.path.join(class_dir, f"{label}_{idx}.tif")
                all_tif_files.append(tif_file)

    # Get img features for all the tif paths
    all_features = {}
    for tif_path in tqdm(all_tif_files, desc="Encoding for: "):
        file_id = tif_path.split("/")[-1].split(".")[0]     # "PATH/AnnualCrop_1.tif" -> "AnnualCrop_1"
        features = load_img(tif_path=tif_path)

        all_features[file_id] = features

    # Save img features
    save_pt_name = f"{mode}.pt"
    if (mode == "sample") or (mode == "top"):
        save_pt_name = f"{mode}_{samples_per_class}_per_class.pt"
    save_dir = os.path.join(os.path.dirname(img_dir), "features_img")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(all_features, os.path.join(save_dir, save_pt_name))





class CNN(nn.Module):
    def __init__(self, in_ch=3, output_dim=10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),  # 3→32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64→32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32→64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32→16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),  # Based on 64 channels, 16x16 spatial
            nn.ReLU(),
            nn.Linear(256, output_dim)  # 10 classes for EuroSAT
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x
    


class CNN_fusion(nn.Module):
    def __init__(self, in_ch=3, input_dim_txt=768, output_dim=10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),  # 3→32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64→32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32→64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32→16
        )

        self.img_adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),  # Based on 64 channels, 16x16 spatial
            nn.ReLU()
        )

        self.txt_adapter = nn.Sequential(
            nn.Linear(input_dim_txt, 256),
            nn.ReLU()
        )

        self.fc = nn.Linear(512, output_dim)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.img_adapter(x1)

        x2 = self.txt_adapter(x2)

        x = torch.cat((x1, x2), dim=-1)
        return self.fc(x)
