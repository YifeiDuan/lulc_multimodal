import rasterio
import numpy as np
import pandas as pd
import math
import random

import torch
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

    # Preprocess for ResNet or CLIP
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])      # Using the mean and std for ImageNet natural images
    ])

    img_tensor = preprocess(rgb_pil).unsqueeze(0)  # shape: (1, 3, 224, 224)
    return img_tensor


def encode_img(tif_path):
    img_tensor = load_img(tif_path=tif_path)

    model = models.resnet50(pretrained=True)
    model.eval()

    # Remove classification layer
    embedding_model = torch.nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        embedding = embedding_model(img_tensor).squeeze()  # torch tensor; shape: (2048,)
    
    return embedding

def batch_encode_img(img_dir="/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS", 
                     mode="selected", 
                     samples_per_class=None,
                     df_path=None):
    valid_modes = ["all", "selected", "sample"]
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

    # Get img embeddings for all the tif paths
    all_embeddings = {}
    for tif_path in tqdm(all_tif_files, desc="Encoding for: "):
        file_id = tif_path.split("/")[-1].split(".")[0]     # "PATH/AnnualCrop_1.tif" -> "AnnualCrop_1"
        embedding = encode_img(tif_path=tif_path)

        all_embeddings[file_id] = embedding

    # Save img embeddings
    save_pt_name = f"{mode}.pt"
    if mode == "sample":
        save_pt_name = f"{mode}_{samples_per_class}_per_class.pt"
    save_dir = os.path.join(os.path.dirname(img_dir), "embeddings_img")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(all_embeddings, os.path.join(save_dir, save_pt_name))

