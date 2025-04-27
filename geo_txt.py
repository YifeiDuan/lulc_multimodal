import rasterio
from rasterio.transform import xy
import numpy as np
import pandas as pd
import math
import random

import torch

from pyproj import Transformer

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

import openai
from openai import OpenAI


key = open("/content/drive/My Drive/Courses/6.8300/Final Project/6-8300-key", "r").read().strip("\n")
client = OpenAI(api_key=key)

def get_geo_coords(tif_path):
    # e.g. tif_path = "/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"

    with rasterio.open(tif_path) as src:
        # Get the lat/lon of the center pixel
        center_x = src.width // 2
        center_y = src.height // 2
        easting, northing = xy(src.transform, center_y, center_x)
    
        # Transform the coordinate to standard lat/lon under "epsg:4326" system (WGS 84 coordinate system)
        transformer = Transformer.from_crs(src.crs, "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)

    return lon, lat


def gen_geo_txt(tif_path):

    # Get coordinates (lon, lat)
    lon, lat = get_geo_coords(tif_path=tif_path)

    # Generate text
    try:
        #Make your OpenAI API request here
        prompt = f"Summarize some information about the geospatial location with (latitude: {lat}, longitude: {lon})"
    
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        
        geo_txt = response.choices[0].message.content
        
    except openai.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass

    return geo_txt


def encode_geo_txt(tif_path):
    geo_txt = gen_geo_txt(tif_path=tif_path)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # The embedding model you want to use
            input=geo_txt,
            encoding_format="float"
        )
    except openai.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass

    # Access the embedding vectors
    embedding = response.data[0].embedding

    return torch.tensor(embedding)



def batch_encode_geo_txt(img_dir="/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS", 
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

    # Prep for saving img embeddings
    save_pt_name = f"{mode}.pt"
    if mode == "sample":
        save_pt_name = f"{mode}_{samples_per_class}_per_class.pt"
    save_dir = os.path.join(os.path.dirname(img_dir), "embeddings_geo_txt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get img embeddings for all the tif paths
    all_embeddings = {}
    for idx, tif_path in tqdm(enumerate(all_tif_files), total=len(all_tif_files), desc="Encoding for: "):
        file_id = tif_path.split("/")[-1].split(".")[0]     # "PATH/AnnualCrop_1.tif" -> "AnnualCrop_1"
        embedding = encode_geo_txt(tif_path=tif_path)

        all_embeddings[file_id] = embedding

        if ((idx+1)%200 == 0) or (idx == len(all_tif_files)-1):
            torch.save(all_embeddings, os.path.join(save_dir, save_pt_name))


    
