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

from google import genai
from google.api_core import exceptions

# Configure API Key
api_key = open(f"/content/drive/My Drive/Courses/6.8300/Final Project/6-8300-key-gemini", "r").read().strip("\n")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=api_key)

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

    geo_txt = "Unsuccessful generation"

    # Generate content
    model_name = "gemini-2.5-flash-preview-04-17"
    
    try:
        #Make your OpenAI API request here
        prompt = f"Summarize information about the geospatial location with (latitude: {lat}, longitude: {lon}). Only provide information you are certain about."
    
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt
        )
            
        geo_txt = response.text

    except exceptions.PermissionDenied:
        print("\nError: Permission denied. Check your API key and ensure the Gemini API is enabled for your project.")
        pass
    except exceptions.NotFound:
        print(f"\nError: Model '{model_name}' not found or you don't have access. Please check the model name.")
        pass
    except exceptions.InvalidArgument as e:
        print(f"\nError: Invalid argument. This might be due to an invalid task_type for the model, missing required parameters, or other issues. Details: {e}")
        pass
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        pass
    

    return geo_txt


def encode_geo_txt(geo_txt):

    model_name = "models/text-embedding-004"
    try:
        response = client.models.embed_content(
            model=model_name, 
            contents=geo_txt
        )
    except exceptions.PermissionDenied:
        print("\nError: Permission denied. Check your API key and ensure the Gemini API is enabled for your project.")
        pass
    except exceptions.NotFound:
        print(f"\nError: Model '{model_name}' not found or you don't have access. Please check the model name.")
        pass
    except exceptions.InvalidArgument as e:
        print(f"\nError: Invalid argument. This might be due to an invalid task_type for the model, missing required parameters, or other issues. Details: {e}")
        pass
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        pass
    

    # Access the embedding vectors
    embedding = list(response.embeddings)[0].values

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
    if (mode == "sample") or (mode == "top"):
        save_pt_name = f"{mode}_{samples_per_class}_per_class.pt"
    save_dir = os.path.join(os.path.dirname(img_dir), "embeddings_geo_txt_gemini")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get img embeddings for all the tif paths
    all_geo_txt = []
    all_embeddings = {}
    for idx, tif_path in tqdm(enumerate(all_tif_files), total=len(all_tif_files), desc="Encoding for: "):
        file_id = tif_path.split("/")[-1].split(".")[0]     # "PATH/AnnualCrop_1.tif" -> "AnnualCrop_1"
        geo_txt = gen_geo_txt(tif_path=tif_path)
        embedding = encode_geo_txt(geo_txt=geo_txt)

        all_geo_txt.append({
            "file": file_id,
            "geo_txt": geo_txt
        })
        all_embeddings[file_id] = embedding

        if ((idx+1)%200 == 0) or (idx == len(all_tif_files)-1):
            pd.DataFrame.from_records(all_geo_txt).to_csv(os.path.join(save_dir, f"{mode}_{samples_per_class}_per_class_txt.csv"))
            torch.save(all_embeddings, os.path.join(save_dir, save_pt_name))


    
