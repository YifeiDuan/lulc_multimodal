import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

LULC_labels_map = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}

class UnimodalDataset(Dataset):
    def __init__(self, file_path, ids):
        # Load the dataset from the .pt file
        data = torch.load(file_path)

        self.ids = []
        self.data = []
        self.labels = []
        
        # Assuming your data is stored as key-value pairs, where the values are the feature vectors
        for key, value in data.items():
            if key in ids:
                self.ids.append(key)
                # Each feature vector is assumed to be a tensor
                self.data.append(value)
                # Add a LULC label
                label = key.split("_")[0]       # key: e.g. "AnnualCrop_100"
                self.labels.append(LULC_labels_map[label])

        # Convert lists to tensors
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.ids[idx], self.data[idx], self.labels[idx]



class EarlyFusionDataset(Dataset):
    def __init__(self, file_path_1, file_path_2, ids):
        # Load the two datasets from the .pt files
        data1 = torch.load(file_path_1)
        data2 = torch.load(file_path_2)
        
        self.ids = []
        self.data = []
        self.labels = []
        
        # Iterate over the provided identifiers and select the relevant data points
        for key in ids:
            if key in data1.keys() and key in data2.keys():
                self.ids.append(key)
                # Concatenate the feature vectors from both datasets
                feature_vector = torch.cat((data1[key], data2[key]), dim=-1)
                self.data.append(feature_vector)
                
                # Add a LULC label
                label = key.split("_")[0]       # key: e.g. "AnnualCrop_100"
                self.labels.append(LULC_labels_map[label])

        # Convert lists to tensors
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.ids[idx], self.data[idx], self.labels[idx]


class BimodalDataset(Dataset):
    def __init__(self, file_path_1, file_path_2, ids):
        # Load the dataset from the .pt file
        data1 = torch.load(file_path_1)
        data2 = torch.load(file_path_2)

        self.ids = []
        self.data_img = []
        self.data_txt = []
        self.labels = []
        
        # Iterate over the provided identifiers and select the relevant data points
        for key in ids:
            if key in data1.keys() and key in data2.keys():
                self.ids.append(key)
                # Retrieve the feature vectors from both datasets
                self.data_img.append(data1[key])
                self.data_txt.append(data2[key])
                
                # Add a LULC label
                label = key.split("_")[0]       # key: e.g. "AnnualCrop_100"
                self.labels.append(LULC_labels_map[label])

        # Convert lists to tensors
        self.data_img = torch.stack(self.data_img)
        self.data_txt = torch.stack(self.data_txt)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.data_img[idx], self.data_txt[idx], self.labels[idx]