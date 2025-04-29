from models import BilinearPoolingModel
from dataset import BimodalDataset

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse
import os, sys

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path_1, file_path_2, batch_size=16):
    sample_per_class = int(len(torch.load(file_path_1))/len(LULC_labels))

    train_size_per_class = int(0.7 * sample_per_class)
    test_size_per_class = int(0.15 * sample_per_class)
    val_size_per_class = int(sample_per_class) - train_size_per_class - test_size_per_class

    train_ids, val_ids, test_ids = [], [], []
    for idx in range(1, train_size_per_class+1):
        for label in LULC_labels:
            train_ids.append(f"{label}_{idx}")

    for idx in range(train_size_per_class+1, train_size_per_class+val_size_per_class+1):
        for label in LULC_labels:
            val_ids.append(f"{label}_{idx}")
    
    for idx in range(train_size_per_class+val_size_per_class+1, sample_per_class+1):
        for label in LULC_labels:
            test_ids.append(f"{label}_{idx}")

    # Create dataset instances for each split
    train_dataset = BimodalDataset(file_path_1, file_path_2, train_ids)
    val_dataset = BimodalDataset(file_path_1, file_path_2, val_ids)
    test_dataset = BimodalDataset(file_path_1, file_path_2, test_ids)

    # Create DataLoader for each dataset
    batch_size = batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def train_loop(model, train_loader, val_loader, epochs=50, lr=0.001):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_acc = [], []
    val_losses, val_acc = [], []
    for epoch in tqdm(range(epochs), desc="Training: "):
        # Training step
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for ids, data_img, data_txt, labels in train_loader:
            data_img, data_txt, labels = data_img.to(device), data_txt.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data_img, data_txt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation step
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient calculation
            for ids, data_img, data_txt, labels in val_loader:
                data_img, data_txt, labels = data_img.to(device), data_txt.to(device), labels.to(device)
                outputs = model(data_img, data_txt)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Print results for this epoch
        print(f'Epoch [{epoch+1}/{epochs}]:' + \
              f'Loss: {running_loss/len(train_loader):.4f}, ' + \
              f'Accuracy: {100 * correct / total:.2f}%, '+  \
              f'Validation Loss: {val_loss/len(val_loader):.4f}, ' + \
              f'Validation Accuracy: {100 * val_correct / val_total:.2f}%')
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        train_acc.append(correct / total)
        val_acc.append(val_correct / val_total)
    
    return model, train_losses, val_losses, train_acc, val_acc


def evaluation(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0
    
    criterion = nn.CrossEntropyLoss()

    all_ids = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for ids, data_img, data_txt, labels in data_loader:
            data_img, data_txt, labels = data_img.to(device), data_txt.to(device), labels.to(device)
            outputs = model(data_img, data_txt)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()

            all_ids.extend(list(ids))
            all_labels.extend(list(labels.cpu().numpy()))
            all_preds.extend(list(predicted.cpu().numpy()))

    
    eval_acc = eval_correct/eval_total
    loss = eval_loss/eval_total

    eval_records = {
        "ids": all_ids,
        "labels_true": all_labels,
        "labels_pred": all_preds 
    }

    return eval_acc, loss, pd.DataFrame.from_dict(eval_records)


def plot_train(train_losses, val_losses, save_path, mode="Loss"):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel('Epochs')
    plt.ylabel(mode)
    plt.title(f'{mode} Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, save_path, acc, loss):
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.annotate(f'Accuracy: {acc * 100:.2f}%', 
                 xy=(0.5, 1.05), 
                 xycoords='axes fraction',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=14, color='green', weight='bold')
    plt.annotate(f'CrossEntropyloss: {loss:.4f}', 
                 xy=(0.5, 1.10), 
                 xycoords='axes fraction',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=14, color='green', weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



def train_eval_model(file_path_1="/content/drive/MyDrive/Courses/6.8300/Final Project/embeddings_img/top_200_per_class.pt",
                    file_path_2="/content/drive/MyDrive/Courses/6.8300/Final Project/embeddings_geo_txt/top_200_per_class.pt",
                    output_dir="/content/drive/MyDrive/Courses/6.8300/Final Project/results/bilinear_pooling",
                    epochs=10,
                    lr=0.001,
                    batch_size=16,
                    hidden_layers=1,
                    hidden_units=[1024],
                    input_dim_1=2048,
                    input_dim_2=768):
    
    train_loader, val_loader, test_loader = load_data(file_path_1=file_path_1, 
                                                      file_path_2=file_path_2, 
                                                      batch_size=batch_size)

    model = BilinearPoolingModel(input_dim_img = input_dim_1, 
                            input_dim_txt = input_dim_2, 
                            output_dim=len(LULC_labels), 
                            hidden_layers=hidden_layers, 
                            hidden_units=hidden_units,
                            activation=nn.ReLU,
                            device=device)
    
    model, train_losses, val_losses, train_acc, val_acc = train_loop(model=model, 
                                                                    train_loader=train_loader, 
                                                                    val_loader=val_loader, 
                                                                    epochs=epochs, lr=lr)
    
    test_acc, test_loss, test_df = evaluation(model=model, 
                                              data_loader=test_loader)

    output_dir = os.path.join(output_dir, f"epochs={epochs}_lr={lr}_hidden={hidden_layers}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_df.to_csv(os.path.join(output_dir, "test_df.csv"))
    
    plot_train(train_losses, 
               val_losses,
               os.path.join(output_dir, "loss_curve.jpg"),
               mode="Loss")
    plot_train(train_acc, 
               val_acc,
               os.path.join(output_dir, "acc_curve.jpg"),
               mode="Accuracy")
    
    plot_confusion_matrix(test_df["labels_true"], 
                          test_df["labels_pred"], 
                          os.path.join(output_dir, "test_confusion_matrix.jpg"),
                          acc=test_acc,
                          loss=test_loss)
    
