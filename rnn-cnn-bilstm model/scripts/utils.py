import math
import time
import random

# Pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

import matplotlib.pyplot as plt

RANDOM_SEED = 0

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)

def train_model(model, datamodule, num_epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for seqs, labs, lengths in tqdm_notebook(train_loader, desc=f'Epoch {epoch+1}'):
            seqs, labs, lengths = seqs.to(device), labs.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * seqs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labs).sum().item()
            total_train += labs.size(0)
        
        # Training metrics
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        torch.cuda.empty_cache()
    
    return history

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seqs, labs, lengths in dataloader:
            seqs, labs, lengths = seqs.to(device), labs.to(device), lengths.to(device)
            
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labs)
            
            running_loss += loss.item() * seqs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def test_model(model, dataloader, criterion, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seqs, labs, lengths in dataloader:
            seqs, labs, lengths = seqs.to(device), labs.to(device), lengths.to(device)
            
            # Forward pass
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labs)
            
            # Update statistics
            running_loss += loss.item() * seqs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    
    # Compute average loss and accuracy
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()