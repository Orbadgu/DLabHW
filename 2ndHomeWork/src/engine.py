import torch
import numpy as np
import os
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def extract_features(model, dataloader, device, save_prefix="train"):
    """4. Model için CNN'den özellik çıkarma"""
    model.eval()
    features_list = []
    labels_list = []

    os.makedirs('features', exist_ok=True)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Extracting {save_prefix} features"):
            inputs = inputs.to(device)
            # Özellik çıkarıcı kısmı kullan (Sınıflandırıcıdan hemen öncesi)
            features = model.features(inputs)
            features = features.view(features.size(0), -1)  # Flatten

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)

    np.save(f'features/X_{save_prefix}.npy', X)
    np.save(f'features/y_{save_prefix}.npy', y)

    print(f"{save_prefix} features saved! Shape: {X.shape}")