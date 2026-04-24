import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.models import BaseCNN, ImprovedCNN, get_pretrained_alexnet
from src.engine import train_one_epoch, test_model, extract_features
from src.hybrid import train_hybrid_svm
from torch.optim.lr_scheduler import StepLR


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss()

    model1 = BaseCNN().to(device)
    model2 = ImprovedCNN().to(device)
    model3 = get_pretrained_alexnet().to(device)

    for param in model3.alexnet.features.parameters():
        param.requires_grad = False

    models_to_train = [
        ("Base CNN", model1, optim.Adam(model1.parameters(), lr=0.001)),
        ("Improved CNN", model2, optim.Adam(model2.parameters(), lr=0.001)),
        ("AlexNet (Pretrained)", model3, optim.Adam(model3.alexnet.classifier.parameters(), lr=0.0001))
    ]

    epochs = 30
    for model_name, model, optimizer in models_to_train:
        print(f"\n--- Training {model_name} ---")

        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc = test_model(model, testloader, criterion, device)

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.5f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print("\n--- Model 4: Feature Extraction for Hybrid Model ---")
    extract_features(model2, trainloader, device, save_prefix="train")
    extract_features(model2, testloader, device, save_prefix="test")

    print("\n--- Training Hybrid Model ---")
    train_hybrid_svm()


if __name__ == "__main__":
    main()
