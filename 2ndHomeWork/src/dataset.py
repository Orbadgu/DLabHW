import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    # Ön İşleme (Preprocessing)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Görüntüyü hafifçe kaydırır
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 15 dereceye kadar rastgele döndürür
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 Veri Seti
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader