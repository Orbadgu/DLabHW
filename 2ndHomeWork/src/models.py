import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# MODEL 1: LeNet-5 Benzeri Temel CNN
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # CIFAR-10 giriş: 3x32x32
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x16

            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# MODEL 2: Derinleştirilmiş ve Optimize Edilmiş CNN (V3)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # 1. Blok (32x32 -> 16x16)
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2. Blok (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3. Blok - YENİ EKLENDİ (8x8 -> 4x4)
            # Çözünürlük düştüğü için kernel_size=3 yapıp daha ince detaylara odaklanıyoruz
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 3. havuzlamadan sonra boyut 4x4'e düştü (128 filtre * 4 * 4 = 2048)
            nn.Linear(128 * 4 * 4, 256),  # Orta katman kapasitesini 120'den 256'ya çıkardık
            nn.ReLU(),
            nn.Dropout(0.3),  # 0.5'ten 0.3'e düşürdük (Model artık daha rahat öğrenecek)
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.3),  # 0.5'ten 0.3'e düşürdük
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# MODEL 3: Hazır Model (AlexNet)
class PretrainedAlexNet(nn.Module):
    def __init__(self):
        super(PretrainedAlexNet, self).__init__()
        # PyTorch'tan pretrained AlexNet'i indir
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # Son katmanı CIFAR-10 (10 sınıf) için değiştir
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, 10)

    def forward(self, x):
        # CIFAR-10'un 32x32 olan görüntülerini, AlexNet'in beklediği 224x224 boyutuna büyüt
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.alexnet(x)

def get_pretrained_alexnet():
    return PretrainedAlexNet()