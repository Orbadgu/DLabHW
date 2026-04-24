### 1. Introduction
In this study, feature extraction and classification were performed on image data using Convolutional Neural Networks. The CIFAR-10 benchmark dataset, containing 10 different object classes, was selected for this task. The scope of the work includes a comparative analysis of a custom-designed architecture based on LeNet-5, a deeper version with optimized hyperparameters (Improved CNN), and the well-established AlexNet architecture utilizing transfer learning. Additionally, a hybrid classification system was designed by training a canonical machine learning model, Support Vector Machines (SVM), using the feature sets obtained from the Improved CNN feature extraction mechanism. This report examines the theoretical frameworks, experimental results, and relative advantages of the developed architectures.

### 2. Method
The project was designed following reproducibility principles, and the dataset was configured to be downloaded dynamically at runtime instead of being hosted in the GitHub repository.

**2.1. Dataset and Preprocessing**
The CIFAR-10 dataset, consisting of 32x32 pixel RGB images, was used. To prevent overfitting, aggressive data augmentation was applied to the training set through Random Crop, Random Horizontal Flip, and Random Rotation. Data were normalized on a per-channel basis before being fed into the architectures.

**2.2. Architectures and Theoretical Framework**

* **Base CNN:** This architecture consists of 2 convolutional layers, 2 max-pooling layers, and 3 linear layers, inspired by the LeNet-5 design. Convolutional layers extract local features such as edges and textures, while pooling layers reduce dimensions to decrease computational load and provide spatial invariance.
* **Improved CNN:** The capacity of the base architecture was increased to 3 convolutional blocks with 32, 64, and 128 filters. To ensure stability, Batch Normalization was applied after each convolution, and Dropout layers with a 0.3 rate were added between fully connected layers to prevent memorization.
* **Pretrained AlexNet:** The pretrained AlexNet model from the torchvision library was utilized. The initial feature extraction layers were frozen to avoid catastrophic forgetting, and only the final classifier layer was modified and trained for the 10 CIFAR-10 classes. CIFAR-10 images were scaled to 224x224 dimensions using interpolation.
* **Hybrid CNN and SVM System:** Tensors passing through the feature extraction part of the Improved CNN were saved to disk in npy format (Training: 50000x2048, Test: 10000x2048). These vectors were fed into a LinearSVC algorithm, which draws a linear decision boundary and is optimized for large datasets.

**2.3. Training Parameters**
The architectures were trained in a CPU environment using Cross-Entropy Loss and the Adam optimizer for 30 epochs. To ensure stable progress on the loss surface, a StepLR scheduler was used to reduce the learning rate by half every 10 epochs, starting from an initial rate of 0.001.

### 3. Results
The accuracy rates at the end of the training processes are presented in Table 1. The AlexNet architecture was trained for 5 epochs to preserve its pretrained weights.

**Table 1: Training and Test Accuracy Rates**

| Model | Parameter Optimization | Training Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **Base CNN** | 30 Epochs, StepLR | 71.30% | 73.11% |
| **Improved CNN** | 30 Epochs, StepLR | 78.73% | 81.40% |
| **Pretrained AlexNet** | 5 Epochs, Frozen Features | 89.70% | 88.16% |
| **Hybrid CNN and SVM System** | LinearSVC, C=0.1 | - | 78.05% |

The Classification Report obtained for the Hybrid CNN and SVM System, powered by the features of the Improved CNN, is provided in Table 2.

**Table 2: Hybrid System Classification Report**

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| 0 Airplane | 0.76 | 0.82 | 0.79 | 1000 |
| 1 Automobile | 0.87 | 0.90 | 0.89 | 1000 |
| 2 Bird | 0.72 | 0.68 | 0.70 | 1000 |
| 3 Cat | 0.60 | 0.61 | 0.61 | 1000 |
| 4 Deer | 0.73 | 0.78 | 0.75 | 1000 |
| 5 Dog | 0.72 | 0.68 | 0.70 | 1000 |
| 6 Frog | 0.83 | 0.82 | 0.83 | 1000 |
| 7 Horse | 0.84 | 0.80 | 0.82 | 1000 |
| 8 Ship | 0.87 | 0.87 | 0.87 | 1000 |
| 9 Truck | 0.85 | 0.85 | 0.85 | 1000 |

### 4. Discussion
Based on the experimental results, the following conclusions were reached:

* **Superiority of Depth:** While the **Base CNN** with two convolutional blocks achieved 73.11% accuracy, the **Improved CNN** reached 81.40% accuracy. This proves that increasing the number of layers allows the model to extract more abstract and high-level features.
* **Regularization and Generalization:** The combination of Data Augmentation and Dropout successfully balanced training and test accuracy for the **Improved CNN**. The network learned general patterns instead of memorizing, resulting in higher success on unseen data.
* **Efficiency of the Hybrid System:** Compared to the end-to-end trained **Improved CNN**, the **Hybrid CNN and SVM System** displayed a very close performance of 78.05%. The report indicates high success in classes with sharp geometric lines like automobiles and ships, while it struggled to separate visually similar classes like cats and dogs on a linear plane.
* **Transfer Learning Comparison:** **Pretrained AlexNet** reached 88.16% accuracy in only 5 epochs. Although this shows the superiority of large-scale pretraining, the 81.40% success of the custom **Improved CNN** is an efficient alternative considering its lower parameter count and training cost.

### 5. References
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.

PyTorch Documentation. (2026). *Models and pre-trained weights*. Retrieved from https://pytorch.org/vision/0.9/models.html

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, pp. 2825-2830.
