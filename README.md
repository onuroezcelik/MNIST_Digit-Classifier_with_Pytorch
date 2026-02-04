# MNIST Digit Classifier with Pytorch

## Project Overview

This project implements a fully connected neural network using **PyTorch** to classify handwritten digits from the **MNIST dataset**. The MNIST dataset contains 60,000 training images and 10,000 test images of digits 0–9, each of size 28x28 pixels.

The goal of this project is to build, train, evaluate, and improve a neural network model to achieve high classification accuracy (>90%) on the test set.

---

## Project Steps

### 1. Data Loading and Preprocessing

* **MNIST dataset** is loaded using `torchvision.datasets.MNIST`.
* Images are converted to **PyTorch tensors** using `transforms.ToTensor()`.
* Data is **normalized** using the training set mean and standard deviation `(mean=0.1307, std=0.3081)`.
* Flattening of images (`28x28 → 784`) is done **inside the model** using `nn.Flatten()`.

**Reasoning:**
Normalization stabilizes gradients and speeds up training. Flattening prepares the 2D images for fully connected layers.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

---

### 2. Data Exploration

* **Data shapes and sizes** are inspected:

  ```python
  print(len(train_dataset), len(test_dataset))  # 60000, 10000
  print(train_dataset[0][0].shape)  # torch.Size([1, 28, 28])
  ```
* Example images are visualized using a helper function `show5()` and `matplotlib.pyplot`.

---

### 3. Neural Network Architecture

* **Fully connected network** with **2 hidden layers**:

  * `fc1`: 784 → 128
  * `fc2`: 128 → 64
  * `fc3`: 64 → 10 (output classes)
* **Activation function:** ReLU for hidden layers
* **Flatten layer** converts 28x28 images into 784-dimensional vectors.

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # logits
```

* Optional for probability output: `F.softmax(x, dim=1)` in forward method.

---

### 4. Loss Function and Optimizer

* **Loss function:** `nn.CrossEntropyLoss()` → suitable for multi-class classification.
* **Optimizer:** `optim.Adam(model.parameters(), lr=0.001)` → updates model parameters to minimize loss.

---

### 5. Model Training

* Model is trained for **5 epochs** (adjustable) on the training set.
* **Average loss per epoch** is tracked.
* **GPU support** is used if available:

  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```

---

### 6. Model Evaluation

* Model is set to **evaluation mode**: `model.eval()`

* Predictions are obtained using the **test DataLoader**.

* **Accuracy** is computed by comparing predicted classes to true labels:

  ```python
  correct = 0
  total = 0
  with torch.no_grad():
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')
  ```

* Achieved **Test Accuracy:** 96.93%

---

### 7. Hyperparameter Tuning

* Hyperparameters modified to improve accuracy:

  * Learning rate
  * Hidden layer sizes
  * Batch size
  * Number of epochs

These adjustments helped achieve **>90% classification accuracy**, meeting project requirements.

---

### 8. Saving and Loading the Model

* Save trained model:

  ```python
  torch.save(model.state_dict(), "mnist_model.pth")
  ```
---
