#  Deep Autoencoder for FashionMNIST Image Reconstruction

This project demonstrates the construction, training, and evaluation of a **deep fully connected autoencoder** using PyTorch. The autoencoder is trained on the **FashionMNIST** dataset to learn efficient low-dimensional representations of grayscale fashion images and reconstruct them from these representations.

---

##  What is an Autoencoder?

An **autoencoder** is a type of neural network used for **unsupervised learning**, primarily for **dimensionality reduction**, **denoising**, or **image compression**. It works by encoding the input into a lower-dimensional space and then decoding it back to reconstruct the original input.

---

###  Features

- Loads FashionMNIST dataset using `torchvision.datasets`
- Defines a **deep symmetric autoencoder** with 5 encoding and 5 decoding layers
- Uses **MSELoss** as the reconstruction criterion
- Trains the model using the **Adam** optimizer
- Saves output reconstructions every 5 epochs during training
- Produces a final comparison image and training loss plot

---

## Model Architecture

### Encoder (input: 784 = 28×28 image):
- `Linear(784 → 256)` → ReLU
- `Linear(256 → 128)` → ReLU
- `Linear(128 → 64)` → ReLU
- `Linear(64 → 32)` → ReLU
- `Linear(32 → 16)` → ReLU

### Decoder (mirror of encoder):
- `Linear(16 → 32)` → ReLU
- `Linear(32 → 64)` → ReLU
- `Linear(64 → 128)` → ReLU
- `Linear(128 → 256)` → ReLU
- `Linear(256 → 784)` → ReLU


---

## 🛠️ Setup & Dependencies

Install dependencies:

```bash
pip install torch torchvision matplotlib
