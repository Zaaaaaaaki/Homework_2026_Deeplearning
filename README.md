# CNN on MNIST with Activation Function Ablation
## Author
洪子康，韩晴，司博涵

## Project Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset using **PyTorch**. The implementation is based on a classic LeNet-style architecture and includes an **activation function ablation study** to investigate the influence of different activation functions on network performance.

Three activation functions are evaluated:

- ReLU
- Sigmoid
- Tanh

The experiments compare their convergence behavior and final classification accuracy under identical network architecture and training settings.

---

## Network Architecture

The CNN consists of:

```
Input (1×28×28)
        │
Conv2d(1 → 10, 5×5)
        │
Activation
        │
MaxPool2d(2×2)
        │
Conv2d(10 → 20, 5×5)
        │
Activation
        │
MaxPool2d(2×2)
        │
Flatten (320)
        │
Linear(320 → 50)
        │
Activation
        │
Linear(50 → 10)
```

---

## Experimental Settings

| Parameter | Value |
|-----------|------:|
| Dataset | MNIST |
| Batch Size | 256 |
| Learning Rate | 0.01 |
| Momentum | 0.5 |
| Epoch | 10 |
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |

---

## Activation Function Ablation

Three experiments were conducted under the **same network architecture and hyperparameters**:

| Activation | Final Test Accuracy |
|------------|--------------------:|
| ReLU | **97.99%** |
| Sigmoid | **11.35%** |
| Tanh | **96.34%** |

### Analysis

- **ReLU** achieves the fastest convergence and the highest accuracy because it effectively alleviates the gradient vanishing problem.
- **Sigmoid** almost fails to converge. Under the same learning rate and training epochs, the gradients quickly vanish, making parameter updates extremely slow.
- **Tanh** performs significantly better than Sigmoid since its output is zero-centered, but it still suffers from saturation and converges more slowly than ReLU.

These results demonstrate why ReLU has become the standard activation function in modern convolutional neural networks.

---

## Project Structure

```
.
├── data/
│   └── mnist_jpg/
│       ├── train_xxx_label.jpg
│       └── test_xxx_label.jpg
│
├── cnn_mnist.ipynb
├── README.md


---

## Dataset Format

Images are stored as JPG files with the following naming convention:

```
train_1_5.jpg
train_2_0.jpg
...
test_1_8.jpg
```

where:

- `train` / `test` indicates the dataset split.
- The last number represents the digit label.

---

## Running the Code

```bash
python cnn_ablation.py
```

The program automatically performs three experiments:

1. ReLU
2. Sigmoid
3. Tanh

and outputs:

- Training Loss
- Training Accuracy
- Test Accuracy
- Loss Curve
- Accuracy Curve

---

## Dependencies

- Python 3.10+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pillow

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Author

Hong Zekang

Course Project — Convolutional Neural Networks (CNN)