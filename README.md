# 🐶 Puppy Finding Problem – Deep Learning for Dog Breed Classification

The task of this project was to help a kennel owner, Paco, distinguish between escaped Golden Retrievers and Collies using machine learning. Given a dataset of dog images across multiple breeds, I implemented, trained, and evaluated several deep learning models for image classification—experimenting with Convolutional Neural Networks (CNNs), Transfer Learning, and Vision Transformers (ViTs).

## 📁 Project Overview

### Objective
Build a binary image classifier to distinguish **Golden Retrievers** from **Collies**, using a dataset of 10 dog breeds and advanced deep learning techniques.

### Dataset
- ~12,775 64×64 RGB dog images across 10 breeds.
- Images split into training, validation, test, and challenge partitions.

## 🧠 Techniques Explored

### ✅ Data Preprocessing
- Calculated per-channel mean and standard deviation on the training set to avoid data leakage.
- Standardized all image partitions using training-set statistics.

### 🧱 Convolutional Neural Network (CNN)
- Designed and trained a 3-layer CNN from scratch.
- Evaluated training vs. validation AUROC to monitor overfitting.
- Experimented with different patience values for early stopping (5 vs. 10).
- Increased filters in the final convolutional layer from 8 → 64 to evaluate overfitting behavior.
- Final CNN (8 filters) performance:
  - **Train AUROC:** 0.9973
  - **Validation AUROC:** 0.9744
  - **Test AUROC:** 0.7516

### 🔄 Transfer Learning
- Pretrained a CNN on 8 auxiliary dog breeds.
- Transferred convolutional weights to the binary classifier.
- Evaluated 4 transfer strategies by freezing 0–3 convolutional layers.
- Best performance achieved when freezing only the first conv layer:
  - **Test AUROC improved to:** 0.7952
- Demonstrated strong generalization over training from scratch.

### 🤖 Vision Transformers (ViT)
- Implemented a lightweight Vision Transformer with:
  - Patch embedding
  - Learnable [CLS] token
  - Positional encoding
  - 2 Transformer blocks and 2 attention heads
- ViT achieved strong validation performance:
  - **Validation AUROC:** 0.9659
  - However, **Test AUROC dropped to:** 0.5524 due to overfitting and limited training data
- Compared to CNNs, the ViT had fewer parameters (~11.5k vs. ~39.7k) but underperformed due to lack of inductive biases and data-hungriness.

### 🧪 Challenge Model (Custom Design)
- Created a custom CNN leveraging pretrained convolutional weights from the source model.
- Fine-tuned all layers and optimized with:
  - AdamW optimizer
  - Weight decay: 1e-3
  - No dropout (was found to hurt performance)
- Used validation AUROC for model checkpoint selection.
- Final challenge performance:
  - **Validation AUROC:** 0.9237
  - **Test AUROC:** 0.7956

## 🛠️ Tools & Technologies
- Python, NumPy, PyTorch
- Matplotlib (for visualizations)
- Jupyter Notebooks
- Git for version control

## 📈 Key Learnings
- Importance of proper preprocessing and avoiding data leakage.
- Transfer learning can significantly boost generalization when target data is limited.
- Vision Transformers require more data to outperform CNNs.
- Architectural simplicity + smart pretraining > complex models for small datasets.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   conda create --name 445p2 --file requirements.txt
   conda activate 445p2
