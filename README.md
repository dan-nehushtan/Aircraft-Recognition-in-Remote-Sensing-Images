<div align="center">

# ✈️ Aircraft Recognition in Remote Sensing Images

**Deep Learning for Fine-Grained Aircraft Classification**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

*MECH 3465 — Robotics & Machine Intelligence Coursework*

**Authors:** Dan Nehushtan · Louie Burns

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Development Journey](#-development-journey)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)

---

## 🔍 Overview

This project explores **fine-grained aircraft classification** from remote sensing imagery using deep learning. Starting from a minimal single-layer CNN baseline, we iteratively developed and evaluated increasingly sophisticated models — culminating in experiments with **ResNet50** transfer learning and **Vision Transformers (ViT)**.

The primary task focused on classifying aircraft by **manufacturer** (e.g., Boeing, Airbus, Cessna), with additional exploration of **family** and **variant** classification hierarchies.

### Objectives

- Build a baseline CNN classifier for aircraft manufacturer recognition
- Systematically improve performance through architectural and training enhancements
- Explore transfer learning with pre-trained models (ResNet50, ViT)
- Analyse the impact of data preprocessing, augmentation, and class balancing strategies

---

## 📊 Dataset

The project uses the **FGVC Aircraft Recognition Dataset** — a benchmark for fine-grained visual categorisation containing **10,000 images** of aircraft.

| Split | Images |
|:------|-------:|
| Train | 3,334 |
| Validation | 3,333 |
| Test | 3,333 |
| **Total** | **10,000** |

### Classification Hierarchies

| Level | Classes | Description | Example |
|:------|--------:|:------------|:--------|
| **Manufacturer** | 30 | Aircraft maker | Boeing, Airbus, Cessna |
| **Family** | 70 | Aircraft model family | Boeing 737, A320, F-16 |
| **Variant** | 100 | Specific model variant | 737-800, A320, 777-300 |

### Manufacturers in Dataset

> Airbus · ATR · Antonov · Beechcraft · Boeing · Bombardier Aerospace · British Aerospace · Canadair · Cessna · Cirrus Aircraft · Dassault Aviation · Dornier · Douglas Aircraft Company · Embraer · Eurofighter · Fairchild · Fokker · Gulfstream Aerospace · Ilyushin · Lockheed Corporation · Lockheed Martin · McDonnell Douglas · Panavia · Piper · Robin · Saab · Supermarine · Tupolev · Yakovlev · de Havilland

---

## 🎯 Approach

### Strategy

The manufacturer classification task was chosen strategically — it offers **fewer classes (30)** than family (70) or variant (100) while providing more balanced class distributions. For the final model, the **top 5 manufacturers by sample count** were dynamically selected to ensure adequate training data per class.

### Progressive Improvement Pipeline

```
Baseline CNN          →    Data Augmentation     →    Batch Normalisation
(1 conv, 64×64)            (flip, rotation)            + Deeper Network
        ↓                        ↓                          ↓
  LR Scheduling        →    Transfer Learning     →    Vision Transformer
  (StepLR decay)            (ResNet50)                  (ViT - HuggingFace)
```

---

## 🏗️ Model Architectures

### 1. Baseline — SimpleCNN

The minimal starting point for benchmarking.

```
Input (3×64×64)
  → Conv2d(3→16, 3×3) → ReLU → MaxPool2d(2×2)
  → Flatten → Linear → Output (5 classes)
```

| Parameter | Value |
|:----------|:------|
| Image Size | 64×64 |
| Optimizer | SGD (lr=0.01) |
| Batch Size | 64 |
| Epochs | 15 |
| Augmentation | None |

### 2. Improved — Enhanced CNN

Deeper architecture with regularisation and modern training techniques.

```
Input (3×128×128)  
  → Conv2d(3→16) → BatchNorm → ReLU → MaxPool
  → Conv2d(16→32) → BatchNorm → ReLU → MaxPool
  → Conv2d(32→64) → BatchNorm → ReLU → MaxPool
  → Flatten → Linear(128) → Dropout(0.5) → Output (5 classes)
```

| Parameter | Value |
|:----------|:------|
| Image Size | 128×128 |
| Optimizer | SGD (lr=0.01) |
| Scheduler | StepLR (step=9, γ=0.5) |
| Batch Size | 32 |
| Epochs | 40 |
| Augmentation | RandomHorizontalFlip, RandomRotation(5°) |
| Normalisation | Mean=0.5, Std=0.5 |

### 3. ResNet50 — Transfer Learning

Pre-trained ResNet50 with fine-tuned classification head.

| Parameter | Value |
|:----------|:------|
| Image Size | 224×224 |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | StepLR (step=5, γ=0.1) |
| Batch Size | 16 |
| Label Smoothing | 0.1 |
| Augmentation | Flip, Rotation(15°), ColorJitter |
| Normalisation | ImageNet statistics |

### 4. Vision Transformer (ViT)

Hugging Face `ViTForImageClassification` fine-tuned from `dima806/military_aircraft_image_detection`.

| Parameter | Value |
|:----------|:------|
| Image Size | 224×224 |
| Optimizer | AdamW (lr=2e-7) |
| Batch Size | 64 (train) / 32 (eval) |
| Epochs | 3 |
| Weight Decay | 0.02 |
| Warmup Steps | 50 |
| Class Balancing | RandomOverSampler |

### 5. Custom Residual CNN (EnhancedCNN)

A custom-designed residual network with skip connections.

```
Input (3×224×224)
  → ResidualBlock(3→32) → ResidualBlock(32→64) → ResidualBlock(64→128)
  → AdaptiveAvgPool2d → Flatten → Linear → Output
```

| Parameter | Value |
|:----------|:------|
| Image Size | 224×224 |
| Optimizer | Adam (lr=0.001) |
| Scheduler | StepLR (step=5, γ=0.5) |
| Epochs | 100 |
| Augmentation | Flip, Rotation(15°), ColorJitter, Affine, Grayscale |

---

## 📈 Results

### Improved Model — Training Progress (50 Epochs)

| Metric | Best Value | At Epoch |
|:-------|:-----------|:---------|
| **Train Accuracy** | 95.7% | 40 |
| **Test Accuracy** | 52.6% | 44 |
| **F1 Score** | 0.514 | 45 |
| **Train Loss** | 0.299 | 40 |
| **Test Loss** | 1.232 | 22 |

### Training Curve Summary

```
Epoch   Train Acc   Test Acc   F1 Score   Learning Rate
─────   ─────────   ────────   ────────   ─────────────
  1      33.1%      25.9%      0.129      0.01
  5      61.3%      28.6%      0.167      0.005
 10      77.6%      38.6%      0.356      0.0025
 18      90.0%      51.4%      0.503      0.00125
 25      93.9%      51.7%      0.505      0.0003125
 35      95.3%      52.2%      0.511      7.81e-05
 50      95.6%      52.4%      0.513      9.77e-06
```

### Baseline vs Improved — Comparison

| Metric | Baseline | Improved | Improvement |
|:-------|:---------|:---------|:------------|
| Architecture | 1-conv CNN | 3-conv CNN + BN | Deeper + regularised |
| Image Resolution | 64×64 | 128×128 | 4× more pixels |
| Best F1 Score | ~0.38 | ~0.51 | **+34% relative** |
| Training Epochs | 15 | 40–50 | Longer training |
| LR Schedule | Fixed | StepLR decay | Adaptive |
| Augmentation | None | Flip + Rotation | Increased variety |

---

## 📁 Repository Structure

```
📦 Aircraft-Recognition-in-Remote-Sensing-Images
├── 📂 Actual Coursework/
│   ├── 📂 FINAL RESULTS/                    ← Final submitted models & metrics
│   │   ├── 📓 Baseline Model.ipynb           ← SimpleCNN baseline
│   │   ├── 📓 Improved Model.ipynb           ← Enhanced CNN with improvements
│   │   ├── 📊 Epoch_Metrics_Table.csv        ← Training metrics (50 epochs)
│   │   └── 🏆 best_model.pth                ← Saved best model weights
│   │
│   ├── 📂 Code Attempts/                    ← 23 iterative development versions (V1–V23)
│   │
│   ├── 📂 Code Attempts - Manufacturer Instead/
│   │   ├── 📓 V01 - Base Case.ipynb          ← Initial manufacturer classification
│   │   ├── 📓 V03 - Added Batch Normalisation.ipynb
│   │   ├── 📓 V06 - Added layers and dropout.ipynb
│   │   ├── 📓 V08 - External Models.ipynb    ← ResNet50 transfer learning
│   │   ├── 📓 V12 - Exclude Boeing.ipynb     ← Class imbalance experiments
│   │   └── ... (14 versions total)
│   │
│   ├── 📂 Code Attempts - Analysis and Improvement/
│   │   ├── 📓 V01 - Baseline Code.ipynb      ← Family classification baseline
│   │   ├── 📓 V05 - Add more layers.ipynb
│   │   ├── 📓 ChatGPT Improvements (V6).ipynb ← Custom ResidualBlock CNN
│   │   └── ... (7 versions total)
│   │
│   ├── 📂 Hugging Face/                     ← ViT transformer experiments
│   ├── 📂 Lecturer Code Attempts/           ← Template-based experiments
│   └── 📂 dataoriginal/                     ← Dataset labels & annotations
│       ├── families.txt                      ← 70 aircraft families
│       ├── manufacturers.txt                 ← 30 manufacturers
│       ├── variants.txt                      ← 100 variants
│       ├── images_box.txt                    ← Bounding box annotations
│       └── images_{task}_{split}.txt         ← Train/val/test splits
│
├── 📂 Group_1_BurnsNehushtan/               ← Final group submission
│   ├── 📓 Code.ipynb                        ← Complete submitted notebook
│   ├── 📄 Code.pdf                          ← PDF export
│   └── 📄 Report.docx                      ← Written analysis report
│
├── 📂 Coursework 1/                         ← Earlier development & exploration
│   ├── 📓 V2 - military-aircraft-detection-vit.ipynb  ← ViT experiments
│   ├── 📓 V8 - Bounding Boxes.ipynb         ← Bounding box preprocessing
│   └── ... (11 versions)
│
└── 📂 Lab/                                  ← Lab templates & exercises
```

---

## 🔄 Development Journey

The project evolved through **50+ notebook iterations** across multiple classification strategies:

### Phase 1 — Initial Exploration
- Started with family classification (70 classes) — proved too fine-grained for a simple CNN 
- Explored bounding box preprocessing to crop aircraft from full images
- Investigated the dataset structure and class distributions

### Phase 2 — Strategic Pivot to Manufacturer Classification
- Switched to manufacturer classification (30 classes → top 5 selected)
- Built baseline SimpleCNN as a benchmark
- Identified Boeing and Airbus class dominance as a key challenge

### Phase 3 — Systematic Improvement
- **V03:** Added Batch Normalisation → improved training stability
- **V04:** Increased batch size → smoother gradient updates
- **V05–V06:** Deeper network + Dropout → better feature extraction + regularisation
- **V07:** Increased image resolution (64→128) → captured finer details
- **Learning rate scheduling** (StepLR) → prevented overshooting during convergence

### Phase 4 — Advanced Approaches
- **ResNet50** transfer learning with ImageNet pre-training
- **Vision Transformer (ViT)** fine-tuning via Hugging Face
- **Custom Residual CNN** with skip connections
- Class exclusion experiments (removing Boeing/Airbus) to study imbalance effects

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib scikit-learn pandas numpy
pip install transformers datasets  # For ViT experiments
```

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Aircraft-Recognition-in-Remote-Sensing-Images.git
   cd Aircraft-Recognition-in-Remote-Sensing-Images
   ```

2. **Download the FGVC Aircraft Dataset**
   - Place images in the expected directory structure
   - Label files are provided in `Actual Coursework/dataoriginal/`

3. **Run the final models**
   - Open `Actual Coursework/FINAL RESULTS/Baseline Model.ipynb` for the baseline
   - Open `Actual Coursework/FINAL RESULTS/Improved Model.ipynb` for the improved model
   - Open `Group_1_BurnsNehushtan/Code.ipynb` for the complete submission

### Hardware

- Models were developed and tested with **CUDA GPU** support
- The notebooks include automatic GPU detection (`torch.cuda.is_available()`)
- CPU training is supported but significantly slower

---

## 💡 Key Findings

1. **Simple CNNs struggle with fine-grained classification** — A single-layer CNN on 64×64 images achieved only ~38% F1 on 5-class manufacturer recognition, near random chance

2. **Image resolution matters significantly** — Doubling from 64×64 to 128×128 provided meaningful accuracy gains by preserving discriminative aircraft features

3. **Batch Normalisation + Dropout** were the most impactful single architectural additions, improving both training stability and generalisation

4. **Learning rate scheduling is essential** — StepLR decay (halving every 9 epochs) allowed the model to converge to better optima than fixed-rate training

5. **Class imbalance is a major challenge** — Boeing and Airbus dominate the dataset; intelligent class selection and balancing strategies (like RandomOverSampler) are critical

6. **The gap between train and test accuracy (~43%)** indicates overfitting remains a challenge, suggesting the dataset may be too small for the model complexity or that stronger augmentation / regularisation is needed

7. **Transfer learning (ResNet50, ViT)** represents the most promising direction for future work, leveraging features learned from millions of images

---

## 🛠️ Technologies Used

| Technology | Purpose |
|:-----------|:--------|
| **Python 3.8+** | Core language |
| **PyTorch** | Deep learning framework |
| **torchvision** | Image transforms, pre-trained models |
| **Hugging Face Transformers** | ViT model & training |
| **scikit-learn** | Metrics (F1, confusion matrix), oversampling |
| **Matplotlib** | Visualisation & plots |
| **Pandas / NumPy** | Data handling |
| **Jupyter Notebook** | Development environment |
| **CUDA** | GPU acceleration |

---

<div align="center">

**MECH 3465 — Robotics & Machine Intelligence**  
University of Leeds · 2024/25

</div>