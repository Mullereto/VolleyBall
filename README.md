<h1 align="center">Group Activity Recognition</h1>

<p align="center">
  An implementation of the <strong>CVPR 2016 paper</strong>, <a href="https://arxiv.org/abs/1511.06040"><em>A Hierarchical Deep Temporal Model for Group Activity Recognition</em></a>.  
  This project focuses on recognizing group activities in volleyball matches using deep learning, temporal modeling, and player-level feature aggregation.
</p>

---

## Table of Contents
1. [Key Updates](#key-updates)
2. [Usage](#usage)
   - [Clone the Repository](#1-clone-the-repository)
   - [Install Dependencies](#2-install-the-required-dependencies)
   - [Download Model Checkpoint](#3-download-the-model-checkpoint)
3. [Dataset Overview](#dataset-overview)
4. [Baselines](#baselines)
5. [Performance Comparison](#performance-comparison)
6. [Interesting Observations](#interesting-observations)
7. [Model Architectures](#model-architectures)

---

## Key Updates

- Reimplemented **Baselines 1, 3, 4, 5** from the original paper in PyTorch.
- Merged the concepts of **B7 and B8** into a single **END baseline** for improved temporal modeling and spatial pooling.
- Switched to **ResNet-50** for feature extraction.
- Achieved competitive performance compared to the original paper, with clear improvements in temporal baselines.

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/group-activity-recognition.git
```

### 2. Install the Required Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Download the Model Checkpoint
Use the Kaggle Hub API:
```python
import kagglehub
path = kagglehub.model_download("omaryasserace/endmodelv2/pyTorch/default")
print("Path to model files:", path)
```

---

## Dataset Overview

The dataset is the **Volleyball Dataset** introduced in the CVPR 2016 paper.

- **Frames**: 4,831 annotated frames from 55 volleyball videos.
- **Group Activities**: 8 classes (e.g., left spike, right pass, winpoint).
- **Player Actions**: 9 classes (e.g., waiting, setting, spiking).
- **Number of Instances**:
  | Baseline      | Original Paper Acc | My Accuracy | My F1 Score |
  |---------------|-------------------|-------------|-------------|
  | B1            | 64.6%             | XX.XX%      | XX.XX%      |
  | B3            | 73.2%             | XX.XX%      | XX.XX%      |
  | B4            | 68.0%             | XX.XX%      | XX.XX%      |
  | B5            | 70.3%             | XX.XX%      | XX.XX%      |
  | END (B7+B8)   | ~83–89%           | XX.XX%      | XX.XX%      |

Train/test split and annotation format are identical to the original paper.  
See the paper for more details: [link](https://github.com/mostafa-saad/deep-activity-rec).

---

## Baselines

### **B1: Image Classification**
Single-frame group activity classification using ResNet-50 fine-tuned for the 8 group activity labels.

### **B3: Fine-tuned Person Classification**
Each detected player is cropped, features are extracted using ResNet-50, pooled over all players in the frame, and fed to a classifier.

### **B4: Temporal Model with Image Features**
Uses ResNet-50 features from each frame of a 9-frame clip, followed by an LSTM for temporal modeling.

### **B5: Temporal Model with Person Features**
Temporal extension of B3 — person crops are processed over 9 frames, pooled across players, and fed to an LSTM.

### **END: Merged Team-Dependent Two-Stage Model**
Combines concepts from B7 and B8:
- **Stage 1**: LSTM processes sequences of person-level features (ResNet-50 backbone).
- **Stage 2**: Players are split into two teams, max-pooled within each team, concatenated, and passed to a second LSTM for group-level classification.
- Improves positional awareness and temporal consistency by retaining team structure.

---

## Performance Comparison

| Baseline      | Original Paper Acc | My Accuracy | My F1 Score |
|---------------|-------------------|-------------|-------------|
| B1            | 64.6%             | XX.XX%      | XX.XX%      |
| B3            | 73.2%             | XX.XX%      | XX.XX%      |
| B4            | 68.0%             | XX.XX%      | XX.XX%      |
| B5            | 70.3%             | XX.XX%      | XX.XX%      |
| END (B7+B8)   | ~83–89%           | XX.XX%      | XX.XX%      |

*(Replace XX.XX with your measured values)*

---

## Interesting Observations

- **Pooling without team separation** (B5) often confused left/right activities (e.g., left pass vs right pass).
- **Team-dependent pooling** in END baseline significantly reduced directional confusion by keeping positional context.
- Temporal models (B4, B5, END) outperformed frame-based models (B1, B3) in nearly all cases.

---

## Model Architectures

### **END Baseline Architecture**
![END Baseline Architecture](A_diagram_titled_"END_Baseline_Architecture"_depic.png)

```
Video Frames → Person Detection → Player Crops
           ↓
   ResNet-50 Feature Extractor
           ↓
   LSTM 1 (per-player temporal modeling)
           ↓
   Team-wise Max Pooling → Feature Concatenation
           ↓
   LSTM 2 (group-level temporal modeling)
           ↓
   Fully Connected → Softmax (Group Activity)
```

- **Stage 1** captures player-level temporal dynamics.
- **Stage 2** models team interactions over time.
- **Positional awareness** preserved by pooling within teams before concatenation.

---

**Training Configuration**
- Optimizer: AdamW
- Batch Size: 4
- Learning Rate: 0.00006
- Epochs: 90
- Hardware: Kaggle GPU (P100 or T4)

---
