# Retinal Screening Using EfficientNetB0

## Overview

This project presents a lightweight deep learning approach for **binary retinal disease screening** using retinal images from the **OLIVES dataset**.

The proposed system uses an **ImageNet-pretrained EfficientNetB0** model as a frozen feature extractor, followed by a lightweight classification head for distinguishing between:

* **Normal (0)**
* **Disease (1)**

A key aspect of this project is the use of a **patient-wise data split**, ensuring that images from the same patient do not appear across the training, validation, and test sets. This provides a more realistic evaluation of the model's ability to generalize to previously unseen patients.

The project is based on the research work:

> **Patient-Wise Validated EfficientNetB0 for Binary Retinal Disease Screening on the OLIVES Dataset**

---

## Key Features

* Binary retinal disease classification
* Patient-wise train/validation/test splitting
* ImageNet-pretrained EfficientNetB0
* Frozen EfficientNetB0 backbone
* Lightweight trainable classification head
* Retinal image preprocessing
* Evaluation on images from unseen patients
* Accuracy, ROC-AUC, precision, recall, BCE loss, and confusion matrix analysis
* Designed as a research baseline rather than a clinically deployed diagnostic system

---

## Dataset

The project uses the **OLIVES dataset**.

According to the associated research paper, the dataset contains:

| Description         |  Value |
| ------------------- | -----: |
| Total images        | 78,822 |
| Total patients      |     87 |
| Training patients   |     60 |
| Validation patients |     13 |
| Test patients       |     14 |
| Training images     | 54,323 |
| Validation images   | 10,486 |
| Test images         | 13,707 |

### Patient-Wise Splitting

The dataset is divided at the **patient level** rather than randomly at the image level:

```text
87 Patients
│
├── 60 Patients → Training
├── 13 Patients → Validation
└── 14 Patients → Testing
```

This prevents images belonging to the same patient from appearing in both training and testing data.

The final test evaluation is therefore performed on retinal images from **14 previously unseen patients**.

> **Note:** The dataset itself is not included in this repository.

---

## Image Preprocessing

The original retinal images are:

* Single-channel TIFF images
* Original resolution: **504 × 496**
* Resized to **224 × 224**
* Converted from one channel to three channels by replication
* Pixel values normalized by dividing by 255

The resulting images are compatible with the EfficientNetB0 input configuration.

### Preprocessing Pipeline

```text
Original TIFF Image
        ↓
504 × 496
        ↓
Resize to 224 × 224
        ↓
Single Channel → 3 Channels
        ↓
Normalize Pixel Values
        ↓
EfficientNetB0
```

No image augmentation such as flipping, rotation, zoom, contrast adjustment, or brightness adjustment was used in the reported experiment.

---

## Model Architecture

The model is based on **EfficientNetB0 pretrained on ImageNet**.

The original classification head is removed and replaced with a lightweight binary classification head.

### Architecture

```text
Input Image
224 × 224 × 3
       ↓
EfficientNetB0
(ImageNet Pretrained)
       ↓
Global Average Pooling
       ↓
Dropout (0.3)
       ↓
Dense Layer
Sigmoid Activation
       ↓
Binary Prediction
Normal / Disease
```

### Trainable Parameters

The EfficientNetB0 backbone is kept **frozen** during training.

| Parameter Type           |     Count |
| ------------------------ | --------: |
| Total parameters         | 4,050,852 |
| Trainable parameters     |     1,281 |
| Non-trainable parameters | 4,049,571 |

Only the final classification head is trained.

This makes the model relatively parameter-efficient while still benefiting from features learned through ImageNet pretraining.

---

## Training Configuration

The model was trained using **TensorFlow/Keras** with the following configuration:

| Configuration            | Value                |
| ------------------------ | -------------------- |
| Backbone                 | EfficientNetB0       |
| Pretraining              | ImageNet             |
| Backbone                 | Frozen               |
| Classification type      | Binary               |
| Optimizer                | Adam                 |
| Learning rate            | 0.001                |
| Loss function            | Binary Cross-Entropy |
| Batch size               | 32                   |
| Maximum epochs           | 10                   |
| Dropout                  | 0.3                  |
| Classification threshold | 0.5                  |

### Training Callbacks

The training process uses:

* **ModelCheckpoint**
* **EarlyStopping**
* **ReduceLROnPlateau**

The learning rate is reduced by a factor of **0.2** after two epochs without improvement in validation loss.

---

## Evaluation

The final model is evaluated using retinal images belonging to **14 patients that were not used during training or validation**.

The following metrics are reported:

* Accuracy
* ROC-AUC
* Binary Cross-Entropy loss
* Precision
* Recall / Sensitivity
* Confusion matrix
* Per-class precision, recall, and F1-score

A probability threshold of **0.5** is used for converting the model's sigmoid output into the final binary prediction.

---

## Results

The model achieved the following performance on the test set:

| Metric                    |     Result |
| ------------------------- | ---------: |
| Accuracy                  | **91.66%** |
| ROC-AUC                   | **97.11%** |
| Precision                 | **91.08%** |
| Recall / Sensitivity      | **95.44%** |
| Binary Cross-Entropy Loss | **0.2425** |

### Confusion Matrix

The reported confusion matrix is based on **13,707 test images**:

|                    | Predicted Normal | Predicted Disease |
| ------------------ | ---------------: | ----------------: |
| **Actual Normal**  |            4,720 |               768 |
| **Actual Disease** |              375 |             7,844 |

From the reported results:

* True Negatives (TN): **4,720**
* False Positives (FP): **768**
* False Negatives (FN): **375**
* True Positives (TP): **7,844**

The relatively high recall of **95.44%** indicates that the model identifies a large proportion of disease-positive retinal images in the evaluated test set.

---

## Project Structure

The current repository contains the following core files:

```text
retinal-screening/
│
├── data_pipeline.py
├── train_model.py
├── evaluate_model.py
└── README.md
```

### `data_pipeline.py`

Handles the data preparation pipeline required for model development, including preprocessing and dataset preparation.

### `train_model.py`

Contains the model training workflow using EfficientNetB0 and the configured training parameters.

### `evaluate_model.py`

Evaluates the trained model on the test data and calculates the reported performance metrics, including the confusion matrix and classification metrics.

---

## How to Run

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd retinal-screening
```

Run the data pipeline:

```bash
python data_pipeline.py
```

Train the model:

```bash
python train_model.py
```

Evaluate the trained model:

```bash
python evaluate_model.py
```

> The exact dataset paths, arguments, and execution requirements depend on the implementation contained in the Python files. The OLIVES dataset must be obtained separately and is not included in this repository.

---

## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **EfficientNetB0**
* **ImageNet pretrained weights**
* **Deep Learning**
* **Computer Vision**
* **Binary Classification**

---

## Research Significance

A major focus of this project is **patient-wise validation**.

In medical imaging, randomly splitting individual images can result in images from the same patient appearing in both training and testing sets. This can make model performance appear better than its ability to generalize to genuinely unseen patients.

By splitting the data at the patient level, this project evaluates the model using patients that were not present during training.

The resulting evaluation therefore provides a more meaningful baseline for studying retinal disease screening performance.

---

## Limitations

This project is intended as a **research baseline** and should not be interpreted as a clinically validated diagnostic system.

The study has several limitations:

1. The original disease categories are reduced to a binary Normal/Disease classification.
2. The EfficientNetB0 backbone remains completely frozen.
3. No image augmentation was used in the reported experiment.
4. External dataset validation was not performed.
5. Prospective clinical testing was not performed.
6. Calibration analysis was not included.
7. Subgroup performance analysis was not included.
8. An independent baseline comparison was not performed.

These limitations provide opportunities for future improvement and further research.

---

## Future Work

Potential extensions of this work include:

* Partial fine-tuning of EfficientNetB0
* Clinically motivated image augmentation
* Grad-CAM-based explainability
* Multi-class or multi-label retinal disease classification
* Improved handling of class imbalance
* Probability calibration
* Subgroup performance analysis
* External dataset validation
* Prospective clinical evaluation

---

## Reproducibility Note

The reported experiment uses a patient-wise split of:

```text
60 patients → Training
13 patients → Validation
14 patients → Testing
```

The final reported evaluation is based on **13,707 test images**.

Maintaining the patient-wise separation is important when reproducing the reported evaluation because an image-level random split could introduce patient-level data leakage.

---

## Associated Research Paper

**Patient-Wise Validated EfficientNetB0 for Binary Retinal Disease Screening on the OLIVES Dataset**

Authors:

* P. Preetha
* S. Anusri
* R. Barani Dharan
* K. Keerthanadharshini

**KPR Institute of Engineering and Technology, Coimbatore, India**

---

## Disclaimer

This project is developed for academic and research purposes**.

The model is not intended to replace professional medical examination, diagnosis, or clinical decision-making. The reported results are based on the specific dataset and experimental setup described in the associated research work.

---

## Acknowledgement

This work uses the **OLIVES retinal imaging dataset** for research and experimental evaluation.

The repository is intended to provide the implementation associated with the reported research methodology and results.
