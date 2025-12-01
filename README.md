# 🌄 Intel Landscape Image Classification – Fine-Tuning Pretrained Image Models

This repository contains a Jupyter Notebook that fine-tunes deep learning models on the **Intel Image Classification dataset**, a collection of **14,000+ RGB images** across **6 outdoor landscape categories**:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The notebook demonstrates a full end-to-end deep learning workflow using **TensorFlow/Keras**, including data preprocessing, augmentation, model building, fine-tuning pretrained models, and evaluation.

---

## 🚀 Features

### ✔️ **End-to-End Training Pipeline**

* Loads Intel Landscape dataset
* Performs directory traversal, dataset verification
* Splits training, validation, and test sets

### ✔️ **Data Preprocessing**

* Resizing to 150×150
* Normalization
* One-hot label encoding

### ✔️ **Data Augmentation**

Using `ImageDataGenerator`:

* Rescaling
* Rotation
* Shear
* Zoom
* Horizontal flip
* Validation split

### ✔️ **Model Training**

Fine-tunes state-of-the-art CNN architectures:

* **InceptionV3**
* **EfficientNetB0/B3** (based on your notebook)

Includes:

* Callback support (`EarlyStopping`, `ReduceLROnPlateau` ,`ModelCheckpoint`)
* Transfer learning + unfrozen layers for fine-tuning
* Performance monitoring with accuracy/loss plots

### ✔️ **Evaluation**

* Generates accuracy, loss curves
* Computes validation accuracy
* Performs prediction on sample test images

---

## 📁 Dataset

The Intel Image Classification dataset is publicly available on Kaggle:

🔗 **[https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)**

The notebook expects the following directory structure:

```
├── seg_train
├── seg_test
└── seg_pred
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* ImageDataGenerator

---

## 🧠 How the Models Work

The notebook uses **Transfer Learning**:

1. Loads pretrained CNN on ImageNet
2. Freezes base layers
3. Adds custom classification head
4. Trains on Intel dataset
5. Optionally unfreezes deeper layers to fine-tune

This approach improves accuracy, reduces training time, and requires fewer training samples.

---

## 📊 Evaluation

Typical performance ranges:

* **93%+ validation accuracy** (InceptionV3 / EfficientNet)
* High generalization with strong augmentation
* Stable training curves using callbacks

---

## ▶️ Running the Notebook

1. Open Jupyter Notebook / Kaggle Notebook
2. Upload the dataset
3. Install dependencies:

   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```
4. Run all cells in order
5. View training results and evaluation metrics
