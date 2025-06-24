# 📦 Dataset Preparation

This folder contains scripts to help you download, extract, and prepare datasets required for training and evaluation.

---

## ✅ Step-by-step Instructions

### 1. Download the Dataset

Choose one of the following scripts depending on the dataset:


- For **MVImageNet**, run:
  ```bash
  bash down_mvimgnet.sh
  ```


---

### 3. Prepare the Dataset Splits

#### For **CO3D**:

Run the following script to split the dataset into training and test sets:

```bash
python split_train_test_protocols.py
```

Make sure all extracted folders are correctly located before running.

#### For **MVImageNet**:

After extracting the zip files, go into the dataset folder:

```bash
cd MVImgNet_Release
```

You can now use the prepared data directly.

---

Let us know if any part fails or if you need help setting up the structure!
