# 📦 Dataset Preparation

This folder contains scripts to help you download, extract, and prepare datasets required for training and evaluation.

---

## ✅ Step-by-step Instructions

### 1. Download the Dataset

Choose one of the following scripts depending on the dataset:

- For **CO3D**, run:
  ```bash
  bash down_co3d.sh
  ```


---

### 2. Extract the Zip Files

After downloading, run the following script to extract all `.zip` files:

```bash
bash extract_co3d.sh
```

This will unzip all archives and delete the original `.zip` files afterward.

---

### 3. Prepare the Dataset Splits

#### For **CO3D**:

Run the following script to split the dataset into training and test sets:

```bash
python split_train_test.py
```

Make sure all extracted folders are correctly located before running.
