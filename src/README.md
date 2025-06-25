# VESSA: Video-based Efficient Self-Supervised Adaptation

![VESSA Pipeline](../images/vessa_pipeline.png)

**VESSA** is a lightweight and effective approach for adapting vision foundation models to new domains using short object-centric videos.  
It leverages self-supervised distillation to extract robust representations without requiring any manual labels.

This repository contains all components required to reproduce the experiments, including model code, environment configuration, and dataset preparation scripts.

The implementation is based on the **[Scenic](https://github.com/google-research/scenic)** library and uses **[JAX](https://github.com/google/jax)** as the underlying framework for efficient and scalable training.

# 🧠 VESSA Training and Evaluation – Source Code

Welcome to the core implementation of **VESSA**: Video-based Efficient Self-Supervised Adaptation.

> ❗ Before running any training or evaluation, make sure you have:
> 1. Set up your environment as described in [`./Environment/`](../Environment/)
> 2. Prepared your dataset according to the steps in [`./Datasets/`](../Datasets/)

This folder (`./src`) contains all source code modules, including training and evaluation pipelines, model configurations, and utility scripts.

---

## 🚀 Running Training

Training is performed using the `main_vessa` script, based on a configuration file that defines data paths, hyperparameters, and model architecture.

```bash
python -m main_vessa --config=config_adapt.py --workdir=dir/save/files/train
```

If you encounter permission issues or are working in a restricted environment, you may prepend with:

```bash
sudo -E python -m main_vessa --config=config_adapt.py --workdir=dir/save/files/train
```

- `--config`: Python file that defines training parameters (e.g., `config_adapt.py`)
- `--workdir`: Directory where logs, checkpoints, and metrics will be saved

---

## 📊 Running Evaluation (k-NN)

To evaluate the learned representations using a k-Nearest Neighbors classifier:

```bash
python -m knn_main --config=config_eval.py --workdir=dir/save/files/test
```

With sudo (if needed):

```bash
sudo -E python -m knn_main --config=config_eval.py --workdir=dir/save/files/test
```

---

## 📂 Directory Contents

- `main_vessa.py` — self-supervised training loop using DINO-style distillation  
- `knn_main.py` — k-NN based evaluation on extracted embeddings  
- `config_adapt.py` — configuration for training (VESSA adaptation)  
- `config_eval.py` — configuration for k-NN evaluation  
- `models/` — vision transformer backbone and lightweight adaptation layers  
- `utils/` — data loading, checkpointing, metrics, and logging utilities  

---

## 📄 Logs and Pretrained Checkpoints

We provide logs and pretrained weights for **ViT-Base** models trained with **VESSA** on two datasets.

### 📦 CO3D (ViT-Base)

- 📝 Training Log: [Download CO3D Log](<insert_log_link_here>)
- 🧠 Pretrained Weights: [Download CO3D ViT-Base Checkpoint](<insert_checkpoint_link_here>)

### 📦 MVImageNet (ViT-Base)

- 📝 Training Log: [Download MVImageNet Log](<insert_log_link_here>)
- 🧠 Pretrained Weights: [Download MVImageNet ViT-Base Checkpoint](<insert_checkpoint_link_here>)

> Replace `<insert_log_link_here>` and `<insert_checkpoint_link_here>` with your actual URLs when available.

---

## 🔁 Reproducing Evaluation with Pretrained Weights

If you want to **evaluate the pretrained models without training**:

1. Download the checkpoint for the desired dataset (CO3D or MVImageNet).
2. Place the downloaded file inside your desired `--workdir` directory, e.g.:

```
/your/project/path/
└── dir/save/files/test/
    └── checkpoint.pkl  # or the provided name
```

3. Modify `config_eval.py` to include the path to this checkpoint:

```python
config.init_from = '/your/project/path/dir/save/files/test/checkpoint.pkl'
```

4. Run the evaluation:

```bash
python -m knn_main --config=config_eval.py --workdir=dir/save/files/test
```

This will extract embeddings from the pretrained model and perform a k-NN classification to reproduce the results reported in the paper.

---

## 🗂️ Output Directory Structure

The `--workdir` directory will contain:

```
workdir/
├── config.py              # Final resolved configuration
├── checkpoints/           # Model weights (e.g., .pkl or .npz)
├── logs/                  # Logs from W&B or custom logger
├── metrics/               # Training/validation metrics
└── eval_embeddings.npy    # Features used in k-NN evaluation
```

---

## 🧪 Reproducibility Checklist

- ✅ Environment installed via `prepare_env.sh`  
- ✅ Dataset organized as instructed in [`./Dataset/`](../Dataset/)  
- ✅ Config files adjusted with correct paths  
- ✅ Checkpoints downloaded and available in `--workdir`  

---

## 📬 Contact

For questions, feel free to open an issue or contact [Jesimon Barreto](https://github.com/jesimonbarreto).
