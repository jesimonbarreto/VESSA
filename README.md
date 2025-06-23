# VESSA: Video-based Efficient Self-Supervised Adaptation

![VESSA Pipeline](./images/vessa_pipeline.png)

**VESSA** is a lightweight and effective approach for adapting vision foundation models to new domains using short object-centric videos.  
It leverages self-supervised distillation to extract robust representations without requiring any manual labels.

This repository contains all components required to reproduce the experiments, including model code, environment configuration, and dataset preparation scripts.

The implementation is based on the **[Scenic](https://github.com/google-research/scenic)** library and uses **[JAX](https://github.com/google/jax)** as the underlying framework for efficient and scalable training.

---

## 📁 Repository Structure

- `./Environment/` — contains scripts and instructions to prepare the environment  
- `./Dataset/` — contains tools and instructions to construct and organize the datasets  
- `./image/` — contains figures and diagrams used for publication and documentation  
- `./core/` — core implementation of the method (models, training, evaluation)

---

## ⚙️ Setup

To set up your environment, please follow the instructions in:

📂 [`./Environment/`](./Environment/)

---

## 📦 Dataset Preparation

To construct and preprocess the dataset used in this project, refer to:

📂 [`./Dataset/`](./Dataset/)

---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{barreto2025vessa,
  title   = {VESSA: Video-based Efficient Self-Supervised Adaptation},
  author  = {Jesimon Barreto, Carlos Caetano, Andre Araujo, William Schwartz},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```

---

## 📬 Contact

For questions, feel free to open an issue or contact [Jesimon Barreto](https://github.com/jesimonbarreto).
