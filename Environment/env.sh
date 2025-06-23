#!/bin/bash

# =========================================
# Environment Setup Script for This Project
# =========================================

# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # For Windows: use venv\Scripts\activate.bat

# 2. Clone and install Scenic (original version, temporarily)
git clone https://github.com/google-research/scenic.git
cd scenic/
pip install .

# 3. Go back and remove Scenic to replace with your customized fork
cd ../
rm -rf scenic
git clone https://github.com/jesimonbarreto/scenic.git
cd scenic/

# 4. Install required dependencies
pip install Pillow
pip install tensorflow-addons==0.21.0

# 5. Force compatible versions of TensorFlow and Keras
pip uninstall -y tensorflow
pip uninstall -y keras
pip install tensorflow==2.13.1
pip install keras==2.13.1

# 6. Install additional dependencies
pip install imageio
pip install matplotlib
pip install wandb
pip install keras_applications==1.0.8
pip install scikit-learn
pip install numpy==1.26.4

# 7. Install JAX with TPU support and version pinning
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jax==0.5.3 jaxlib==0.5.3