git clone https://github.com/google-research/scenic.git
cd scenic/
#scenic version base used
git checkout 1e7282a
pip install -e .
cd ../
rm -rf scenic
git clone https://github.com/jesimonbarreto/scenic.git
cd scenic
pip install Pillow
pip install tensorflow-addons==0.21.0
pip uninstall tensorflow
pip uninstall keras
pip install tensorflow==2.13.1
pip install keras==2.13.1
pip install imageio
pip install matplotlib
pip install wandb
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install keras_applications==1.0.8
pip install scikit-learn
pip install numpy==1.26.4
pip install jax==0.5.3 jaxlib==0.5.3