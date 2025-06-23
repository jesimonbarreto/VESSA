"""MVImgNet dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
#import tensorflow_datasets.public_api as tfds

import os
import numpy as np
import jax
import jax.numpy as jnp
import re
import random
# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)  # Fixes seed for Python's random module

# JAX seed, typically used for JAX random operations (if needed)
jax_key = jax.random.PRNGKey(seed_value)

classes_co3d = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase"
]



# Função para obter a posição do número na lista ordenada
def get_position(name_class):
    """
    Retorna a posição de um número na lista ordenada.
    """
    if name_class in classes_co3d:
        return classes_co3d.index(name_class)
    else:
        return -1  # Retorna -1 se o número não estiver na lista

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mvimgnet dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  n_total_pairs = 0


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(MVImgNet): Specifies the tfds.core.DatasetInfo object
    video_shape = (
        None,
        224,
        224,
        3,
    )
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            #'video': tfds.features.Video(
            #  video_shape,
            #  encoding_format= 'jpeg'),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names=list(classes_co3d)),
            #'index': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label', 'index'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    path = '/mnt/disks/stg_dataset/dataset/CO3D/'
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')  

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": train_path,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "datapath": test_path,
            },
        ),
    ]

  def process_image(self, image_path):
      # Leia o arquivo da imagem
      image = tf.io.read_file(image_path)
      # Decodifique a imagem para um tensor
      image = tf.image.decode_jpeg(image, channels=3)
      # Redimensione a imagem
      image = tf.image.resize(image, [224, 224])
      # Normalize a imagem
      #image = tf.cast(image, tf.float32) / 255.0
      # Converta o tensor para um numpy array
      return image.numpy()

  # Função para extrair o número da sequência do nome do arquivo
  def get_sequence_number(self, path):
      # Usa regex para encontrar o número no nome do arquivo
      match = re.search(r'(\d+)', path)
      if match:
          return int(match.group(1))
      return None
  
  # Função para selecionar n valores aleatórios
  def select_random_values(self, sorted_paths, n):
    # Garantir que n não seja maior que o número de elementos disponíveis
    n = min(n, len(sorted_paths))
    
    # Selecionar n valores aleatórios sem substituição
    random_indices = random.sample(range(len(sorted_paths)), n)
    
    # Retornar os valores correspondentes aos índices selecionados
    random_values = [sorted_paths[i] for i in random_indices]
    
    return random_values


  def _generate_examples(self, datapath):
    """Yields examples."""
    
    datapath, file_path = os.path.split(datapath)
    if not datapath.endswith('/'):
        datapath += '/'
    
    if file_path == 'train':
        file_path = '/mnt/disks/stg_dataset/dataset/CO3D/train.npz'
    else:
        file_path = '/mnt/disks/stg_dataset/dataset/CO3D/test.npz'
    
    n = 1


    train_ref = np.load(file_path, allow_pickle=True)
    keys_ref = train_ref.keys()
    

    for label in tf.io.gfile.listdir(datapath):
      full_path = tf.io.gfile.join(datapath, label)
      if not tf.io.gfile.isdir(full_path):
         continue
      train_class_ref = train_ref[label]
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        if obj_var not in train_class_ref:
           continue
        dir_search = os.path.join(datapath, label, obj_var, 'images', "*.jpg")
        frames_video = tf.io.gfile.glob(dir_search)
        #base_names = [os.path.basename(fpath) for fpath in frames_video]
        id = label+'_'+obj_var

        # Ordena a lista de paths usando o número da sequência como chave
        #frames_video = sorted(frames_video, key=self.get_sequence_number)

        # Seleciona os pares
        samples = self.select_random_values(frames_video, n)
        
        if len(samples) == 0:
           continue
        
        for k, image_path in enumerate(samples):
          img = self.process_image(image_path)
          img = img.astype(jnp.uint8)
          record = {
            "image": img,
            "label": get_position(label),
            #"index": str(k)+'_'+id
          }
          yield str(k)+'_'+id, record
