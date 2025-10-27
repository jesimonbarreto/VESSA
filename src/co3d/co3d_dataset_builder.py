"""CO3D dataset."""

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


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for CO3D dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  n_total_pairs = 0


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(Co3D): Specifies the tfds.core.DatasetInfo object
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
            'image1': tfds.features.Image(encoding_format='jpeg'),
            'image2': tfds.features.Image(encoding_format='jpeg'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image1','image2'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(Co3D): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    path = '/mnt/disks/stg_dataset/dataset/CO3D/'
    train_path = os.path.join(path, 'train')  

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": train_path,
            },
        )
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
  
  # Função para selecionar pares com distância x entre as posições
  def select_pairs_with_distance(self, sorted_paths, x, n):
      max_start_index = len(sorted_paths) - x - 1
      if max_start_index < 0:
          #raise ValueError("Distância x é muito grande para a lista fornecida.")
         return []
      pairs = []
      for _ in range(n):
          start_index = random.randint(0, max_start_index)
          end_index = start_index + x
          pairs.append((sorted_paths[start_index], sorted_paths[end_index]))
      
      return pairs
  
  def load_npz(self, npz_path):
    """Loads an .npz file and returns the array with shape (1, 384)."""
    return jnp.load(npz_path, allow_pickle=True)#['arr_0']

  def calculate_cosine_distance_dot(self, vector1, vector2):
      """Calculates the cosine distance using the dot product."""
      # Dot product between vectors
      dot_prod = jnp.dot(vector1.flatten(), vector2.flatten())
      # Norms of the vectors
      norm_v1 = jnp.linalg.norm(vector1)
      norm_v2 = jnp.linalg.norm(vector2)
      
      # Cosine distance
      distance = 1 - (dot_prod / (norm_v1 * norm_v2))
      return distance

  def check_npz_exists(self, frames_video):
      """Checks if all .npz files corresponding to the frames exist."""
      for frame in frames_video:
          npz_path = frame.replace('.png', '.npz')
          if not os.path.exists(npz_path):
              return False
      return True

  def find_most_distant_pairs(self, frames_video, n):
    """
    Compares the cosine distances between the embeddings of the frames
    and returns the n pairs with the greatest distances, along with the maximum and minimum distances.
    """
    # Check if all .npz files exist
    if not self.check_npz_exists(frames_video):
        return None

    # Load the embedding vectors (1, 384) for each frame
    vectors = []
    for frame in frames_video:
        npz_path = frame.replace('.jpg', '.npy')
        vectors.append(self.load_npz(npz_path))

    n_frames = len(vectors)
    distances = []
    
    # Calculate the cosine distance between each pair of frames
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            dist = self.calculate_cosine_distance_dot(vectors[i], vectors[j])
            distances.append((dist, frames_video[i], frames_video[j]))  # Save the distance and the paths of the corresponding frames
    
    # Sort distances in descending order (most distant pairs first)
    distances.sort(reverse=True, key=lambda x: x[0])
    
    # Form the 'n' most distant pairs, without repeating frames
    pairs = []
    used = set()
    
    for dist, frame1, frame2 in distances:
        if frame1 not in used and frame2 not in used:
            pairs.append((frame1, frame2))
            used.add(frame1)
            used.add(frame2)
        
        # Stop when we have 'n' pairs
        if len(pairs) == n:
            break
    
    # Extract the distances for the selected pairs
    selected_distances = [dist for dist, frame1, frame2 in distances if (frame1, frame2) in pairs or (frame2, frame1) in pairs]
    
    # Calculate the maximum and minimum distance from the selected pairs
    max_distance = max(selected_distances) if selected_distances else None
    min_distance = min(selected_distances) if selected_distances else None

    return pairs, max_distance, min_distance
  

  def read_file_load(self, ARQUIVO_CONTROLE = "control_time.npz"):
    # Inicializa o arquivo caso não exista
    if not os.path.exists(ARQUIVO_CONTROLE):
        parametros = np.array([2, 3, 4])
        controle = np.array([0, 0, 0])
        np.savez(ARQUIVO_CONTROLE, parametros=parametros, controle=controle)

    # Carrega o arquivo existente
    dados = np.load(ARQUIVO_CONTROLE)
    parametros = dados['parametros']
    controle = dados['controle']

    # Encontra o primeiro índice com valor 0
    indices_disponiveis = np.where(controle == 0)[0]
    if len(indices_disponiveis) == 0:
        raise ValueError("Todos os parâmetros já foram utilizados.")

    idx = indices_disponiveis[0]
    parametro = parametros[idx]

    # Atualiza o controle e salva novamente
    controle[idx] = 1
    np.savez(ARQUIVO_CONTROLE, parametros=parametros, controle=controle)

    return int(parametro)

  def _generate_examples(self, datapath):
    """Yields examples."""
    
    ARQUIVO_CONTROLE = "/mnt/disks/stg_dataset/dataset/CO3D/control_time.npz"

    datapath, file_path = os.path.split(datapath)
    if not datapath.endswith('/'):
        datapath += '/'
    
    if file_path == 'train':
        file_path = '/mnt/disks/stg_dataset/dataset/CO3D/train.npz'
        dist =  5 # 3,5,7,9,10
        n = 3
    else:
        file_path = '/mnt/disks/stg_dataset/dataset/CO3D/test.npz'
        dist =  5 # 3,5,7,9,10
        n = 3

    #/mnt/disks/stg_dataset/dataset/CO3D/tv/video/images/frame000001.jpg
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
        frames_video = sorted(frames_video, key=self.get_sequence_number)

        dist = random.randint(5, 11)

        # Seleciona os pares
        pairs = self.select_pairs_with_distance(frames_video, dist, n)
        
        if len(pairs) == 0:
           continue
        
        for k ,image_path in enumerate(pairs):
          img1 = self.process_image(image_path[0])
          img1 = img1.astype(jnp.uint8)
          img2 = self.process_image(image_path[1])
          img2 = img2.astype(jnp.uint8)
          record = {
            #"video": video_,
            "image1": img1,
            "image2": img2,
          }
          self.n_total_pairs+=1
          #print('number total samples '+str(self.n_total_pairs))
          yield str(k)+'_'+id, record