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
from PIL import Image

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)  # Fixes seed for Python's random module

# JAX seed, typically used for JAX random operations (if needed)
jax_key = jax.random.PRNGKey(seed_value)


mvimgnet_classes = [
    "bag", "bottle", "washer", "vessel", "train", "telephone", "table", "stove", "sofa", "skateboard", 
    "rifle", "pistol", "remote control", "printer", "flowerpot", "pillow", "piano", "mug", "motorcycle", 
    "microwave", "microphone", "mailbox", "loudspeaker", "laptop", "lamp", "knife", "pot", "helmet", 
    "guitar", "bookshelf", "faucet", "earphone", "display", "dishwasher", "computer keyboard", "clock", 
    "chair", "car", "cap", "can", "camera", "cabinet", "bus", "bowl", "bicycle", "bench", "bed", "bathtub", 
    "basket", "ashcan", "airplane", "umbrella", "plush toy", "toy figure", "towel", "toothbrush", "toy bear", 
    "toy cat", "toy bird", "toy insect", "toy cow", "toy dog", "toy monkey", "toy elephant", "toy fish", 
    "toy horse", "toy sheep", "toy mouse", "toy tiger", "toy rabbit", "toy dragon", "toy snake", "toy chook", 
    "toy pig", "rice cooker", "pressure cooker", "toaster", "dryer", "battery", "curtain", "blackboard eraser", 
    "bucket", "calculator", "candle", "cassette", "cup sleeve", "computer mouse", "easel", "fan", "cookie", 
    "fries", "donut", "coat rack", "guitar stand", "can opener", "flashlight", "hammer", "scissors", "screw driver", 
    "spanner", "hanger", "jug", "fork", "chopsticks", "spoon", "ladder", "ceiling lamp", "wall lamp", "lamp post", 
    "light switch", "mirror", "paper box", "wheelchair", "walking stick", "picture frame", "shower", "toilet", 
    "sink", "power socket", "bagged snacks", "tripod", "selfie stick", "hair dryer", "lipstick", "glasses", 
    "sanitary napkin", "toilet paper", "rockery", "Chinese hot dishes", "root carving", "flower", "book", 
    "pipe PVC metal pipe", "projector", "cabinet air conditioner", "desk air conditioner", "refrigerator", 
    "percussion", "strings", "wind instruments", "balloons", "scarf", "shoe", "skirt", "pants", "clothing", 
    "box", "soccer", "roast duck", "pizza", "ginger", "cauliflower", "broccoli", "cabbage", "eggplant", 
    "pumpkin", "winter melon", "tomato", "corn", "sunflower", "potato", "sweet potato", "Chinese cabbage", 
    "onion", "momordica charantia", "chili", "cucumber", "grapefruit", "jackfruit", "star fruit", "avocado", 
    "shakyamuni", "coconut", "pineapple", "kiwi", "pomegranate", "pawpaw", "watermelon", "apple", "banana", 
    "pear", "cantaloupe", "durian", "persimmon", "grape", "peach", "power strip", "racket", "toy butterfly", 
    "toy duck", "toy turtle", "bath sponge", "glove", "badminton", "lantern", "chestnut", "accessory", "shovel", 
    "cigarette", "stapler", "lighter", "bread", "key", "toothpaste", "swim ring", "watch", "telescope", "eggs", 
    "bun", "guava", "okra", "tangerine", "lotus root", "taro", "lemon", "garlic", "mango", "sausage", "besom", 
    "lock", "ashtray", "conch", "seafood", "hairbrush", "ice cream", "razor", "adhesive hook", "hand warmer", 
    "thermometer", "bell", "sugarcane", "adapter(water pipe)", "calendar", "insecticide", "electric saw", 
    "inflator", "ironmongery", "bulb"
]

filter_imagnet = [2, 7, 10, 12, 13, 15, 19, 20, 21, 22, 23, 26, 33, 34, 47,
                  49, 51, 76, 81, 83, 84, 94, 96, 113, 120, 123, 133, 136,
                  149, 151, 152, 158, 166, 168, 173, 175, 179, 187, 197,
                  200, 214, 221, 224]


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
            'image1': tfds.features.Image(encoding_format='jpeg'),
            'image2': tfds.features.Image(encoding_format='jpeg'),
            #'label': tfds.features.ClassLabel(names=list(mvimgnet_classes)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image1','image2', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(MVImgNet): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    path = os.path.join('/mnt/disks/dataset/mvimgnet/data/')
    

    # TODO(MVImgNet): Returns the Dict[split names, Iterator[Key, Example]]
  
    '''dirname = self.builder_config.dirname
    url = _URL_PREFIX + "{}.tgz".format(dirname)
    path = dl_manager.download_and_extract(url)
    train_path = os.path.join(path, dirname, "train")
    val_path = os.path.join(path, dirname, "val")'''

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": path,
            },
        )
    ]

  '''def _generate_examples(self, datapath):
    """Yields examples."""
    for label in tf.io.gfile.listdir(datapath):
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        for fpath in tf.io.gfile.glob(os.path.join(datapath, label, obj_var, "*.jpg")):
          fname = os.path.basename(fpath)
          record = {
              "image": fpath,
              "label": mvimgnet_classes[label],
              "label_number": label,
              "obj_var": obj_var
          }
          yield fname, record
  '''
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
          npz_path = frame.replace('.jpg', '.npy')
          if not os.path.exists(npz_path):
              return False
      return True

  
  def limited_true_function(self, max_true=10):
    true_count = 0

    def deterministic_true():
        nonlocal true_count
        if true_count < max_true:
            true_count += 1
            return True
        return False

    return deterministic_true
  
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
  
  import random





  def find_most_distant_pairs_randomized(self, frames_video, n):
    """
    Selects frames randomly and finds the farthest frame from each selected frame,
    forming n pairs while ensuring no frame is reused in multiple pairs.
    """
    # Check if all .npz files exist
    if not self.check_npz_exists(frames_video):
        return None

    # Load the embedding vectors (1, 384) for each frame
    vectors = []
    for frame in frames_video:
        npz_path = frame.replace('.jpg', '.npy')
        vectors.append(self.load_npz(npz_path))

    # List to track used frames
    used_frames = set()
    pairs = []

    while len(pairs) < n and len(used_frames) < len(frames_video):
        # Randomly select a frame that hasn't been used yet
        available_frames = [i for i in range(len(frames_video)) if i not in used_frames]
        if not available_frames:
            break

        random_idx = random.choice(available_frames)
        used_frames.add(random_idx)

        # Find the farthest frame from the selected frame
        farthest_idx = None
        max_distance = -float('inf')
        
        for i in range(len(vectors)):
            if i != random_idx and i not in used_frames:
                dist = self.calculate_cosine_distance_dot(vectors[random_idx], vectors[i])
                if dist > max_distance:
                    max_distance = dist
                    farthest_idx = i
        
        if farthest_idx is not None:
            pairs.append((frames_video[random_idx], frames_video[farthest_idx]))
            used_frames.add(farthest_idx)

    # Extract distances for the selected pairs
    #distances = [
    #    self.calculate_cosine_distance_dot(self.load_npz(pair[0].replace('.jpg', '.npy')), self.load_npz(pair[1].replace('.jpg', '.npy')))
    #    for pair in pairs
    #]

    # Calculate the maximum and minimum distance from the selected pairs
    max_distance = 0 #max(distances) if distances else None
    min_distance = 0 #min(distances) if distances else None

    return pairs, max_distance, min_distance


  def _generate_examples(self, datapath):
    """Yields examples."""
    detmn_true = self.limited_true_function(max_true=20)
    dir_saveimage = "/home/jesimonbarreto/plots/"
    cont = 1

    for label in tf.io.gfile.listdir(datapath):
      if int(label) not in filter_imagnet:
         continue
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        dir_search = os.path.join(datapath, label, obj_var,'images', "*.jpg")
        frames_video = tf.io.gfile.glob(dir_search)
        #base_names = [os.path.basename(fpath) for fpath in frames_video]
        id = label+'_'+obj_var
        dist = 5
        n = 3

        # Ordena a lista de paths usando o número da sequência como chave
        frames_video = sorted(frames_video, key=self.get_sequence_number)

        # Seleciona os pares
        #pairs = self.select_pairs_with_distance(frames_video, dist, n)
        pairs = self.find_most_distant_pairs_randomized(frames_video, n)
        #metrics mse, 
        if pairs is None:
           continue
        
        pairs, max_distance, min_distance = pairs
        if len(pairs) == 0:
           continue
        
        #print(f'Max {max_distance} Min {min_distance}')
        
        for k ,image_path in enumerate(pairs):
          plot_image = detmn_true()
          img1 = self.process_image(image_path[0])
          img1 = img1.astype(jnp.uint8)
          img2 = self.process_image(image_path[1])
          img2 = img2.astype(jnp.uint8)
          if plot_image:
            Image.fromarray(img1).save(f"{dir_saveimage}{cont}_1.png")
            Image.fromarray(img2).save(f"{dir_saveimage}{cont}_2.png")
            cont+=1

          record = {
            #"video": video_,
            "image1": img1,
            "image2": img2,
            #"label": int(label)
          }
          self.n_total_pairs+=1
          #print('number total samples '+str(self.n_total_pairs))
          yield str(k)+'_'+id, record