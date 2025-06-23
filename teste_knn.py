"""Script for Knn evalulation."""
import functools

from clu import metric_writers
from absl import flags
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils
import ops  # pylint: disable=unused-import

import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

import vit_dino as vit
import utils_dino as utils
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import datasets
from scenic.dataset_lib import dataset_utils

import dino_dataset  # pylint: disable=unused-import
#from scenic.dataset_lib import tinyImagenet_dataset
import datasets_eval
import optax
from scenic.train_lib import lr_schedules
import copy

import functools
from typing import Any, Callable, Dict, Tuple, Optional, Type
import flax
from flax import jax_utils
from flax import linen as nn
from jax import vmap
from jax.lax import map as map_

from functools import partial
from jax import jit

FLAGS = flags.FLAGS


def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  lead_host = jax.process_index() == 0

  data_rng, rng = jax.random.split(rng)
  #dataset = train_utils.get_dataset(
  #    config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  
  train(
      rng=rng,
      config=config,
      #dataset=dataset,
      workdir=workdir,
      writer=writer)
  
def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    #dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:

  
  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  train_dir = config.get('train_dir')
  print(f'{train_dir}')
  steps = config.get('steps_checkpoints')
  files_save = config.get('dir_files')
  num_classes = config.get('num_classes')

  for step in steps:

    print(f"step: {step}")
    

    @jax.vmap
    def euclidean_distance(x1, x2):
      return jnp.linalg.norm(x1 - x2, axis=-1)

    @jax.vmap
    def cosine_similarity(x1, x2):
      return jnp.dot(x1, x2) / (jnp.linalg.norm(x1, axis=-1) * jnp.linalg.norm(x2, axis=-1))
    
    def compute_diff(u, v):
      return (u[:, None] - v[None, :]) ** 2

    compute_diff = jax.vmap(compute_diff, in_axes=1, out_axes=-1)

    p_argsort = jax.pmap(jnp.argsort, in_axes=0)


    def compute_distance(U, V):
      return compute_diff(U, V).mean(axis=-1)
    
    def compute_dist(u, v):
      return jnp.linalg.norm(u[:, None] - v[None, :], axis=-1)
    
    devices = jax.device_count()
    n_test = config.dataset_configs.batch_size_test
    
    ks = config.get('ks')
    
    def compute_k_closest(U, V, k):
      D = compute_distance(U, V)
      D = D.reshape(devices, n_test // devices, -1)
      nearest = p_argsort(D)[..., 1:k+1]
      return nearest
    
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import numpy as np

    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=150, 
                              centers=np.array(centers),
                              random_state=1)
    
    from sklearn.model_selection import train_test_split
    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)

    train_data, test_data, train_labels, test_labels = res

    train_data = train_data.reshape(4,30,2)
    train_labels = train_labels.reshape(4,-1)
    test_data = test_data.reshape(2,15,2)
    test_labels = test_labels.reshape(2,-1)

    for k in ks:
      print(f'K: {k}')
      len_test = 0
      correct_pred = 0
      predicts_acc = []
      for i in range(len(test_data)):
        print(f'processing step eval {i}')
        emb_test = test_data[i]
        label_eval = test_labels[i]
        #print(f'embeeding shape test {emb_test.shape}')
        dist_all = []
        labels = []
        len_test += len(emb_test)
        for j in range(len(train_data)):
          #batch_train = next(dataset.train_iter)
          emb_train = train_data[j] #data_load['emb']#extract_features(batch_train)
          label_train =  train_labels[j] #data_load['label']#batch_train['label'][0]
          dist_ = compute_dist(emb_test, emb_train)
          if j == 0:
            print(f' shape: emb train {emb_train.shape} emb test {emb_test.shape}')
            print(f' shape: dist {dist_.shape} emb test {emb_test.shape}')
          
          dist_all.append(dist_)
          labels.append(label_train)
        dist_all = jnp.concatenate(dist_all, axis=1)
        labels = jnp.concatenate(labels)
        if i ==0:
          print(f' shape: dist_all {dist_all.shape} labels {labels.shape}')
        
        k_nearest = jnp.argsort(dist_all)[:, :k]
        print(f' shape kneares {k_nearest.shape}')
        #k_nearest = k_nearest.reshape(-1, k)
        k_nearest_labels = labels.squeeze()[k_nearest]  # Shape: (n, 5)
        print(f' shape kneares labels{k_nearest_labels.shape}')
        print(f' shape kneares labels{k_nearest_labels}')
        print(f' labels {label_eval}')

        #most_repetitive_labels = [(num_classes-1) - jnp.bincount(row, minlength=num_classes)[::-1].argmax() for row in k_nearest_labels]
        most_repetitive_labels = jnp.array([jnp.bincount(row).argmax() for row in k_nearest_labels])
        y_pred = most_repetitive_labels

        comparison = jnp.array(y_pred).squeeze() == label_eval.squeeze()
        corrects = comparison.sum()  # Proportion of correct matches
        print(f"Step {step} -----> Corrects: {corrects:.1f} / total {len(comparison)}")
        predicts_acc.append(corrects)
        #class_rate = (labels[k_nearest, ...].mean(axis=1).round() == batch_eval['label'][0]).mean()
        #print(f"{class_rate=}")
      
      predicts_acc = jnp.asarray(predicts_acc)
      result = jnp.sum(predicts_acc)/len_test
      
      print(f"{k} Neighborhood: Accuracy total : {result:.4f} ---- executions {predicts_acc.shape} ----- step {step}")
  
    # Calculate accuracy as the ratio of correct predictions to total test samples
    #accuracy = correct_pred / len_test

    #print(f"Number of correct predictions: {correct_pred}")
    #print(f"Number of total predictions: {len_test}")
    #print(f"Accuracy: {accuracy:.4f}")
    

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)