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
from jax.nn import softmax

import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

import vit_vessa as vit
import utils_vessa as utils
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import datasets
from scenic.dataset_lib import dataset_utils

import vessa_dataset  # pylint: disable=unused-import
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

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]

def normalize(input, p=2.0, axis=1, eps=1e-12):
    norms = jnp.linalg.norm(input, ord=p, axis=axis, keepdims=True)
    return input / jnp.maximum(norms, eps)

def representation_fn_eval(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    project_feats = True,
    gather_to_host: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks. If `gather_to_host` is True, these are collected from all hosts.
  """
  #variables = {'params': train_state.params, **train_state.model_state}

  '''embedding = flax_model.apply(
    variables, 
    batch['sample'],
    train=False,
    return_feats = True,
    debug=False, 
    project_feats = project_feats,
  )'''
  embedding = flax_model.apply(
        {'params': train_state.params},
        batch['image_resized'],
        seqlen=-1,
        seqlen_selection='consecutive',
        drop_moment='late',
        backbone = True,
        train=False)
  embedding = jnp.squeeze(embedding['x_norm_clstoken'])
  embedding = normalize(embedding)

  if gather_to_host:
    embedding = jax.lax.all_gather(embedding, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  

  return embedding

def knn_evaluate(
  dataset: dataset_utils.Dataset,
  config: ml_collections.ConfigDict,
  train_state: utils.TrainState,
  model: nn.Module
):

  result = eval(
      config=config,
      dataset=dataset,
      train_state=train_state,
      model=model
  )
  return result
  
def eval(
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    train_state: utils.TrainState,
    model: nn.Module
):

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  files_save = config.get('dir_files')
  num_classes = config.get('num_classes')
    
  #project feats or not
  representation_fn_knn = functools.partial(
      representation_fn_eval,
      flax_model = model.flax_model, 
  )
  repr_fn = jax.pmap(
          representation_fn_knn, 
          #donate_argnums=(1,),
          axis_name='batch',
    )

    # extract features
  @jax.jit
  def extract_features(batch):
      features = repr_fn(train_state, batch)
      return features  # Return extracted features for the batch
    
  dir_save_ckp = os.path.join(files_save,f'ckp_')
  dir_save_y = os.path.join(files_save,f'y_')

  if not os.path.exists(dir_save_ckp):
      os.makedirs(dir_save_ckp)
    
  if not os.path.exists(dir_save_y):
      os.makedirs(dir_save_y)
  if config.get('extract_train'):
      print('Starting to extract features train')
      print_result = True
      for i in range(config.steps_per_epoch):
        path_file = os.path.join(dir_save_ckp,f'ckp_b{i}')
        batch_train = next(dataset.train_iter)
        emb_train = extract_features(batch_train)
        norm_res = round(jnp.linalg.norm(jnp.array([emb_train[0,0,0]]), ord=2))==1
        if print_result:
          print(f'shape emb_train {emb_train.shape}')
          print(f'processing batch {i} shape {emb_train.shape}. Norma 1 {norm_res}')
          print_result=False
        if not norm_res:
          emb_train = normalize(emb_train)
        label_train = batch_train['label']
        emb_train = emb_train[0]
        bl, bg, emb = emb_train.shape
        emb_train = emb_train.reshape((bl*bg, emb))
        label_train = label_train.reshape((bl*bg))
        jnp.savez(path_file, emb=emb_train, label=label_train)
      print('Finishing extract features train')
  else:
      print('Not extract train')

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

  def calculate_similarity(train_samples, test_samples):
      return jnp.dot(test_samples, train_samples.T)

  def compute_distance(U, V):
      return compute_diff(U, V).mean(axis=-1)

  def compute_dist(u, v):
      return jnp.linalg.norm(u[:, None] - v[None, :], axis=-1)
    
    # Função para calcular a acurácia de um batch
  def calculate_batch_correct_predictions(probas, labels):
      predictions = jnp.argmax(probas, axis=1)
      correct_predictions = jnp.sum(predictions == labels)
      return correct_predictions
    
  devices = jax.device_count()
  n_test = config.dataset_configs.batch_size_test
    
  ks = config.get('ks')
    
  def compute_k_closest(U, V, k):
      D = compute_distance(U, V)
      D = D.reshape(devices, n_test // devices, -1)
      nearest = p_argsort(D)[..., 1:k+1]
      return nearest
    
  def one_hot(x, num_classes):
      return jax.nn.one_hot(x, num_classes)
    
  len_test = 0
  T=config.get('T')
  total_correct_predictions = {k: 0 for k in ks}
  total_samples = 0
  max_k = jnp.array(ks).max()
  print_result=True
  for i in range(config.steps_per_epoch_eval):
      #print(f'processing step eval {i}')
      batch_eval = next(dataset.valid_iter)
      emb_test = extract_features(batch_eval)[0]
      bl, bg, emb = emb_test.shape
      emb_test = emb_test.reshape((bl*bg, emb))
      label_eval = batch_eval['label'].reshape((bl*bg))
      norm_res = round(jnp.linalg.norm(jnp.array([emb_test[0]]), ord=2))==1
      #print(f'processing batch test {i} shape {emb_test.shape}. Norma 1 {norm_res}')
      if not norm_res:
        emb_test = normalize(emb_test)
      if print_result:
          print(f'embedding shape test {emb_test.shape}')
          print_result=False
      
      sim_all = []
      labels = []
      len_test += len(batch_eval)
      for j in range(config.steps_per_epoch):
        emb_file_save = os.path.join(dir_save_ckp,f'ckp_b{j}')
        data_load = jnp.load(emb_file_save+'.npz')
        emb_train = data_load['emb']#extract_features(batch_train)
        label_train = data_load['label']#batch_train['label'][0]

        sim = calculate_similarity(emb_train, emb_test)
        sim_all.append(sim)
        labels.append(label_train)
      
      sim_all = jnp.concatenate(sim_all, axis=1)
      labels = jnp.concatenate(labels)

      # Usamos argsort para obter os índices que ordenariam a matriz
      sorted_indices = jnp.argsort(sim_all, axis=-1)[:, ::-1]  # Ordena em ordem decrescente
      topk_indices = sorted_indices[:, :max_k]

      # Selecionamos os maiores valores de similaridade usando os índices ordenados
      topk_sims = jnp.take_along_axis(sim_all, topk_indices, axis=-1)
      labels = labels[topk_indices]#jnp.take_along_axis(labels, topk_indices, axis=-1)

      batch_size = labels.shape[0]
      topk_sims_transform = softmax(topk_sims / T, axis=1)
      
      matmul = one_hot(labels, num_classes=num_classes) * topk_sims_transform[:, :, None]
      
      probas_for_k = {k: jnp.sum(matmul[:, :k, :], axis=1) for k in ks}
      print_result = True

      for k in ks:
        correct_predictions = calculate_batch_correct_predictions(probas_for_k[k], label_eval)
        total_correct_predictions[k] += correct_predictions
        if print_result:
          print(f'Using k = {k} -- batch {batch_size}/{correct_predictions} certos')
          print_result = False
      total_samples += batch_size
      

    # Calcular a acurácia total para cada K
  total_accuracies = {k: total_correct_predictions[k] / total_samples for k in ks}

    # Resultado
  result = {}
  print("Total Accuracy:")
  for k, accuracy in total_accuracies.items():
        print(f"K:{k} Accuracy: {accuracy:.4f}")
        result['K_'+str(k)]=round(accuracy*100,2)
  return result