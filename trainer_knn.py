"""LOCA Training Script."""

import copy
import functools
from typing import Any, Callable, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
from jax import nn as opr
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
import utils_dino as utils
import vit_dino as vit
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils
import math, sys

"""Script for Knn evalulation."""
import functools

from clu import metric_writers

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

import classification_with_knn_eval_trainer
import datasets
import knn_utils
import models
import vit_dino as vit




import jax.numpy as jnp
from scenic.core import Module, builders
from scenic.layers import stop_gradient
from sklearn.model_selection import train_test_split
import stax


class KNNModel(Module):
  def __init__(self, k):
    self.k = k

  @builders.call
  def __call__(self, data, query):
    # Stop gradient on data to prevent backpropagation through it
    data = stop_gradient(data)

    # Calculate pairwise distances
    distances = jnp.sum((data[:, None, :] - query[None, ...])**2, axis=-1)

    # Sort distances for each query point
    sorted_distances = jnp.sort(distances, axis=-1)

    # Select k nearest neighbors
    nearest_neighbors = jnp.take_along_axis(data, sorted_distances[:, :, :self.k], axis=-1)

    return nearest_neighbors

def train_knn(data, labels, k):
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

  # Define and build KNN model
  model = KNNModel(k)
  model = builders.build_model(model, data=X_train)

  # Define loss function and optimizer
  def loss_fn(model_outputs, target_labels):
    # Calculate distance between predicted and true labels
    distances = jnp.sum((model_outputs - target_labels)**2, axis=-1)
    return jnp.mean(distances)

  optimizer = stax.optimizers.adam(0.01)

  # Train the model
  for epoch in range(10):
    # Forward pass and calculate loss
    model_outputs = model(X_train)
    loss = loss_fn(model_outputs, y_train)

    # Backpropagation and update weights
    grads = stax.gradient(loss_fn)(model, model_outputs, y_train)
    updates, _ = optimizer(grads, model.state)
    model = stax.optimize(updates, model)

  return model

def test_knn(model, X_test, y_test):
  # Get predictions for test data
  predictions = model(X_test)

  # Calculate accuracy
  accuracy = jnp.mean(predictions == y_test)

  print(f"Test accuracy: {accuracy}")

# Example usage
data = jnp.random.rand(100, 10)
labels = jnp.random.randint(0, 3, size=100)

# Train the model with k=5
model = train_knn(data, labels, k=5)

# Test the model on unseen data
test_knn(model, X_test, y_test)

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]





def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the utils.TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    dataset: The dataset that has train_iter and meta_data.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training.
  """

  lead_host = jax.process_index() == 0

  # Build the evalulationloss_fn, metrics, and flax_model.
  model = vit.ViTDinoModel(config, dataset.meta_data)

  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)

  rng, init_rng = jax.random.split(rng)

  #Load model
  train_state = train_utils.TrainState(
    params=params, 
    model_state=model_state
  )
  chrono = train_utils.Chrono()
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params

  #Predict train samples with model

  _, x_train= model.apply(
        {'params': train_state.params},
        batch['reference'],
        train=True)


  #predict val samples with model
  

  #call train knn pass train samples and annotations

  #call test knn pass train samples and annotaions

  #Associa o modelo a representação de features
  representation_fn_knn = functools.partial(
    classification_with_knn_eval_trainer.representation_fn_eval,
    flax_model = model.flax_model, 
    project_feats = config.project_feats_knn
  )

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size
  
  knn_evaluator = knn_utils.KNNEvaluator(
    config,
    representation_fn_knn,
    knn_eval_batch_size,
    config.get("extract_only_descrs",False),
  )

  train_dir = config.get('train_dir')

  if config.test_pretrained_features:

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      0,
      writer,
      config.preextracted,
    )

  for epoch in range(config.knn_start_epoch,config.knn_end_epoch+1):

    step = epoch * config.steps_per_epoch

    print(f"step: {step}")

    if not config.preextracted:
      ckpt_file = os.path.join(train_dir,str(step))  
      ckpt_info = ckpt_file.split('/')
      ckpt_dir = '/'.join(ckpt_info[:-1])
      ckpt_num = ckpt_info[-1].split('_')[-1]

      try:

        train_state, _ = train_utils.restore_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )
        
      except:

        sys.exit("no checkpoint found")

      train_state = jax_utils.replicate(train_state)

    else:

      train_state = None

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      step,
      writer,
      config.preextracted,
    )

  train_utils.barrier_across_hosts()