"""Training utilities."""
import os
from typing import Any, Dict, Tuple, Optional

import flax
from flax import jax_utils
from flax import struct
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from tensorflow.io import gfile
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from collections import abc
import os
import re
from typing import Any, Dict, Mapping, List, Optional, Union

from absl import logging
import flax
from flax.training import checkpoints
import numpy as np

from scenic.train_lib import train_utils
from tensorflow.io import gfile

#from PIL import ImageFilter, ImageOps



@flax.struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  model_state: Optional[Any] = None
  ema_params: Optional[Any] = None
  params: Optional[Any] = None
  old_params: Optional[Any] = None
  state: Optional[Any] = None
  ema_state: Optional[Any] = None
  global_step: Optional[int] = 0
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


def save_checkpoint(workdir: str,
                    train_state: TrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False,
                    keep_every_n_steps: int = 50000):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint
      at the current or a later step already exits (default: False).
    keep_every_n_steps: Keep every checkpoints every n steps.
  """
  if jax.process_index() == 0:
    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        keep=max_to_keep,
        keep_every_n_steps=keep_every_n_steps)


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[
                           TrainState, int]:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory to restore the checkpoint.
    train_state: An instance of TrainState that holds the state of
      training.
    assert_exist: Assert that there is at least one checkpoint exists in
      the given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  if train_state is None:
    raise ValueError('Please use `restore_pretrained_checkpoint` for loading'
                     'a checkpoint without providing a Scenic TrainState.')
  train_state = checkpoints.restore_checkpoint(checkpoint_path, train_state,
                                               step)
  return train_state, int(train_state.global_step)


def to_cpu(array: jnp.ndarray):
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    array: Replicated array of shape
      [num_hosts, num_devices, local_batch_size, ...].

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(array)))


def prepare_input(inputs: Dict[str, jnp.ndarray],
                  config: ml_collections.ConfigDict) -> Dict[str, jnp.ndarray]:
  """Prepare the different views for LOCA training."""
  
  n_crops = config.ncrops
  mode = config.mode
  
  sample_x = jnp.concatenate([inputs['x1'], inputs['x2']])
  # views.
  batch = dict()
  if n_crops > 0:
    crops = jnp.concatenate(
        [inputs['crop' + str(i)] for i in range(n_crops)])

    batch['sample'] = [sample_x, crops]
  else:
    batch['sample'] = [sample_x]

  if mode == 'random':
    sample_x_add = jnp.concatenate([inputs['x3'], inputs['x4']])
    batch['sample'] = [sample_x, sample_x_add]

  return batch

def prepare_input_class(inputs: Dict[str, jnp.ndarray],
                  config: ml_collections.ConfigDict) -> Dict[str, jnp.ndarray]:
  """Prepare the different views for LOCA training."""
  
  n_crops = config.ncrops
  mode = config.mode
  
  sample_x = jnp.concatenate([inputs['image1'], inputs['image1']])
  # views.
  batch = dict() 
  batch['sample'] = [sample_x]
  return batch

def prepare_input_frame(inputs: Dict[str, jnp.ndarray],
                  config: ml_collections.ConfigDict,
                  epoch: int) -> Dict[str, jnp.ndarray]:
  """Prepare the different views for LOCA training."""
  
  n_crops = config.ncrops
  if epoch % 2 == 0:
    sample_x = jnp.concatenate([inputs['x1'], inputs['x2']])
  else:
    sample_x = jnp.concatenate([inputs['x3'], inputs['x4']])
  # views.
  batch = dict()
  if n_crops > 0:
    crps = jnp.concatenate(
        [inputs['crops' + str(i)] for i in range(n_crops)])

    batch['sample'] = [sample_x, crps]
  else:
    batch['sample'] = [sample_x]


  return batch


def sinkhorn(x, num_itr=3, distributed=True):
  """Sinkhorn-Knopp algorithm."""
  for _ in range(num_itr):
    # Total weight per prototype per device.
    weight_per_proto = jnp.sum(x, axis=0, keepdims=True)
    if distributed:
      # Globally.
      weight_per_proto = jax.lax.psum(weight_per_proto, axis_name='batch')
    x /= weight_per_proto

    # Total weight per sample.
    weight_per_sample = jnp.sum(x, axis=-1, keepdims=True)
    # x sums to 1 for each sample (it is an assignment).
    x /= weight_per_sample
  return x


def print_tree(d, depth=0, print_value=False):
    for k in d.keys():
        if isinstance(d[k], FrozenDict):
            print('  ' * depth, k)
            print_tree(d[k], depth + 1, print_value)
        else:
            if print_value:
                print('  ' * depth, k, d[k])
            else:
                print('  ' * depth, k)


def compare_params(lhs, rhs, depth):
    for k in lhs.keys():
        if isinstance(lhs[k], FrozenDict):
            print('  ' * depth, k)
            compare_params(lhs[k], rhs[k], depth + 1)
        else:
            print('  ' * depth, k, jnp.mean(jnp.abs(lhs[k] - rhs[k])))

from typing import Any, Dict, Mapping, List, Optional, Union

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]

def restore_pretrained_checkpoint(
    checkpoint_path: str,
    train_state: Optional[TrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None) -> TrainState:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training. This function also take care converting pre-Linen
  checkpoints.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    assert_exist: Assert that there is at least one checkpoint exists in the
      given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    Training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  if 'params' in restored_train_state:
    # restored_train_state was trained using optax
    restored_params = flax.core.freeze(restored_train_state['params'])
  else:
    # restored_train_state was trained using flax.optim. Note that this does
    # not convert the naming of pre-Linen checkpoints.
    restored_params = restored_train_state['optimizer']['target']
    if 'params' in restored_params:  # Backward compatibility.
      restored_params = restored_params['params']
      restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    restored_params = flax.core.freeze(restored_params)

  print(restored_train_state.keys())  # Lista todas as chaves disponíveis
  restored_model_state = flax.core.freeze(restored_train_state['model_state'])

  if not train_state:
    return restored_train_state, restored_params
    train_state = TrainState()
    params = restored_params
  else:
    # Inspect and compare the parameters of the model with the init-model.
    params = inspect_params(
        expected_params=train_state.params,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  train_state = train_state.replace(
      # Inspect and compare the parameters of the model with the init-model.
      params=params,
      ema_params=params,
      model_state=restored_model_state,
      global_step=int(restored_train_state['global_step']),
      rng=restored_train_state['rng'],
      metadata=restored_train_state.get('metadata', None))
  return train_state


def inspect_params(*,
                   expected_params: PyTree,
                   restored_params: PyTree,
                   fail_if_extra: bool = True,
                   fail_if_missing: bool = True,
                   fail_if_shapes_mismatch: bool = False) -> PyTree:
  """Inspects whether the params are consistent with the expected keys."""

  def _flatten_params(d, parent_key='', sep='/'):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
      path = parent_key + sep + k if parent_key else k
      if isinstance(v, abc.MutableMapping):
        items.extend(_flatten_params(v, path, sep=sep).items())
      else:
        items.append((path, v))
    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
      items.append((parent_key, {}))
    return dict(items)

  expected_flat = _flatten_params(flax.core.unfreeze(expected_params))
  restored_flat = _flatten_params(flax.core.unfreeze(restored_params))
  missing_keys = expected_flat.keys() - restored_flat.keys()
  extra_keys = restored_flat.keys() - expected_flat.keys()

  is_shape_mismatch = False
  for key in restored_flat:
    if key in expected_flat:
      restored_shape = None
      expected_shape = None
      # Handle empty nodes (without trainable params)
      if not isinstance(restored_flat[key], dict):
        restored_shape = restored_flat[key].shape
      if not isinstance(expected_flat[key], dict):
        expected_shape = expected_flat[key].shape

      if restored_shape != expected_shape:
        is_shape_mismatch = True
        print('Key: %s. Expected shape: %s. Restored shape: %s', key,
                        expected_flat[key].shape, restored_flat[key].shape)

  # Adds back empty dict explicitly, to support layers without weights.
  # Context: FLAX ignores empty dict during serialization.
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      restored_params[k] = {}  # pytype: disable=unsupported-operands
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    print('Inspect recovered empty keys:\n%s', empty_keys)

  print('Inspect missing keys:\n%s', missing_keys)
  print('Inspect extra keys:\n%s', extra_keys)

  if fail_if_shapes_mismatch and is_shape_mismatch:
    raise ValueError('Shape mismatch between restored and target model')

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(
        f'Missing params from checkpoint: {missing_keys}.\n'
        f'Extra params in checkpoint: {extra_keys}.\n'
        f'Restored params from checkpoint: {restored_flat.keys()}.\n'
        f'Expected params from code: {expected_flat.keys()}.')
  return restored_params