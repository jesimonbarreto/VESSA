"""Data generators for the universal embedding datasets."""

import collections
import functools
import os
from typing import Optional, Union, List

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
from tensorflow.io import gfile
from scenic.train_lib import train_utils
import ops  # pylint: disable=unused-import
import tensorflow_datasets as tfds
import data_utils
import tensorflow as tf

import json

import numpy as np
from collections import OrderedDict


PRNGKey = jnp.ndarray

IMAGE_RESIZE = 256
IMAGE_SIZE = 224



"""Data generators for a LOCA dataset."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import builder
import mvimagenetest_dataset
import CO3Dtest_dataset


@datasets.add_dataset('eval_dataset')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns a generator for training Vessa on a specified dataset.

  Args:
    batch_size: int; Determines the training batch size.
    eval_batch_size: int; Not used.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: Not used.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes train_iter and dict of meta_data.
  """
  del eval_batch_size, rng
  logging.info('Loading train split of the %s for eval training.',
               dataset_configs.dataset)
  n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                              dataset_configs.train_split,
                                              data_dir=dataset_configs.get('dataset_dir'))

  tfds.load(
    dataset_configs.dataset,
    data_dir= dataset_configs.get('dataset_dir'),
    download = True
  )

  filter_cls = None 
  if dataset_configs.get('filter_classes'):
    desired_classes = tf.constant(dataset_configs.get('desired_classes'))
    filter_cls=functools.partial(
            data_utils.filter_classes_ts, allowed_labels=desired_classes
    )
  
  train_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=dataset_configs.batch_size_train,
      filter_fn=None,
      preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_train, remove_tpu_dtypes=True, log_data=True),
      shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      drop_remainder=True,
      cache=False,
      ignore_errors=True)
  
  eval_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.test_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=dataset_configs.batch_size_test,
      filter_fn=filter_cls,
      preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_train, remove_tpu_dtypes=True, log_data=True),
      shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      drop_remainder=True,
      cache=False,
      ignore_errors=True)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    assert dataset_configs.shuffle_buffer_size is not None
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)
    eval_ds = dataset_utils.distribute(eval_ds, dataset_service_address)

  #n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
  #                                            dataset_configs.train_split)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  image_size = train_ds.element_spec['image_resized']
  labels_size = train_ds.element_spec['label_onehot']
  logging.info(f' train {train_ds.element_spec.keys()}')
  logging.info(f' test {eval_ds.element_spec.keys()}')
  logging.info(f' image {image_size}')
  logging.info(f' labels {labels_size}')

  
  input_shape = (-1,) + tuple(train_ds.element_spec['image_resized'].shape[1:])
  labels_size = train_ds.element_spec['label_onehot'].shape
  logging.info(' input shape details %s', input_shape)
  logging.info(' samples details %s', labels_size)
  

  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': n_train_ex,
      'input_dtype': getattr(jnp, dtype_str),
      'label_data' : labels_size,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
