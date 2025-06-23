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
from scenic.train_lib import pretrain_utils
import ops  # pylint: disable=unused-import
from jax.nn import softmax

import os
import sys
import re


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
from flax.core import freeze, unfreeze
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
import utils_dino as utils

from functools import partial
from jax import jit

import wandb


FLAGS = flags.FLAGS

def get_highest_checkpoint(directory):
    """
    Retorna o caminho completo do arquivo checkpoint com o maior número no formato `checkpoint_numero`.

    Args:
        directory (str): Caminho do diretório onde buscar os arquivos.

    Returns:
        str: Caminho completo do arquivo com o maior número encontrado no formato `checkpoint_numero`, ou None se não encontrar.
    """
    checkpoint_pattern = re.compile(r"^checkpoint_(\d+)$")
    highest_checkpoint = None
    highest_number = -1

    for file_name in os.listdir(directory):
        match = checkpoint_pattern.match(file_name)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number
                highest_checkpoint = file_name

    if highest_checkpoint:
        return [os.path.join(directory, highest_checkpoint)]

    return []

def get_all_checkpoint_numbers(directory):
    """
    Retorna uma lista dos números dos arquivos de checkpoint no formato `checkpoint_numero`.

    Args:
        directory (str): Caminho do diretório onde buscar os arquivos.

    Returns:
        List[int]: Lista de números extraídos dos arquivos encontrados, ou lista vazia se nenhum for encontrado.
    """
    checkpoint_pattern = re.compile(r"^checkpoint_(\d+)$")
    checkpoint_numbers = []

    for file_name in os.listdir(directory):
        match = checkpoint_pattern.match(file_name)
        if match:
            checkpoint_numbers.append(int(match.group(1)))

    return checkpoint_numbers

def generate_conditional_freeze_layers(rules, negate_flags, use_and=True):
    """
    Retorna uma função lambda que verifica várias condições de 'in' ou 'not in' em cada elemento da lista.

    Parâmetros:
        rules (list[str]): Lista de strings para verificar no nome da camada.
        negate_flags (list[bool]): Lista de booleans para indicar se deve usar 'not in' (True) ou 'in' (False) para cada regra.

    Retorna:
        function: Função lambda personalizada.
    """
    return lambda layer_name: (all if use_and else any)(
        (rule in layer_name if negate else rule not in layer_name)
        for rule, negate in zip(rules, negate_flags)
    )

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
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:


  # Start a run, tracking hyperparameters
  wandb.init(
      # set the wandb project where this run will be logged
      project=config.project,
      name=config.experiment_name,
      # track hyperparameters and run metadata with wandb.config
      config=config.to_dict()
  )
  
  lead_host = jax.process_index() == 0

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  
  train(
      rng=rng,
      config=config,
      dataset=dataset,
      workdir=workdir,
      writer=writer)
  
def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:

  
  # Build the loss_fn, metrics, and flax_model.
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

  # Only one model function but two sets of parameters.
  ema_params = copy.deepcopy(params)

  # Get learning rate and ema temperature schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate)

  # Create optimizer.
  weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
  
  tx = optax.inject_hyperparams(optax.adamw)(
      learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
      mask=weight_decay_mask,)
  
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  # Create chrono class to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  # Create the TrainState to track training state (i.e. params and optimizer).
  train_state = utils.TrainState(
      global_step=0, opt_state=opt_state, tx=tx, params=params,
      ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  train_dir = config.get('train_dir')
  print(f'{train_dir}')
  steps = config.get('steps_checkpoints')
  files_save = config.get('dir_files')
  num_classes = config.get('num_classes')
  if not config.preextracted:
    all_ckpnt = config.get('get_all_checkpoins', True)
    if all_ckpnt: 
      name_path_step = get_highest_checkpoint(train_dir)
      part_file = int(name_path_step[0].split('_')[-1])
      steps = [part_file]
    else:
      steps = get_all_checkpoint_numbers(train_dir)
      print('\n\n Steps used :')
      print(steps)
      print('\n\n')
  else:
    steps = [0]
  
  for step in steps:

    print(f"step: {step}")
    

    if not config.preextracted:
      ckpt_file = os.path.join(train_dir,'checkpoint_'+str(step))  
      ckpt_info = ckpt_file.split('/')
      ckpt_dir = '/'.join(ckpt_info[:-1])
      ckpt_num = ckpt_info[-1].split('_')[-1]
      print(f"file: {ckpt_file}")
      print(f"ckpt_num: {ckpt_num}")

      #try:

      '''train_state, _ = train_utils.restore_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )'''
      
      train_state = utils.restore_pretrained_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )
        
      #except:

      #  sys.exit("no checkpoint found")
      #  continue

      train_state = jax_utils.replicate(train_state)

    else:
      '''=============================================='''
      print('Here... trying load')
      from load_params import load_params
      print(f' {config.dir_weight} {config.weight_load}')
      params = load_params(config.weight_load,config.dir_weight, params,
                    params_key='teacher_weights',
                    force_random_init= None)

      print('Here... finished load')
      '''=============================================='''
      # Only one model function but two sets of parameters.
      ema_params = copy.deepcopy(params)
      # Create the TrainState to track training state (i.e. params and optimizer).
      train_state = utils.TrainState(
          global_step=0, opt_state=opt_state, tx=tx, params=params,
          ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})
      train_state = jax_utils.replicate(train_state)
    
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
    
    dir_save_ckp = os.path.join(files_save,f'ckp_{step}')
    dir_save_y = os.path.join(files_save,f'y_{step}')

    if not os.path.exists(dir_save_ckp):
      os.makedirs(dir_save_ckp)
    
    if not os.path.exists(dir_save_y):
      os.makedirs(dir_save_y)
    if config.get('extract_train'):
      print('Starting to extract features train')
      for i in range(config.steps_per_epoch):
        path_file = os.path.join(dir_save_ckp,f'ckp_{step}_b{i}')
        batch_train = next(dataset.train_iter)
        emb_train = extract_features(batch_train)
        print(f'shape emb_train {emb_train.shape}')
        norm_res = round(jnp.linalg.norm(jnp.array([emb_train[0,0,0]]), ord=2))==1
        print(f'processing batch {i} shape {emb_train.shape}. Norma 1 {norm_res}')
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
    for i in range(config.steps_per_epoch_eval):
      print(f'processing step eval {i}')
      batch_eval = next(dataset.valid_iter)
      emb_test = extract_features(batch_eval)[0]
      bl, bg, emb = emb_test.shape
      emb_test = emb_test.reshape((bl*bg, emb))
      label_eval = batch_eval['label'].reshape((bl*bg))
      norm_res = round(jnp.linalg.norm(jnp.array([emb_test[0]]), ord=2))==1
      print(f'processing batch test {i} shape {emb_test.shape}. Norma 1 {norm_res}')
      if not norm_res:
        emb_test = normalize(emb_test)
    
      print(f'embeeding shape test {emb_test.shape}')
      sim_all = []
      labels = []
      len_test += len(batch_eval)
      for j in range(config.steps_per_epoch):
        emb_file_save = os.path.join(dir_save_ckp, f'ckp_{step}_b{j}')
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

      for k in ks:
        correct_predictions = calculate_batch_correct_predictions(probas_for_k[k], label_eval)
        total_correct_predictions[k] += correct_predictions
        wandb.log({f'batch_size_{k}':batch_size, 
                     f'correct_predictions{k}':correct_predictions,
                     f'acc_rel{k}':correct_predictions/batch_size})
        print(f'Considerando k== {k} -- batch {batch_size}/{correct_predictions} certos')
      total_samples += batch_size
      

    # Calcular a acurácia total para cada K
    total_accuracies = {k: total_correct_predictions[k] / total_samples for k in ks}

    # Resultado
    print(f'total samples used {total_samples}')
    print("Acurácia total para diferentes valores de K:")
    for k, accuracy in total_accuracies.items():
        wandb.log({
          "step": step,
          "K": k,
          "Accuracy": round(accuracy,4)
        })
        print(f"K-{k} acurácia total: {accuracy:.4f}")

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)