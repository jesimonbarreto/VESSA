"""DINO Training Script."""

import copy
import functools
from typing import Any, Callable, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax
import flax.linen as nn
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state as ts
import jax
from jax import nn as opr
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
import utils_dino as utils
import vit_dino_classification as vit
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils
import math, sys, os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import module_knn


def calculate_means(dictionary):
  """Calculates the mean of values in each NumPy array within a dictionary.

  Args:
    dictionary: A dictionary where each value is a NumPy array.

  Returns:
    A new dictionary with the same keys, but the values are the means.
  """

  new_dict = {}
  for key, array in dictionary.items():
    mean_value = jnp.mean(array)
    new_dict[key] = mean_value
  return new_dict

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]

def plot_example(train_batch, number_plot=5, dir_plot='/home/jesimonbarreto/images/', number_crops=2):
  
  def normalize_vector(vector):
    """Normalizes a JAX NumPy vector to values between 0 and 1."""
    min_val = jnp.min(vector)
    max_val = jnp.max(vector)
    return (vector - min_val) / (max_val - min_val)
  
  for stepe in range(number_plot):
    img = train_batch['x1'][0,stepe]
    print(f'x1 before_norm max {jnp.max(img)} min {jnp.min(img)}')
    img = normalize_vector(img)
    img = jnp.where(img > 1.0, 1.0, img)
    print(f'x1 after_norm max {jnp.max(img)} min {jnp.min(img)}')
    plt.imsave(os.path.join(dir_plot,f'imagex1_{stepe}.jpg'), img)  # Using matplotlib
    img = train_batch['x2'][0,stepe]
    print(f'x2 before_norm max {jnp.max(img)} min {jnp.min(img)}')
    img = normalize_vector(img)
    img = jnp.where(img > 1.0, 1.0, img)
    print(f'x2 after_norm x2 max {jnp.max(img)} min {jnp.min(img)}')
    plt.imsave(os.path.join(dir_plot,f'imagex2_{stepe}.jpg'), img)
    for vcrop in range(number_crops):
      print(f'{vcrop} de {number_crops}')
      img = train_batch[f'crops{vcrop}'][0,stepe]
      img = normalize_vector(img)
      plt.imsave(os.path.join(dir_plot,f'crops{vcrop}_{stepe}.jpg'), img)

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

def dino_train_step(
    train_state: utils.TrainState,
    batch: Batch,
    center: jnp.ndarray,
    epoch: int,
    *,
    flax_model: nn.Module,
    momentum_parameter_scheduler: Callable[[int], float],
    loss_fn: Any,
    metrics_fn: Any,
    steps_per_epoch: float,
    config: ml_collections.ConfigDict,
) -> Tuple[utils.TrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    momentum_parameter_scheduler: Momentum parameter scheduler for EMA update.
    loss_fn: The cross-entropy loss function.
    metrics_fn: Reports relative position loss and accuracy.
    config: Configurations of the experiment.

  Returns:
    The updated state of training.
  """
  # Some preparations.
  new_rng, dropout_rng, droptok_rng = jax.random.split(train_state.rng, num=3)
  dropout_rng = train_utils.bind_rng_to_host_device(
      dropout_rng, axis_name='batch', bind_to='device')
  step = train_state.global_step
  momentum_parameter = momentum_parameter_scheduler(step)
  bs = batch['image1'].shape[0]  # Per-device batch size.
  labels = batch['label']
  batch = utils.prepare_input_class(batch, config)

  def training_loss_fn(params, center, epoch):
    use_ema = config.apply_cluster_loss
    drop_moment = 'late' if config.apply_cluster_loss else 'early'

    logits = flax_model.apply(
        {'params': params},
        batch['sample'],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        backbone = True,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})['x_class']
    
    loss = optax.softmax_cross_entropy(
      logits=logits, 
      labels=jax.nn.one_hot(labels, logits.shape[-1])
    ).mean()

    pred_labels = jnp.argmax(logits, axis=-1)
    acc = (pred_labels == labels).mean()
    

    return loss, (loss, acc)
  
  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (total_loss, (loss_dino, acc)), grad = compute_gradient_fn(
      train_state.params, center, epoch)
  metrics = acc
  metrics = (
      dict(total_loss=(total_loss, 1),
           acc=(acc, 1) ))

  # Update the network parameters.
  grad = jax.lax.pmean(grad, axis_name='batch')
  if config.get('max_grad_norm', None) is not None:
    grad = optimizers.clip_grads(grad, config.max_grad_norm)
  new_train_state = train_state
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    # update the teacher weights
    new_ema_params = jax.tree_map(
        lambda s, t: momentum_parameter * t + (1 - momentum_parameter) * s,
        new_params, train_state.ema_params)

    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=step + 1,
        opt_state=new_opt_state,
        params=new_params,
        ema_params=new_ema_params,
        rng=new_rng)
  return new_train_state, center, metrics


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    #dataset_val: dataset_utils.Dataset,
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
  #plot flag
  fstexe = True

  # Build the loss_fn, metrics, and flax_model.
  model = vit.ViTDinoModel(config, dataset.meta_data)

  # Start a run, tracking hyperparameters
  wandb.init(
      # set the wandb project where this run will be logged
      project=config.project,
      name=config.experiment_name,
      # track hyperparameters and run metadata with wandb.config
      config=config.to_dict()
  )


  num_local_devices = jax.local_device_count()

  #Center used to calculate the loss
  center = jnp.zeros((num_local_devices, config.model.head_output_dim))
  
  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)
  
  '''  # Função para listar todas as camadas
  def list_layers(params, parent_name=""):
      layer_names = []
      for layer_name, layer_params in params.items():
          full_name = f"{parent_name}/{layer_name}" if parent_name else layer_name
          if isinstance(layer_params, dict):
              # Recursivamente lista subcamadas
              layer_names.extend(list_layers(layer_params, full_name))
          else:
              # Adiciona o nome da camada atual
              layer_names.append(full_name)
      return layer_names

  # Lista os nomes de todas as camadas
  layer_names = list_layers(params)
  for name in layer_names:
      print(name)'''
  _load_weight = config.get('load_weight')
  if _load_weight:
    '''=============================================='''
    #print(f'Here... trying load {params.keys()}')
    from load_params import load_params

    params = load_params(config.load_weights,'/home/jesimonbarreto/', params,
                  params_key='teacher_weights',
                  force_random_init= None)


    #print(f'Here... finished load {params.keys()}')
    '''=============================================='''

  # Only one model function but two sets of parameters.
  ema_params = copy.deepcopy(params)

  # Get learning rate and ema temperature schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate)

  weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
  # Create optimizer.
  if config.transfer_learning:
    params = freeze(params)
    def modify_encoder_block(data, number_layers=11):
      for idx_layer in range(number_layers):
        target_key = f"encoderblock_{idx_layer}"
        if target_key in data and data[target_key] == "zero":
            data[target_key] = {
                "LoRA_0": {"lora_A": "adam", "lora_B": "adam"},
                "LoRA_1": {"lora_A": "adam", "lora_B": "adam"},
                "LayerNorm_0": {"bias": "adam", "scale": "adam"},
                "LayerNorm_1": {"bias": "adam", "scale": "adam"},
                "MlpBlock_0": {
                    "Dense_0": {"bias": "zero",
                                "kernel": "zero"},
                    "Dense_1": {"bias": "zero",
                                "kernel": "zero"},
                },
                "MultiHeadDotProductAttention_0": {
                    "key": {"bias": "zero",
                            "kernel": "zero"},
                    "out": {"bias": "zero",
                            "kernel": "zero"},
                    "query": {"bias": "zero",
                              "kernel": "zero"},
                    "value": {"bias": "zero",
                              "kernel": "zero"},
                },
            }
      
      return data
    
    def modify_encoder_block_total(data, target_keys):
      for target_key in target_keys:
        if target_key in data and data[target_key] == "zero":
            data[target_key] = {
                "LayerNorm_0": {"bias": config.lnorm_0, "scale": config.lnorm_0},
                "LayerNorm_1": {"bias": config.lnorm_1, "scale": config.lnorm_1},
                "MlpBlock_0": {
                    "Dense_0": {"bias": config.mlpblock_dense_0,
                                "kernel": config.mlpblock_dense_0},
                    "Dense_1": {"bias": config.mlpblock_dense_1,
                                "kernel": config.mlpblock_dense_1},
                },
                "MultiHeadDotProductAttention_0": {
                    "key": {"bias": config.multi_key,
                            "kernel": config.multi_key},
                    "out": {"bias": config.multi_out,
                            "kernel": config.multi_out},
                    "query": {"bias": config.multi_query,
                              "kernel": config.multi_query},
                    "value": {"bias": config.multi_value,
                              "kernel": config.multi_value},
                },
            }
      
      return data
    
    def create_mask(params, label_fn, target_keys=None):
      def _map(params, mask, label_fn):
          for k in params:
              if label_fn(k):
                  mask[k] = 'zero'
              else:
                  if isinstance(params[k], FrozenDict):
                      mask[k] = {}
                      _map(params[k], mask[k], label_fn)
                  else:
                      mask[k] = 'adam'
      mask = {}
      _map(params, mask, label_fn)
      if target_keys:
        train_vit_two = config.get('train_two_last_vit', False)
        number_layers_lora_train = 11
        if train_vit_two:
          number_layers_lora_train = 10
          target_keys = ['encoderblock_10','encoderblock_11']
        mask = modify_encoder_block(mask, number_layers=number_layers_lora_train)
        mask = modify_encoder_block_total(mask, target_keys=target_keys)
      return frozen_dict.freeze(mask)

    def zero_grads():
        # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
        def init_fn(_):
            return ()
        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)
    
    list_str_layers = config.get('train_layers') or ["encoder", "ToTokenSequence"]
    list_str_layers_ver = config.get('train_layers_str') or [True, True]
    last_layer_train = config.get('train_layer_comp') or None
    freeze_encoder_and_token = generate_conditional_freeze_layers(
      list_str_layers, list_str_layers_ver, use_and=False
    )
    mask_t = create_mask(params, freeze_encoder_and_token, last_layer_train)
    print(mask_t)
    tx = optax.multi_transform(
        {'adam': optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn, weight_decay=config.weight_decay,),
        'zero': zero_grads()},
         mask_t
        )
    
  elif config.layer_wise:
    params = freeze(params)
    
    def distribute_learning_rates(model_name, num_layers, initial_rate, mult=10, min_v = 0.01):
      """Distributes learning rates exponentially from an initial rate.
      Args:
        model_name: The name of the model (string).
        num_layers: The number of layers in the model (int).
        initial_rate: The initial learning rate (float).
        divisor: The factor by which to divide subsequent rates (int, default=10).
      Returns:
        A dictionary where keys are layer names and values are learning rates.
      """
      learning_rates = {}
      current_rate = initial_rate
      for i in range(1, num_layers + 1):
          learning_rates[f"{model_name}{i}"] = current_rate
          if current_rate < min_v:
            current_rate *= mult
      return learning_rates

    layers = 16
    initial_rate = 0.0000000000000001
    mult = 10
    min_v = 0.01
    dist_lrs = distribute_learning_rates("adam", layers, initial_rate, mult, min_v)

    def create_maskLW(params):
      def _map(params, mask, level=1, cont=1):
          for k in params:
            if isinstance(params[k], FrozenDict):
              mask[k] = {}
              _map(params[k], mask[k], level=level+1, cont=cont)
            else:
              mask[k] = f'adam{cont}'
            if level==1:
              cont+=1       
      mask = {}
      _map(params, mask, level=1, cont=1)
      return frozen_dict.freeze(mask)
    
    def create_lr_rules(config, dist_lrs):
      optimizer_dict = {}
      for name, lr in dist_lrs.items():
        config.lr_configs.base_learning_rate = lr * config.batch_size / 1024
        learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=learning_rate_fn, weight_decay=config.weight_decay
        )
        optimizer_dict[name] = optimizer
      return optimizer_dict

    tx = optax.multi_transform(
        create_lr_rules(config, dist_lrs),
        create_maskLW(params)
        )
    if config.print_lr_infos:
      print(create_maskLW(params))
      print(dist_lrs)

  else:
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
        mask=weight_decay_mask,)
    
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  #config precisa adicionar
  #use_checkpoint
  #use_ckpt_dir
  #
  # Create chrono class to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  # Create the TrainState to track training state (i.e. params and optimizer).
  train_state = utils.TrainState(
    global_step=0, opt_state=opt_state, tx=tx, params=params, model_state=model_state,
      ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})

  if config.use_checkpoint:
     restored_train_state = utils.restore_pretrained_checkpoint(
          config.use_ckpt_dir, train_state, assert_exist=True)
     restored_model_cfg = config.model
      # Load params from the init_model.
     train_state = model.init_from_train_state(  # pytype: disable=attribute-error
          train_state, restored_train_state, restored_model_cfg)
     del restored_train_state
     print('\n\nload here\n\n')
  
  if config.save_state_0:
    unrep_train_state = train_state
    metadata = unrep_train_state.metadata
    metadata['chrono'] = chrono.save()
    unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
    utils.save_checkpoint(workdir, unrep_train_state)
    
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params, ema_params
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  un_loss = config.get('un_loss', False)

  # The function that performs one step of loca training.
  dino_train_step_pmapped = jax.pmap(
      functools.partial(
          dino_train_step,
          flax_model=model.flax_model,
          #alterar loss function
          loss_fn=model.loss_function if not un_loss else model.loss_function_uncertainty,
          metrics_fn=model.get_metrics_fn,
          momentum_parameter_scheduler=momentum_parameter_scheduler,
          steps_per_epoch = steps_per_epoch,
          config=config),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0,1),
  )

  train_metrics, train_summary, ext_log = [], None, []
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  report_progress = periodic_actions.ReportProgress(num_train_steps=total_steps,
                                                    writer=writer)
  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))
  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  logging.info('Starting training loop at step %d.', start_step + 1)
  v={}
  
  result_val={}
  for k_ in config.val.ks:
    result_val['K_'+str(k_)]=0.0
  
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      epoch = jnp.ones((num_local_devices, 1))*step/steps_per_epoch
      epoch = epoch.astype(jnp.int32)
      train_batch = next(dataset.train_iter)

      if config.plot_ex and fstexe:
        print(f' config {config.ncrops}')
        #plot_example(train_batch, 
        #             number_plot=config.number_plot,
        #             dir_plot=config.dir_plot,
        #             number_crops=config.ncrops)
        fstexe = False

      train_state, center, tm = dino_train_step_pmapped(
                                  train_state,
                                  train_batch,
                                  center,
                                  epoch)
      
      if config.transfer_learning or config.layer_wise:
        #print(train_state.opt_state)
        for inner_state in train_state.opt_state.inner_states.values():
          #print(inner_state)
          v = inner_state.inner_state.hyperparams
          break
      else:
        v = train_state.opt_state.hyperparams
      
      v = calculate_means(v)
      ext_log.append(v)
      train_metrics.append(tm)
    for h in hooks:
      h(step)

    ###################### LOG TRAIN SUMMARY ########################
    if (step % config.get('log_summary_steps') == 1) or (step == total_steps):
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs= ext_log,
          writer=writer)
      wb = train_utils.stack_forest(ext_log)
      for key, val in wb.items():
        train_summary[key]=float(val.mean())
      wandb.log(train_summary, step=step)
      wandb.log(result_val)
      chrono.resume()
      train_metrics = []
      ext_log = []
    
    ##################### CHECKPOINTING ###################
    if ((step % config.get('checkpoint_steps') == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          utils.save_checkpoint(workdir, unrep_train_state, max_to_keep=config.max_keep_checkpoint)
          del unrep_train_state
      chrono.resume()  # Un-pause now.

  ##################### VALIDATION ###################
  '''print('Starting Validation...')
  result_val = module_knn.knn_evaluate(
    dataset=dataset_val,
    config=config.val,
    train_state=train_state,
    model=model
  )
  wandb.log(result_val)
  print('Finishing Validation')'''
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  
  # Return the train summary after last step.
  return train_state, train_summary
