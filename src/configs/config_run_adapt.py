# pylint: disable=line-too-long
"""Default config for Vessa training on ImageNet2012 for 100 epochs."""

import ml_collections

VARIANT = 'B/14'
_IMAGENET_TRAIN_SIZE = 20412 #(number of video filtered) * n pairs of each video
_IMAGENET_TEST_SIZE = 4535 #4535
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def get_config():
  global _IMAGENET_TRAIN_SIZE, _IMAGENET_TEST_SIZE

  """Returns the default config for a 100 epoch DINO training on ImageNet2012."""
  config = ml_collections.ConfigDict()
  #WANDB
  config.project = 'Exp_explora'
  config.experiment_name = 'continue_learning'
  #config
  config.transfer_learning = True
  config.train_layers = ["encoder", "ToTokenSequence"]
  '''config.train_layers = ["ToTokenSequence_0", "encoder_norm",
                        "key", "MlpBlock_0", "out" 
                         ]'''

  config.train_two_last_vit= True
  config.int_train_last_layers = 2

  #Uncert loss
  config.un_loss = True

  

  config.train_layer_comp = ['encoderblock_11'] #None
  config.lnorm_0 = "adam"
  config.lnorm_1 = "adam"
  config.mlpblock_dense_0 = "adam"
  config.mlpblock_dense_1 = "adam"
  config.multi_key = "adam"
  config.multi_out = "adam"
  config.multi_query = "adam"
  config.multi_value = "adam"
  config.train_layers_str = [True, True] #, True, True, True, True]
  config.use_checkpoint = False #use checkpoint basewith other training 
  config.use_ckpt_dir = '/mnt/disks/stg_dataset/head_2/'
  config.layer_wise = False
  config.print_lr_infos = False
  #LORA
  config.lora_use = True
  config.lora_rank = 64
  # Dataset.
  config.dataset_name = 'vessa_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  reference_resolution = 448
  n_queries = 10
  config.mode = 'video' #'video_crops' # video or random
  
  #plot
  config.plot_ex = False
  config.number_plot = 2
  config.dir_plot = '/home/jesimonbarreto/images/'

  # Training.'MVImagenet'
  #Loss: L2, Lwf, cosine, v*vessa X (1-v)other
  config.alpha_loss = 1
  config.beta_loss = 1
  config.gama_loss = 1
  config.teta_loss= 0.7
  config.max_grad_norm = 1
  config.num_training_epochs = 10#400
  config.batch_size = 128
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch

  #DINO
  config.global_crops_scale = (0.14, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.25)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 10 #change other parameters
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.07
  config.warmup_teacher_temp_epochs = 0
  
  config.dataset_configs.number_of_focal_queries = n_queries - 1

  if config.mode == 'video' and config.ncrops == 0:
    config.dataset_configs.pp_train = (
        f'copy("image1", "x1")'+
        f'|copy("image2", "x2")'+
        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x1", "x1"), outkey=("x1", "image1"))' +
        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x2", "x2"), outkey=("x2", "image2"))' +
        '|value_range(0, 1, data_key="x1")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
        '|random_grayscale(0.2, data_key="x1")' +
        '|random_blur(1.0, data_key="x1")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'

        '|value_range(0, 1, data_key="x2")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
        '|random_grayscale(0.2, data_key="x2")' +
        '|random_blur(0.1, data_key="x2")' +
        '|random_solarize(0.2, data_key="x2")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")'+
        '|keep("x1", "x2")'
    )
  elif config.mode == 'video' and config.ncrops > 0:
    config.dataset_configs.pp_train = (
        f'copy("image1", "x1")'+
        f'|copy("image2", "x2")'+
        ''.join([f'|copy("image{1 if i % 2 == 0 else 2}", "crop{i}")' for i in range(config.ncrops)]) +

        f'|copy_resize_file(448, {config.global_crops_scale}, inkey=("x1", "x1"), outkey=("x1", "image1"))' +
        f'|copy_resize_file(448, {config.global_crops_scale}, inkey=("x2", "x2"), outkey=("x2", "image2"))' +
        '|value_range(0, 1, data_key="x1")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
        '|random_grayscale(0.2, data_key="x1")' +
        '|random_blur(1.0, data_key="x1")' +
        '|value_range(0, 1, data_key="x1")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'

        '|value_range(0, 1, data_key="x2")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
        '|random_grayscale(0.2, data_key="x2")' +
        '|random_blur(0.1, data_key="x2")' +
        '|random_solarize(0.2, data_key="x2")' +
        '|value_range(0, 1, data_key="x2")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")'+

        ''.join([f'|copy_resize_file(96, {config.local_crops_scale}, inkey=("crop{i}", "crop{i}"), outkey=("crop{i}", "image1"))' for i in range(config.ncrops)]) +
        ''.join([f'|value_range(0, 1, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_grayscale(0.2, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_blur(0.5, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|value_range(0, 1, data_key="crop{i}")' for i in range(config.ncrops)]) +
        '|keep("x1", "x2"' + ''.join([f', "crop{i}"' for i in range(config.ncrops)]) + ')'
      )
  elif config.mode == 'frame' and config.ncrops > 0:
    config.dataset_configs.pp_train = (
        f'copy("image1", "x1")'+
        f'|copy("image1", "x2")'+
        ''.join([f'|copy("image1", "crop{i}")' for i in range(config.ncrops)]) +

        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x1", "x1"), outkey=("x1", "image1"))' +
        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x2", "x2"), outkey=("x2", "image1"))' +
        '|value_range(0, 1, data_key="x1")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
        '|random_grayscale(0.2, data_key="x1")' +
        '|random_blur(1.0, data_key="x1")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'

        '|value_range(0, 1, data_key="x2")' +
        #if use function simulate cam movements
        #'|cam_motion(max_translate=0.1, max_rotate=10.0, max_scale=0.05,brightness_delta=0.1, contrast_range=(0.9, 1.1), data_key="x2")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
        '|random_grayscale(0.2, data_key="x2")' +
        '|random_blur(0.1, data_key="x2")' +
        '|random_solarize(0.2, data_key="x2")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")'+

        ''.join([f'|copy_resize_file(96, {config.local_crops_scale}, inkey=("crop{i}", "crop{i}"), outkey=("crop{i}", "image1"))' for i in range(config.ncrops)]) +
        ''.join([f'|value_range(0, 1, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_grayscale(0.2, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|random_blur(0.5, data_key="crop{i}")' for i in range(config.ncrops)]) +
        ''.join([f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="crop{i}")' for i in range(config.ncrops)]) +
        '|keep("x1", "x2"' + ''.join([f', "crop{i}"' for i in range(config.ncrops)]) + ')'
      )
  
  else:
    config.dataset_configs.pp_train = (
        f'copy("image1", "x1")'+
        f'|copy("image1", "x2")'+
        f'|copy("image2", "x3")'+
        f'|copy("image2", "x4")'+
        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x1", "x1"), outkey=("x1", "image1"))' +
        f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x2", "x2"), outkey=("x2", "image2"))' +
        '|value_range(0, 1, data_key="x1")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
        '|random_grayscale(0.2, data_key="x1")' +
        '|random_blur(1.0, data_key="x1")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'

        '|value_range(0, 1, data_key="x2")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
        '|random_grayscale(0.2, data_key="x2")' +
        '|random_blur(0.1, data_key="x2")' +
        '|random_solarize(0.2, data_key="x2")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")'+

        '|value_range(0, 1, data_key="x3")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x3")' +
        '|random_grayscale(0.2, data_key="x3")' +
        '|random_blur(0.1, data_key="x3")' +
        '|random_solarize(0.2, data_key="x3")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x3")'+

        '|value_range(0, 1, data_key="x4")' +
        '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x4")' +
        '|random_grayscale(0.2, data_key="x4")' +
        '|random_blur(0.1, data_key="x4")' +
        '|random_solarize(0.2, data_key="x4")' +
        f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x4")'+
        
        '|keep("x1", "x2", "x3", "x4")'
    )
  
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.dataset = 'co3d' # dataset example co3d or mvimgnet
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.dataset_dir = '/mnt/disks/stg_dataset/dataset/imagenet/'


  # Model.
  version, patch = VARIANT.split('/')
  patch = int(patch)
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [patch, patch]
  config.model.posembs = (518 // patch, 518 // patch)
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {'Ti': 768,
                          'S': 1536,
                          'B': 3072,
                          'L': 4096,
                          'H': 5120}[version]
  config.model.num_layers = {'Ti': 12,
                             'S': 12,
                             'B': 12,
                             'L': 24,
                             'H': 32}[version]
  config.model.head_output_dim = 65536 
  config.model.attention_dropout_rate = 0.0
  
  #head
  config.model.n_layers = 2
  config.model.head_hidden_dim = 2048
  config.model.head_bottleneck_dim = 256
  
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05
  
  #Verificar esses fatores no codigo
  config.norm_last_layer = True
  config.momentum_teacher = 0.996
  config.use_bn_in_head = False
  config.load_weight = True
  config.load_weights = 'tips_b14' #'vessa_vitb16' #'vessav2_vit'+version.lower()+'14'#'vessa_vitb16'#'vessa_vitdeits16'#'vessav2_vit'+version.lower()+'14'


  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = -1 #int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  # Learning rate.
  config.lr=0.001
  #cosine schedule lr
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = config.steps_per_epoch * 15
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = config.lr * config.batch_size / 1024.
  config.lr_configs.alpha = 0.01

  # Weight decay.
  config.weight_decay = 0.04
  
  #Verificar
  config.weight_decay_end = 0.4
  config.warmup_epochs=10
  config.optimizer = 'adamw'
  config.drop_path_rate= 0.1

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1. / 0.996

  # Logging.
  config.write_summary = True
  config.save_state_0 = False
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.log_summary_steps = 5
  config.max_keep_checkpoint = 2


  return config


