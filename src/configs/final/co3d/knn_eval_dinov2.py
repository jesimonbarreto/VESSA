import ml_collections, os
import jax.numpy as jnp
VARIANT = 'B/14'
_IMAGENET_TRAIN_SIZE = 20412 #237402 #20412 #177559 #9469 #1281167
_IMAGENET_TEST_SIZE = 4535 #4535 #55823
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]
MEAN = [0.5]
STD = [0.5]

def get_config():
  global _IMAGENET_TRAIN_SIZE, _IMAGENET_TEST_SIZE
  """Returns the ViT experiment configuration."""
  config = ml_collections.ConfigDict()
  #WANDB
  config.project = 'final_exp'
  config.experiment_name = 'eval_co3b'
  config.extract_train = True
  config.transfer_learning = False

  config.train_two_last_vit= True
  config.int_train_last_layers = 2

  #LORA
  config.lora_use = True
  config.lora_rank = 64
  
  # Dataset.
  config.dataset_name = 'eval_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  #for cifar 10
  config.dataset_configs.dataset = 'co3dtest' #'mvimgnetest' #'co3dtest' #'imagenet2012'
  config.dataset_configs.dataset_dir = '/mnt/disks/stg_dataset/dataset/imagenet/'
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.test_split = 'validation'
  config.dataset_configs.batch_size_train = 256
  config.dataset_configs.batch_size_test = 64
  config.num_classes = 238
  reference_resolution = 224
  crop_size = 224
  config.T = 0.07

  config.dataset_configs.filter_classes = False
  if config.dataset_configs.filter_classes:
    config.dataset_configs.desired_classes = [
                                              897, 827, 764, 761, 742, 721, 651,
                                              650, 637, 632, 620, 738, 534, 508,
                                              435, 412, 879, 859, 463, 470, 481,
                                              473, 587, 313, 872, 629, 745, 760,
                                              963, 938, 937, 987, 943, 955, 953,
                                              957, 954, 752, 792, 626, 951, 112,
                                              928
                                              ]
    #update number classes variables
    config.num_classes_filter = config.num_classes#len(config.dataset_configs.desired_classes)
    _IMAGENET_TRAIN_SIZE = 1281167#732-1300 per class in the ILSVRC2012 training set. #update quantity samples train each class selected
    _IMAGENET_TEST_SIZE = 2134#update quantity samples train each class selected
  else:
    config.num_classes_filter = config.num_classes


  config.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "image_resized")' +
      #f'|adjust_labels({config.dataset_configs.desired_classes}, {config.num_classes},{config.dataset_configs.filter_classes}, key="label", key_result="label_adj")' +
      f'|onehot({config.num_classes_filter}, key="label", key_result="label_onehot")' +
      '|resize_small(256, data_key="image")'+
      '|resize_small(256, data_key="image_resized")'+
      '|central_crop(224, data_key="image")'+
      '|central_crop(224, data_key="image_resized")'+
      '|value_range(0, 1, data_key="image")' +
      '|value_range(0, 1, data_key="image_resized")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image")'+
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image_resized")'+
      '|keep("image", "image_resized", "label", "label_onehot")'
      #'|keep("image", "image_resized", "label_adj", "label", "label_onehot")'
  )


  ### kNN

  #dir of checkpoints
  config.train_dir = '/mnt/disks/stg_dataset/test_test2'#'/home/jesimonbarreto/exp_test_now/'
  config.preextracted = False
  config.write_summary = True
  config.steps_checkpoints = [-1]
  config.ks = [1,3,5,7,10,20]
  config.dir_files = '/mnt/disks/stg_dataset/eval_files/'

  config.data_dtype_str = 'float32'
  #config.data_dtype_str = 'bfloat16'


  n_queries = 10
  
  config.batch_size = config.dataset_configs.batch_size_train #batch size for extracting embeddings
  #config.knn_eval_batch_size = 32 #batch size for batch knn search 
  
  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 400
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.dataset_configs.batch_size_train
  config.steps_per_epoch_eval = _IMAGENET_TEST_SIZE // config.dataset_configs.batch_size_test
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.4)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 8
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.04
  config.warmup_teacher_temp_epochs = 0
  config.dataset_configs.number_of_focal_queries = n_queries - 1

  
  #MODEL

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
  config.model.head_output_dim = 65536 #4096 #65536
  #head
  config.model.n_layers = 3 #1
  config.model.head_hidden_dim = 2048
  config.model.head_bottleneck_dim = 256 #64 #256
  ##
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05

  # Dino specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = -1#int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  config.checkpoint = False#'/home/jesimonbarreto/scenic/checkpoint_501'
  config.dir_weight = '/home/jesimonbarreto/'
  config.weight_load = 'dinov2_vit'+version.lower()+'14' #'dino_vitb16' #'dinov2_vit'+version.lower()+'14'
  # Learning rate.
  #cosine schedule lr
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = config.steps_per_epoch * 15
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.001 * config.dataset_configs.batch_size_train / 1024
  config.lr_configs.alpha = 0.01

  # Weight decay.
  config.weight_decay = 0.04
  #verificar
  config.weight_decay_end = 0.4
  config.lr=0.0005
  config.warmup_epochs=10
  config.optimizer = 'adamw'
  config.drop_path_rate= 0.1

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1. / 0.996

  return config