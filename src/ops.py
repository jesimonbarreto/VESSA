"""Implementation of data preprocessing ops useful for LOCA training."""
import functools

from scenic.dataset_lib.big_transfer import registry
from scenic.dataset_lib.big_transfer.preprocessing import utils
import tensorflow as tf
import tensorflow.compat.v2 as tf2
import jax.numpy as jnp
import tensorflow_addons as tfa  # para affine transform mais poderosa

# The two following decorators mimic the support for single-input
# single-output ops already in scenic.dataset_lib.big_transfer.preprocessing and
# adapt them to the preprocessing of ops with two `inkey`` and two `outkey`
# arguments.
class TwoInKeysTwoOutKeys(object):
  """Decorator for preprocessing ops with two inkey and two outkey arguments.

  Note: Support for single-input single-output ops is in already in
  scenic.dataset_lib.big_transfer.preprocessing.InKeyOutKey.
  """

  def __init__(self, indefault=("inputs", "label"),
               outdefault=("inputs", "label")):
    self.indefault = indefault
    self.outdefault = outdefault

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args,
                       inkey=self.indefault,
                       outkey=self.outdefault,
                       **kw):
      orig_pp_fn = orig_get_pp_fn(*args, **kw)
      def _ikok_pp_fn(data):
        data[outkey[0]], data[outkey[1]] = orig_pp_fn(
            data[inkey[0]], data[inkey[1]])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn


class BatchedImagePreprocessingWithMask(object):
  """Decorator for preprocessing ops which adds support for image batches.

  Note: Doesn't support decorating ops which add new fields in data.
  """

  def __call__(self, get_pp_fn):
    def tf_apply_to_image_or_images(fn, image_or_images, mask_or_masks):
      """Applies a function to a single element or each element in a batch."""
      static_rank = len(image_or_images.get_shape().as_list())
      if static_rank == 3:  # A single image: HWC
        return fn(image_or_images, mask_or_masks)
      elif static_rank == 4:  # A batch of images: BHWC
        aux = [fn(x, y) for x, y in zip(tf.unstack(image_or_images),
                                        tf.unstack(mask_or_masks))]
        return tf.stack([x for (x, _) in aux]), tf.stack([y for (_, y) in aux])
      else:
        raise ValueError("Unsupported image rank: %d" % static_rank)

    def get_batch_pp_fn(*args, **kwargs):
      """Preprocessing function that supports batched images."""

      def _batch_pp_fn(image, mask, *a, **kw):
        orig_image_pp_fn = get_pp_fn(*args, **kwargs)
        orig_image_pp_fn = functools.partial(orig_image_pp_fn, *a, **kw)
        return tf_apply_to_image_or_images(orig_image_pp_fn, image, mask)

      return _batch_pp_fn

    return get_batch_pp_fn


# The three following functions and decorators mimic the support for
# single-input single-output ops already present in:
# scenic.dataset_lib.big_transfer.preprocessing and adapt them to the
# preprocessing of ops with two `inkey`` and two `outkey` arguments.
def tf_apply_to_image_mask_box(fn, image_or_images, mask_or_masks,
                               box_or_boxes):
  """Applies a function to a single element or each element in a batch."""
  static_rank = len(image_or_images.get_shape().as_list())
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images, mask_or_masks, box_or_boxes)
  elif static_rank == 4:  # A batch of images: BHWC
    aux = [fn(x, y, z) for x, y, z in zip(tf.unstack(image_or_images),
                                          tf.unstack(mask_or_masks),
                                          tf.unstack(box_or_boxes))]
    return (tf.stack([x for (x, _, _) in aux]),
            tf.stack([y for (_, y, _) in aux]),
            tf.stack([z for (_, _, z) in aux]))
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


class BatchedImagePreprocessingWithMaskAndBox(object):
  """Decorator for preprocessing ops, which adds support for image batches.

  Note: Doesn't support decorating ops which add new fields in data.
  """

  def __call__(self, get_pp_fn):
    def get_batch_pp_fn(*args, **kwargs):
      """Preprocessing function that supports batched images."""

      def _batch_pp_fn(image, mask, box, *a, **kw):
        orig_image_pp_fn = get_pp_fn(*args, **kwargs)
        orig_image_pp_fn = functools.partial(orig_image_pp_fn, *a, **kw)
        return tf_apply_to_image_mask_box(orig_image_pp_fn, image, mask, box)

      return _batch_pp_fn

    return get_batch_pp_fn


class ThreeInKeysThreeOutKeys(object):
  """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.

  Note: Support for single-input single-output ops is in already in
  scenic.dataset_lib.big_transfer.preprocessing.InKeyOutKey.
  """

  def __init__(self, indefault=("inputs", "label", "box"),
               outdefault=("inputs", "label", "box")):
    self.indefault = indefault
    self.outdefault = outdefault

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args,
                       inkey=self.indefault,
                       outkey=self.outdefault,
                       **kw):

      orig_pp_fn = orig_get_pp_fn(*args, **kw)

      def _ikok_pp_fn(data):
        data[outkey[0]], data[outkey[1]], data[outkey[2]] = orig_pp_fn(
            data[inkey[0]], data[inkey[1]], data[inkey[2]])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn


@registry.Registry.register("preprocess_ops.init_patch_matching_tracker",
                            "function")
def init_patch_matching_tracker(size, outkey="mask"):
  """Initialize square grid to track patches correspondances in a mask."""

  def _init_patch_matching_tracker(data):
    data[outkey] = tf.reshape(tf.range(size**2), [size, size, 1])
    return data

  return _init_patch_matching_tracker


@registry.Registry.register("preprocess_ops.init_box_tracker", "function")
def init_box_tracker(outkey="box"):
  """Initialize box coordinates that will track the view intersections."""

  def _init_box_tracker(data):
    # First dim is for unvalid/valid box. Format is y,x,h,w.
    data[outkey] = tf.zeros((5), tf.float32)
    return data

  return _init_box_tracker


@registry.Registry.register("preprocess_ops.cropflip_generatemask", "function")
@ThreeInKeysThreeOutKeys()
@BatchedImagePreprocessingWithMaskAndBox()
def cropflip_generatemask(resize_size=224, area_min=5, area_max=100, flip=True,
                          resize_method=tf.image.ResizeMethod.BILINEAR):
  """Crop and flip an image and keep track of these operations with a mask."""
  def _cropflip_generatemask(image, mask, box):
    orig_shape = tf.shape(image)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        orig_shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, size)
    box = tf.concat((box[:1],
                     tf.cast(begin[:2], tf.float32),
                     tf.cast(size[:2], tf.float32)), axis=-1)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    crop = tf.image.resize(crop, [resize_size, resize_size], resize_method)

    # flip
    if flip:
      seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
      crop = tf.image.stateless_random_flip_left_right(crop, seed)
      mask = tf.image.stateless_random_flip_left_right(mask, seed)

    resized_mask = tf.image.resize(mask, size[:2], "nearest")
    paddings = [[begin[0], orig_shape[0] - size[0] - begin[0]],
                [begin[1], orig_shape[1] - size[1] - begin[1]], [0, 0]]
    full_mask = tf.pad(resized_mask, paddings, "CONSTANT", constant_values=-1)

    return crop, full_mask, box
  
  return _cropflip_generatemask

@registry.Registry.register("preprocess_ops.copy_file", "function")
@ThreeInKeysThreeOutKeys()
@BatchedImagePreprocessingWithMaskAndBox()
def copy_file(resize_size=224):
  """Crop and flip an image and keep track of these operations with a mask."""
  def copy_file(image):
    orig_shape = tf.shape(image)
    return image,image,image

  return copy_file


@registry.Registry.register("preprocess_ops.random_crop_distorcedbb", "function")
@utils.InKeyOutKey()
def random_crop_distorcedbb(resize_size=224, global_scale=None):
  """Crop and flip an image and keep track of these operations with a mask."""
  def dbb_random_crop(image):

    resize_method=tf.image.ResizeMethod.BICUBIC
    #resized_image = tf.image.resize(image, [resize_size, resize_size], resize_method)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
        area_range= global_scale,
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    # Process image:
    image_cropped = tf.slice(image, begin, size)
    image_cropped.set_shape([None, None, image.shape[-1]])
    image_cropped = tf.image.resize(image_cropped, [resize_size, resize_size], resize_method)
    
    '''seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
    image_cropped = tf.image.stateless_random_flip_left_right(image_cropped, seed)
    image = tf.image.resize(image, [256, 256], resize_method)
    image = tf.image.central_crop(image, 0.875)
    image = tf.image.resize(image, [resize_size, resize_size], resize_method)'''

    return image_cropped
  return dbb_random_crop

@registry.Registry.register("preprocess_ops.copy_resize_file", "function")
@TwoInKeysTwoOutKeys()
def copy_resize_file(resize_size=224, global_scale=None):
  """Crop and flip an image and keep track of these operations with a mask."""
  def copy_resize_file(image, image_):

    resize_method=tf.image.ResizeMethod.BICUBIC
    #resized_image = tf.image.resize(image, [resize_size, resize_size], resize_method)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
        area_range= global_scale,
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    # Process image:
    image_cropped = tf.slice(image, begin, size)
    image_cropped.set_shape([None, None, image.shape[-1]])
    image_cropped = tf.image.resize(image_cropped, [resize_size, resize_size], resize_method)
    
    seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
    image_cropped = tf.image.stateless_random_flip_left_right(image_cropped, seed)
    image = tf.image.resize(image, [256, 256], resize_method)
    image = tf.image.central_crop(image, 0.875)
    image = tf.image.resize(image, [resize_size, resize_size], resize_method)

    return image, image_cropped
  return copy_resize_file


'''@registry.Registry.register("preprocess_ops.resize_small", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_resize_small(smaller_size, method="area", antialias=True):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: See TF's image.resize method.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.

  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    dtype = image.dtype
    image = tf2.image.resize(image, (h, w), method, antialias)
    return tf.cast(image, dtype)

  return _resize_small
'''
@registry.Registry.register("preprocess_ops.vessa_transform", "function")
@utils.InKeyOutKey()
def vessa_transform(size=224, crop_size=224, mean=[0.5], std=[0.5]):
  """Crop and flip an image and keep track of these operations with a mask."""
  def vessa_transform(image):

    def to_tensor(image):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32) / 255.0
      # Transpose to (C x H x W) format
      image = tf.transpose(image, perm=[2, 0, 1])
      return image
  
    def center_crop(image, crop_size):
      pad_width = 2
      # Define the padding configuration
      padding = [[pad_width, pad_width], [pad_width, pad_width], [0, 0]]
      image = tf.pad(image, padding, mode='CONSTANT', constant_values=0)
      
      # Get the dimensions of the image
      height, width = tf.shape(image)[0], tf.shape(image)[1]

      # Calculate the crop offsets
      offset_height = int((height - crop_size[0]) / 2)
      offset_width = int((width - crop_size[1]) / 2)

      # Crop the image
      cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size[0], crop_size[1])
      return cropped_image
    
    def resize(image, target_size):
      # Get the dimensions of the image
      height, width = tf.shape(image)[0], tf.shape(image)[1]

      # Determine the scaling factor to make the smaller dimension 224
      scale_factor = tf.cond(height < width,
                            lambda: target_size / height,
                            lambda: target_size / width)

      # Resize the image while preserving the aspect ratio
      new_height = int(height * scale_factor)
      new_width = int(width * scale_factor)
      resized_image = tf.image.resize(image, [new_height, new_width])

      return resized_image
    
    # Define image transformation pipeline using tf.image
    def transform_image(image):
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = resize(image, size)  # Resize image
        image = center_crop(image, crop_size=(crop_size, crop_size))  # Center crop image
        image = to_tensor(image)  # Convert image to float32
        image = (image - mean) / std  # Normalize using provided mean and std dev
        return image
    image = transform_image(image)
    return image
  return vessa_transform

@registry.Registry.register("preprocess_ops.flip_with_mask", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing
def flip_with_mask():
  def _flip_with_mask(image, mask):
    seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    return image, mask
  return _flip_with_mask


@registry.Registry.register("preprocess_ops.inception_crop_with_mask",
                            "function")
@ThreeInKeysThreeOutKeys()
@BatchedImagePreprocessingWithMaskAndBox()
def inception_crop_with_mask(resize_size=None, area_min=5, area_max=100,
                             resize_mask=None,
                             resize_method=tf.image.ResizeMethod.BILINEAR):
  """Applies the same inception-style crop to an image and a mask tensor.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Sequence of 2 ints; Resize image to [height, width] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    resize_mask: Whether we should resize the mask.
    resize_method: Resize method.

  Returns:
    Function to crop image and mask tensors.
  """
  def _inception_crop_with_mask(image, mask, box):
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    # Process image:
    image_cropped = tf.slice(image, begin, size)
    image_cropped.set_shape([None, None, image.shape[-1]])
    if resize_size:
      image_cropped = tf.image.resize(image_cropped, resize_size, resize_method)

    # Process box:
    # max(xB - xA, 0) / wA
    box = box[1:]
    relative_begin = tf.stack((
        tf.divide(tf.maximum(tf.cast(begin[0], tf.float32) - box[0], 0.),
                  box[2]),
        tf.divide(tf.maximum(tf.cast(begin[1], tf.float32) - box[1], 0.),
                  box[3])), axis=-1)
    # (min(xA + wA, xB + wB) - (begin * WA + xA)) / wA
    relative_size = tf.stack((
        tf.maximum(0., tf.divide(tf.minimum(
            tf.cast(begin[0], tf.float32) + tf.cast(size[0], tf.float32),
            box[0] + box[2]
            ), box[2]) - relative_begin[0] - tf.divide(box[0], box[2])),
        tf.maximum(0., tf.divide(tf.minimum(
            tf.cast(begin[1], tf.float32) + tf.cast(size[1], tf.float32),
            box[1] + box[3]
            ), box[3]) - relative_begin[1] - tf.divide(box[1], box[3]))),
                             axis=-1)
    # We filter out small boxes.
    valid_box = tf.cast(tf.greater(relative_size[0], 0.1), tf.float32)
    valid_box *= tf.cast(tf.greater(relative_size[1], 0.1), tf.float32)
    valid_box = tf.convert_to_tensor([valid_box])
    box = tf.concat((valid_box, relative_begin, relative_size), axis=-1)

    # Process mask:
    mask_cropped = tf.slice(mask, begin, size)
    mask_cropped.set_shape([None, None, mask.shape[-1]])
    if resize_size:
      mask_cropped = tf.image.resize(
          mask_cropped, resize_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if resize_mask:
      mask_cropped = tf.image.resize(
          mask_cropped, resize_mask, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_cropped, mask_cropped, box
  return _inception_crop_with_mask


@registry.Registry.register("preprocess_ops.random_color_jitter", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def random_color_jitter(proba, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    proba: Probability of applying color jittering.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    Function that distort image color.
  """
  def _apply_transform(i, x):
    """Apply the i-th transformation."""
    def brightness_foo():
      if brightness == 0:
        return x
      else:
        return tf.image.random_brightness(x, max_delta=brightness)
    def contrast_foo():
      if contrast == 0:
        return x
      else:
        return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
    def saturation_foo():
      if saturation == 0:
        return x
      else:
        return tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
    def hue_foo():
      if hue == 0:
        return x
      else:
        return tf.image.random_hue(x, max_delta=hue)
    x = tf.cond(tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
    return x

  @tf.function
  def _random_color_jitter(image):
    do_it = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
    if do_it < proba:
      perm = tf.random.shuffle(tf.range(4))
      for i in range(4):
        image = _apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return image
  return _random_color_jitter


@registry.Registry.register("preprocess_ops.random_grayscale", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def random_grayscale(p):
  """Randomly converts imageto gray."""
  def _to_grayscale(image):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3]),
        lambda: image)
  return _to_grayscale

@registry.Registry.register("preprocess_ops.cam_motion", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def cam_motion(max_translate=0.1,
                max_rotate=10.0,
                max_scale=0.05,
                brightness_delta=0.1,
                contrast_range=(0.9, 1.1)):
  """simulate cam motion."""
  def simulate_cam_motion(image):
    """
    Simula movimento de câmera aplicando transformações geométricas e de iluminação.
    
    Args:
        image: Tensor [H, W, 3], dtype uint8 ou float32.
        max_translate: fração da largura/altura (ex: 0.1 = 10%).
        max_rotate: graus de rotação máximos (ex: 10.0).
        max_scale: variação de escala (ex: 0.05 = ±5%).
        brightness_delta: variação de brilho.
        contrast_range: (min, max) do contraste.

    Returns:
        Imagem transformada, float32 no intervalo [0, 1].
    """
    import math

    # Garante float32 [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)

    # Parâmetros aleatórios
    angle_deg = tf.random.uniform([], -max_rotate, max_rotate)
    angle_rad = angle_deg * math.pi / 180.0
    scale = 1.0 + tf.random.uniform([], -max_scale, max_scale)
    tx = tf.random.uniform([], -max_translate, max_translate) * w
    ty = tf.random.uniform([], -max_translate, max_translate) * h

    # Matriz afim: 8 elementos
    cos_a = tf.math.cos(angle_rad) * scale
    sin_a = tf.math.sin(angle_rad) * scale
    transform = tf.convert_to_tensor([
        cos_a, -sin_a, tx,
        sin_a,  cos_a, ty,
        0.0,    0.0
    ], dtype=tf.float32)

    # Aplica transformação
    image = tfa.image.transform(image, transform, interpolation='BILINEAR')

    # Ajustes de iluminação
    image = tf.image.random_brightness(image, max_delta=brightness_delta)
    image = tf.image.random_contrast(image, lower=contrast_range[0], upper=contrast_range[1])

    # Clipa para [0, 1] só por segurança
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
  return simulate_cam_motion


def gaussian_blur(image, kernel_size, sigma, padding="SAME"):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.cast(kernel_size / 2, tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
  blur_filter = tf.exp(
      -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


@registry.Registry.register("preprocess_ops.random_blur", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def random_blur(height=224, p=1.0):
  """Randomly blurs the image."""
  def _random_blur(image):
    sig = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    ks = height // 10
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: gaussian_blur(image, kernel_size=ks, sigma=sig, padding="SAME"),
        lambda: image)
  return _random_blur


@registry.Registry.register("preprocess_ops.random_solarize", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def random_solarization(p=0.1):
  """Randomly solarizes the images."""
  def _solarize(image):
    image = image * tf.cast(tf.less(image, 0.5), tf.float32) + (
        1.0 - image) * tf.cast(tf.greater_equal(image, 0.5), tf.float32)
    return image
  def _random_solarize(image):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: _solarize(image),
        lambda: image)
  return _random_solarize



@registry.Registry.register("preprocess_ops.generate_crops",
                            "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def generate_crops(resize_size=None,
                   scale_crops=None,
                   flip=True,
                   resize_method=tf.image.ResizeMethod.BILINEAR):
  """Applies the same inception-style crop to an image and a mask tensor.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Sequence of 2 ints; Resize image to [height, width] after crop.
    scale_crops: scale
    resize_method: Resize method.

  Returns:
    Function to crop image and x tensors.
  """
  def _generate_crops(image):
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
        area_range= scale_crops,
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    # Process image:
    image_cropped = tf.slice(image, begin, size)
    image_cropped.set_shape([None, None, image.shape[-1]])

    if resize_size:
      image_cropped = tf.image.resize(image_cropped, resize_size, resize_method)

    if flip:
      seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
      image_cropped = tf.image.stateless_random_flip_left_right(image_cropped, seed)
    
    return image_cropped
  return _generate_crops


@registry.Registry.register("preprocess_ops.random_flip_image", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def random_flip_image(p=0.1):
  """Randomly converts imageto gray."""
  def _to_flip(image):
    seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    return image
  return _to_flip

@registry.Registry.register("preprocess_ops.concatenate", "function")
def concatenate(*keys):
  """Keeps only the given keys."""

  def _concatenate(data):
    return {k: v for k, v in data.items() if k in keys}

  return _concatenate


@registry.Registry.register("preprocess_ops.adjust_labels", "function")
def adjust_labels(desired_classes,
                  num_classes,
                  filter_classes=True,
                  key="labels",
                  key_result="labels_adj"):
  
  """adjust encodes the input.
  """
  # Função para ajustar os rótulos para serem de 0 a len(desired_classes)-1
  def _adjust_labels(data):
    if filter_classes:
      class_mapping = {cls: i for i, cls in enumerate(desired_classes)}
      class_mapping = [class_mapping.get(i, -1) for i in range(num_classes)]
      class_mapping = tf.constant(class_mapping)
      data[key_result] = class_mapping[data[key]]
    return data
  return _adjust_labels


@registry.Registry.register("preprocess_ops.adjust_ids", "function")
def adjust_ids(   key="tfds_id",
                  key_result="tfds_id"):
  
  """adjust encodes the input.
  """
  # Alfabeto fixo (adicione mais símbolos se quiser)
  # Função para ajustar os rótulos para serem de 0 a len(desired_classes)-1
  def _adjust_ids(data):

    # Alfabeto base
    ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-@. ")
    CHAR_TO_INT = {c: i+1 for i, c in enumerate(ALPHABET)}  # +1 para reservar 0 para padding
    INT_TO_CHAR = {i: c for c, i in CHAR_TO_INT.items()}

    # Tabelas TensorFlow
    keys_tensor = tf.constant(list(CHAR_TO_INT.keys()))
    vals_tensor = tf.constant(list(CHAR_TO_INT.values()), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=0)

    reverse_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vals_tensor, keys_tensor), default_value='?')

    # Função para codificar string → int tensor
    def encode_string_tf(s: tf.Tensor, max_len: int = 128) -> tf.Tensor:
        chars = tf.strings.unicode_split(s, input_encoding='UTF-8')
        ids = table.lookup(chars)
        ids = ids[:max_len]
        pad_len = tf.maximum(0, max_len - tf.shape(ids)[0])
        return tf.concat([ids, tf.zeros([pad_len], dtype=tf.int64)], axis=0)
    
    data[key_result] = encode_string_tf(data[key])
    return data
  return _adjust_ids

@registry.Registry.register("preprocess_ops.copy_video", "function")
def get_copy_video(inkey, outkeys):
  """Copies value of `inkey` into `outkey`."""

  def video_copy(data):
    data[outkeys] = data[inkey]
    return data

  return video_copy


@registry.Registry.register("preprocess_ops.decode_video", "function")
@utils.InKeyOutKey()
def get_decode_video(channels=3):
  """Decode an encoded image string, see tf.io.decode_image."""

  def video_decode(image):  # pylint: disable=missing-docstring
    # tf.io.decode_image does not set the shape correctly, so we use
    # tf.io.deocde_jpeg, which also works for png, see
    # https://github.com/tensorflow/tensorflow/issues/8551
    print(image)
    images = tf.map_fn(
        functools.partial(tf.io.decode_jpeg, channels=channels),
        image,
        #back_prop=False,
        dtype=tf.uint8
    )
    return images
    #return tf.io.decode_jpeg(image, channels=channels)

  return video_decode