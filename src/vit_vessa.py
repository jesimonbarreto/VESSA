"""Vision Transformer used in DINO."""

import copy
import functools
from typing import Any, Optional, Tuple

from collections.abc import Callable

import flax.linen as nn
from flax.linen import initializers, Module, compact
import jax
from jax import lax
from jax import nn as opr
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit

class ToTokenSequence(nn.Module):
  """Transform a batch of views into a sequence of tokens."""
  
  patches: ml_collections.ConfigDict
  hidden_size: int
  posembs: Tuple[int, int] = (14, 14)
  positional_embedding: str = 'learned'

  def add_positional_encodings(self, x: jnp.ndarray, w:int, h:int, c:int,
                               positional_embedding: str = '') -> jnp.ndarray:
    """Add positional encodings to the input patch sequence."""

    positional_embedding = positional_embedding or self.positional_embedding
    if positional_embedding == 'learned':
      posemb = self.param(
          'posembed_input',
          nn.initializers.normal(stddev=1/np.sqrt(c)),
          (1, self.posembs[0], self.posembs[1], c), x.dtype)
      # Optionally resize the positional encodings.
      if (h, w) != self.posembs:
        posemb = jax.image.resize(posemb, (1, w, h, c), 'bicubic')
      x = x + posemb
    
    elif positional_embedding == 'learned_1d':
      if w*h == 196:
        pos_emb_shape = (1, (w*h) + 1, c)
        pe = self.param('pos_embedding',
                        nn.initializers.normal(stddev=1/np.sqrt(c)),
                        pos_emb_shape,
                        x.dtype)
      else:
        pos_emb_shape = (1, (w*h) + 1, c)
        pe = self.param('pos_embedding_c',
                        nn.initializers.normal(stddev=1/np.sqrt(c)),
                        pos_emb_shape,
                        x.dtype)

      x = x + pe
    elif positional_embedding == 'sinusoidal_2d':
      x = attention_layers.AddFixedSinCosPositionEmbedding()(x)
    
    return x, posemb

  @nn.compact
  def __call__(self, x: jnp.ndarray, 
               positional_embedding:str = '',
               seqlen: int = -1, 
               seqlen_selection: str = 'unstructured'):
    # Extracting patches and then embedding is in fact a single convolution.
    fh, fw = self.patches.size
    n, w, h, c = x.shape
    #x = jnp.transpose(x, (0, 2, 3, 1))
    x = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                name='embedding')(x)
    
    n, w, h, c = x.shape

    is_initialized = self.has_variable('params','cls')
    if not is_initialized:
      cls = self.param('cls', nn.initializers.normal(1e-6), (1, 1, self.hidden_size), x.dtype)
    else:
      cls = self.get_variable('params','cls')
    
    # Adding positional encodings.
    x, posemb = self.add_positional_encodings(x, w, h, c, positional_embedding)
    
    x = jnp.reshape(x, (n, w*h, c))

    cls_exp = jnp.tile(cls, (x.shape[0], 1, 1))
    x = jnp.concatenate((cls_exp, x), axis=1)

    # Possibly dropping some tokens.
    idx_kept_tokens = None
    n_tokens = self.posembs[0] * self.posembs[1]
    if seqlen > 0:
      rng = self.make_rng('droptok')
      idx_kept_tokens = token_indexes_not_to_drop(
          seqlen, n_tokens, seqlen_selection, rng)
      if len(idx_kept_tokens) < n_tokens:
        x = jnp.take(x, idx_kept_tokens, axis=1)

    return x, posemb


class LoRA(nn.Module):
    """LoRA (Low-Rank Adaptation) usando a estrutura de Dense."""
    input_dim: int  # Última dimensão da entrada (feature_dim)
    rank: int  # Dimensão reduzida de projeção
    lora_A_init: Callable = initializers.lecun_normal()
    lora_B_init: Callable = initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        """Aplica LoRA na última dimensão de x."""
        lora_A = self.param('lora_A', self.lora_A_init, (self.input_dim, self.rank))
        lora_B = self.param('lora_B', self.lora_B_init, (self.rank, self.input_dim))

        # Multiplicação na última dimensão (feature_dim)
        x_proj = lax.dot_general(x, lora_A, (((x.ndim - 1,), (0,)), ((), ())))  # (32, 6, 197, r)
        x_proj = lax.dot_general(x_proj, lora_B, (((x_proj.ndim - 1,), (0,)), ((), ())))  # (32, 6, 197, 197)

        return x + x_proj  # Aplica adaptação de LoRA

class Encoder1DBlockLORA(nn.Module):
  """Transformer encoder layer com LoRA."""
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  lora_use: bool = False
  rank: int = 64  # Parâmetro do LoRA

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Aplica o bloco Encoder1DBlockLORA."""
    assert inputs.ndim == 3
    d_model = inputs.shape[-1]  # Dimensão do embedding

    # Normalização antes da atenção
    x = nn.LayerNorm(dtype=self.dtype)(inputs)

    # Aplicar LoRA nas projeções query e value
    
    lora_q = LoRA(d_model, self.rank)(x)
    lora_v = LoRA(d_model, self.rank)(x)

    # Atenção modificada com LoRA
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate
    )(lora_q, lora_v)

    # Dropout e Stochastic Depth
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs  # Residual Connection

    # MLP block
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6)
    )(y, deterministic=deterministic)

    # Stochastic Depth e Residual Connection
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


def token_indexes_not_to_drop(seqlen, n_tokens, seqlen_selection, rng):
  """Returns only the token indexes to keep in a sequence of tokens."""
  idx_kept_tokens = jnp.arange(n_tokens)
  if seqlen > 0 and seqlen <= n_tokens:
    if seqlen_selection in ['consecutive', 'first']:
      if seqlen_selection == 'first':
        offset = 0
      else:
        offset = jax.random.randint(rng, (1,), 0, n_tokens - seqlen + 1)[0]
      # Workaround because jnp.arange(offset, offset + seqlen) causes
      # a ConcretizationError (even though shape is known to be seqlen...)
      idx_kept_tokens = jnp.ones(seqlen) * offset + jnp.arange(seqlen)
    elif seqlen_selection == 'unstructured':
      idx_kept_tokens = jax.random.permutation(rng, n_tokens)[:seqlen]
  idx_kept_tokens = jnp.asarray(idx_kept_tokens, dtype=jnp.int32)
  return idx_kept_tokens


class ViTDINO(nn.Module):
  """Vision Transformer model for LOCA training.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    n_ref_positions: Number of position in the reference view.
    apply_cluster_loss: Whether to apply the clustering loss.
    head_hidden_dim: Dimension of the hidden layer in the projection mlp.
    head_bottleneck_dim: Dimension of the bottleneck.
    head_output_dim: Dimension of the output ("number of prototypes").
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: Stochastic depth.
    posembs: Positional embedding size.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  n_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  lora_use: bool
  lora_rank: int
  last_layers_train_lora: int
  apply_cluster_loss: bool
  head_hidden_dim: int
  n_ref_positions: int
  head_bottleneck_dim: int
  head_output_dim: int
  positional_embedding: str = 'learned'
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.1
  posembs: Tuple[int, int] = (14, 14)
  dtype: Any = jnp.float32
  loca: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, inputs_kv: Optional[jnp.ndarray] = None,
               train: bool, seqlen: int = -1, use_pe: bool = True,
               drop_moment: str = 'early',
               seqlen_selection: str = 'unstructured', 
               backbone: bool = True,
               debug: bool = False):
    del debug
    # Input image -> sequence of patch tokens.
    to_token_fn = ToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs,
        positional_embedding=self.positional_embedding) 
    x, pos = to_token_fn(
        x,
        #self.positional_embedding, 
        seqlen=seqlen if drop_moment == 'early' else -1,
        positional_embedding=self.positional_embedding if use_pe else 'pe_not_in_use',
        seqlen_selection=seqlen_selection)
    #x_pre = x.copy()
    # ViT Encoder.
    for lyr in range(self.num_layers):
      if self.lora_use:
         if lyr < self.num_layers - self.last_layers_train_lora:
            x = Encoder1DBlockLORA(
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
              self.stochastic_depth,
              name=f'encoderblock_{lyr}',
              rank=self.lora_rank, 
              dtype=jax.dtypes.canonicalize_dtype(self.dtype)
              )(x, deterministic=not train)
         else:
            x = vit.Encoder1DBlock(
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
              self.stochastic_depth,
              name=f'encoderblock_{lyr}',
              dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
                  x, deterministic=not train)
      else:   
        x = vit.Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
            self.stochastic_depth,
            name=f'encoderblock_{lyr}',
            dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
                x, deterministic=not train)
    x_norm = nn.LayerNorm(name='encoder_norm')(x)
    x_cls = x_norm[:, 0]

    x_train = ProjectionHead(
          hidden_dim=self.head_hidden_dim,
          bottleneck_dim=self.head_bottleneck_dim,
          output_dim=self.head_output_dim,
          n_layers=self.n_layers,
          name='projection_head')(
              x_cls, train)

    return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "x_norm": x_norm,
            "masks": None,
            "x_train": x_train
        }

def norm_kernel_init_fn(rng, shape, dtype):
  """Initialize kernel with l2 normalized columns."""
  param = nn.linear.default_kernel_init(rng, shape, dtype)
  param /= (jnp.linalg.norm(param, axis=0, keepdims=True) + 1e-10)
  return param

class ProjectionModule(nn.Module):
  """Projection head.

  Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    n_layers: Depth of the projection head.
  """
  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 4096
  n_layers: int = 2

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
    for i in range(self.n_layers):
      x = nn.Dense(self.hidden_dim)(x)
      x = nn.gelu(x)
      x = nn_layers.IdentityLayer(name=f'mlp_{i}')(x)
    x = nn.Dense(self.bottleneck_dim)(x)
    # Normalize.
    x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = WeightNormDense(self.output_dim, use_bias=False, name='prototypes',
                        kernel_init=norm_kernel_init_fn)(x)
    return x
  
class ProjectionHead(nn.Module):
  """Projection head.

  Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    n_layers: Depth of the projection head.
  """
  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 4096
  n_layers: int = 2

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
    for i in range(self.n_layers):
      x = nn.Dense(self.hidden_dim)(x)
      x = nn.gelu(x)
      x = nn_layers.IdentityLayer(name=f'mlp_{i}')(x)
    x = nn.Dense(self.bottleneck_dim)(x)
    # Normalize.
    x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = WeightNormDense(self.output_dim, use_bias=False, name='prototypes',
                        kernel_init=norm_kernel_init_fn)(x)
    return x


class WeightNormDense(nn.Dense):
  """Linear layer with weight normalized kernel."""

  def param(self, name: str, *args, **kwargs):
    param = super().param(name, *args, **kwargs)
    if name == 'kernel':
      param /= (jnp.linalg.norm(param, axis=0, keepdims=True) + 1e-10)
    return param


class CrossAttentionEncoderBlock(vit.Encoder1DBlock):
  """Transformer layer with cross-attention."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, inputs_kv: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    inputs_kv = nn.LayerNorm(dtype=self.dtype)(inputs_kv)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, inputs_kv)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class ViTVessaModel(base_model.BaseModel):
  """Vision Transformer model for DINO training."""

  
  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    self.student_temp = self.config.student_temp
    self.center_momentum = self.config.center_momentum
    self.ncrops = self.config.ncrops + 2
    self.out_dim = self.config.model.head_output_dim
    
    # we apply a warm up for the teacher temperature because
    # a too high temperature makes the training instable at the beginning
    self.teacher_temp_schedule = jnp.concatenate((
        jnp.linspace(self.config.warmup_teacher_temp,
                    self.config.teacher_temp, self.config.warmup_teacher_temp_epochs),
        jnp.ones(self.config.num_training_epochs - self.config.warmup_teacher_temp_epochs) * self.config.teacher_temp
    ))
    return ViTDINO(
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        n_layers=self.config.model.get('n_layers', 1),
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        lora_use=self.config.get('lora_use', False),
        lora_rank=self.config.get('lora_rank', 64),
        last_layers_train_lora = self.config.get('int_train_last_layers', 1),
        n_ref_positions=self.config.n_ref_positions,
        apply_cluster_loss=self.config.apply_cluster_loss,
        head_hidden_dim=self.config.model.get('head_hidden_dim', 512),
        head_bottleneck_dim=self.config.model.get('head_bottleneck_dim', 256),
        head_output_dim=self.config.model.get('head_output_dim', 1024),
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.0),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        posembs=self.config.model.get('posembs', (16, 16)),
        dtype=model_dtype,
    )
  
  def init_from_train_state(
      self, train_state, restored_train_state, restored_model_cfg):
    """Updates the train_state with data from restored_train_state.

    This function is written to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return vit.init_vit_from_train_state(train_state, restored_train_state,
                                         self.config, restored_model_cfg)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                head_hidden_dim=32,
                head_bottleneck_dim=16,
                head_output_dim=64,
                patches={'size': (4, 4)},
                data_dtype_str='float32')
    })

  def get_metrics_fn(self, split: Optional[str] = None):
    del split
    return functools.partial(
        classification_model.classification_metrics_function,
        target_is_onehot=True,
        metrics=dict(
            {'accuracy': (
                model_utils.weighted_correctly_classified,
                model_utils.num_examples),
             'loss': (
                 model_utils.weighted_unnormalized_softmax_cross_entropy,
                 model_utils.num_examples)}))

  def loss_function(self,
                    teacher_output: jnp.ndarray,
                    student_output: jnp.ndarray,
                    center: jnp.ndarray,
                    epoch: int,
                    weights: Optional[jnp.ndarray] = None) -> float:
    """Returns the cross-entropy loss."""

    #loss = model_utils.weighted_softmax_cross_entropy(predictions, targets,
    #                                                  weights)

    student_out = student_output / self.student_temp
    student_out = jnp.split(student_out, self.ncrops)
    
    #jax.debug.print("🤯 Epoca: {epoch} 🤯", epoch=epoch)
    # teacher centering and sharpening
    temp = self.teacher_temp_schedule[epoch]
    teacher_out = opr.softmax((teacher_output - center) / temp, axis=-1)
    teacher_out = jnp.split(lax.stop_gradient(teacher_out),2)

    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = jnp.sum(-q * opr.log_softmax(student_out[v], axis=-1), axis=-1)
            total_loss += jnp.mean(loss)
            n_loss_terms += 1
    total_loss /= n_loss_terms
    center = self.update_center(teacher_output, center)
    #jax.debug.print("🤯 Center Depois: {center} 🤯", center=center)
    return total_loss, center


  
  def loss_function_uncertainty(self,
                  teacher_output: jnp.ndarray,
                  student_output: jnp.ndarray,
                  center: jnp.ndarray,
                  epoch: int,
                  weights: Optional[jnp.ndarray] = None) -> float:
    """VESSA loss function modified to improve discrimination via uncertainty weighting."""

    # Normalize student output
    student_out = student_output / self.student_temp
    student_out = jnp.split(student_out, self.ncrops)

    # Adjust teacher temperature
    temp = self.teacher_temp_schedule[epoch]
    teacher_out = opr.softmax((teacher_output - center) / temp, axis=-1)
    teacher_out = jnp.split(lax.stop_gradient(teacher_out), 2)

    # Initialize total loss and loss term counter
    total_loss = 0
    n_loss_terms = 0

    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                continue  # Skip cases where student and teacher use the same view
            
            # 🔹 Compute Original DINO Loss 🔹
            loss = jnp.sum(-q * opr.log_softmax(student_out[v], axis=-1), axis=-1)

            # 🔥 **Uncertainty Weighting** 🔥
            entropy = -jnp.sum(q * jnp.log(q + 1e-6), axis=-1)  # Teacher entropy
            gamma = 1.0  # Weighting adjustment factor
            loss = (1 + gamma * entropy) * loss  # Higher weight for high-uncertainty examples

            total_loss += jnp.mean(loss)
            n_loss_terms += 1

    # Normalize final loss
    total_loss /= n_loss_terms

    # Update teacher center
    center = self.update_center(teacher_output, center)

    return total_loss, center


  def cosine_similarity(self, A, B):
    dot_product = jnp.dot(A, B)
    norm_A = jnp.linalg.norm(A)
    norm_B = jnp.linalg.norm(B)
    return dot_product / (norm_A * norm_B + 1e-9) 

  def cosine_distance(self, A, B):
      return 1 - self.cosine_similarity(A, B)

  def cosine_loss(self, preds, targets):
      cosine_distances = jnp.array([self.cosine_distance(p, t) for p, t in zip(preds, targets)])
      return jnp.mean(cosine_distances)
  
  
  # Função para calcular L2 loss
  def l2_loss(self, preds, targets):
    return jnp.mean((preds - targets) ** 2)

  def loss_lwf(self,
                    teacher_output: jnp.ndarray,
                    student_output: jnp.ndarray,
              ) -> float:
    """Returns the cross-entropy loss."""

    
    student_out = student_output / self.student_temp
    student_out = jnp.split(student_out, 2)
    
    teacher_out = opr.softmax((teacher_output) / self.student_temp, axis=-1)
    teacher_out = jnp.split(lax.stop_gradient(teacher_out), 2)

    #jax.debug.print("🤯 teaf: {epoch} 🤯", epoch=teacher_out.shape)
    #jax.debug.print("🤯 stuf: {epoch} 🤯", epoch=student_out.shape)
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = jnp.sum(-q * opr.log_softmax(student_out[v], axis=-1), axis=-1)
            total_loss += jnp.mean(loss)
            n_loss_terms += 1
    total_loss /= n_loss_terms
    #jax.debug.print("🤯 Lossf: {epoch} 🤯", epoch=total_loss.shape)

    return total_loss
    
  
  def reduce(self, value):
    # Dummy function to simulate the reduction operation
    def reduce_sum(x):
      return jax.lax.psum(x, axis_name='batch')
    # Perform the reduction
    global_sum = jax.pmap(reduce_sum)(value)
    return global_sum
    

  def update_center(self, teacher_out, center):
      """
      Update center used for teacher output.
      """
      teacher_output = lax.stop_gradient(teacher_out)
      batch_center = jnp.sum(teacher_output, axis=0, keepdims=True)
      batch_center = self.reduce(batch_center)
      batch_center = batch_center / (len(teacher_output) * jax.local_device_count())
      # ema update
      center = center * self.center_momentum + batch_center * (1 - self.center_momentum)
      return center