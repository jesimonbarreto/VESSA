"""Loading pretrained params utilities."""
import re
from typing import Any, Dict, Optional

from absl import logging
import flax
from flax.training import checkpoints
import jax
import jax.profiler
import numpy as np
from scenic.train_lib import pretrain_utils
from tensorflow.io import gfile
import os,io
from typing import Mapping
import dataclasses


def split_qkv_w(qkv, nheads):
  # qkv in pytorch is for example [2304, 768], in bv it's 3x [768, 12, 64]
  win, wout = qkv.shape[1], qkv.shape[0] // 3
  qkv = qkv.T.reshape(win, 3, nheads, wout // nheads)
  return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def split_qkv_b(qkv, nheads):
  qkv = qkv.reshape(3, nheads, -1)
  return qkv[0], qkv[1], qkv[2]


def get_param_from_pytorch_style_dict(vessa, scenic_name, nheads,
                                      absorbe_ls=True):
  """Converts key name from vessa official weights to this codebase name."""
  n = int((vessa['pos_embed'].shape[1])**0.5)
  easy_mappings = {
      'ToTokenSequence_0/cls': vessa['cls_token'] + vessa['pos_embed'][:, 0, :],
      'ToTokenSequence_0/posembed_input':
          vessa['pos_embed'][:, 1:, :].reshape(1, n, n, -1),
      'ToTokenSequence_0/embedding/kernel':
          vessa['patch_embed.proj.weight'].transpose(2, 3, 1, 0),  # OIHW -> HWIO
      'ToTokenSequence_0/embedding/bias': vessa['patch_embed.proj.bias'],

      'encoder_norm/scale': vessa.get('norm.weight', vessa.get('fc_norm.weight')),
      'encoder_norm/bias': vessa.get('norm.bias', vessa.get('fc_norm.weight')),
  }
  try:
    return easy_mappings[scenic_name]
  except KeyError:
    pass

  # Enumerated blocks.
  assert scenic_name.startswith('encoderblock_'), f'whats this? {scenic_name}'
  iblock = int(re.compile(r'encoderblock_(\d+)').match(scenic_name).group(1))

  prefix = f'encoderblock_{iblock}/'
  pt_prefix = f'blocks.{iblock}.'

  block = {
      prefix + 'LayerNorm_0/scale': vessa[pt_prefix + 'norm1.weight'],
      prefix + 'LayerNorm_0/bias': vessa[pt_prefix + 'norm1.bias'],
      prefix + 'MultiHeadDotProductAttention_0/query/kernel':
          split_qkv_w(vessa[pt_prefix + 'attn.qkv.weight'], nheads)[0],
      prefix + 'MultiHeadDotProductAttention_0/key/kernel':
          split_qkv_w(vessa[pt_prefix + 'attn.qkv.weight'], nheads)[1],
      prefix + 'MultiHeadDotProductAttention_0/value/kernel':
          split_qkv_w(vessa[pt_prefix + 'attn.qkv.weight'], nheads)[2],
      prefix + 'MultiHeadDotProductAttention_0/out/kernel':
          vessa[pt_prefix + 'attn.proj.weight'].T.reshape(
              nheads, -1, vessa[pt_prefix + 'attn.proj.weight'].shape[-1]),
      prefix + 'MultiHeadDotProductAttention_0/out/bias':
          vessa[pt_prefix + 'attn.proj.bias'],
      prefix + 'LayerNorm_1/scale': vessa[pt_prefix + 'norm2.weight'],
      prefix + 'LayerNorm_1/bias': vessa[pt_prefix + 'norm2.bias'],
      prefix + 'MlpBlock_0/Dense_0/kernel':
          vessa[pt_prefix + 'mlp.fc1.weight'].T,
      prefix + 'MlpBlock_0/Dense_0/bias': vessa[pt_prefix + 'mlp.fc1.bias'],
      prefix + 'MlpBlock_0/Dense_1/kernel':
          vessa[pt_prefix + 'mlp.fc2.weight'].T,
      prefix + 'MlpBlock_0/Dense_1/bias': vessa[pt_prefix + 'mlp.fc2.bias'],
  }

  if pt_prefix + 'attn.qkv.bias' in vessa:
    block.update({
        prefix + 'MultiHeadDotProductAttention_0/key/bias': split_qkv_b(
            vessa[pt_prefix + 'attn.qkv.bias'], nheads)[1],
        prefix + 'MultiHeadDotProductAttention_0/query/bias':
            split_qkv_b(vessa[pt_prefix + 'attn.qkv.bias'], nheads)[0],
        prefix + 'MultiHeadDotProductAttention_0/value/bias':
            split_qkv_b(vessa[pt_prefix + 'attn.qkv.bias'], nheads)[2],
    })

  if pt_prefix + 'ls1.gamma' in vessa:
    ls1 = vessa[pt_prefix + 'ls1.gamma']
    ls2 = vessa[pt_prefix + 'ls2.gamma']
    block.update({prefix + 'gamma_1': ls1, prefix + 'gamma_2': ls2})
    if absorbe_ls:
      k_key = prefix + 'MultiHeadDotProductAttention_0/out/kernel'
      b_key = prefix + 'MultiHeadDotProductAttention_0/out/bias'
      block[k_key] = block[k_key] * ls1
      block[b_key] = block[b_key] * ls1
      k_key = prefix + 'MlpBlock_0/Dense_1/kernel'
      b_key = prefix + 'MlpBlock_0/Dense_1/bias'
      block[k_key] = block[k_key] * ls2
      block[b_key] = block[b_key] * ls2
  return block[scenic_name]


def get_param_from_deit(deit, scenic_name, nheads, absorbe_ls=True):
  """Converts key name from deit-3 official weights to this codebase name."""
  easy_mappings = {}
  n = int((deit['pos_embed'].shape[1])**0.5)
  d = deit['patch_embed.proj.bias'].shape[0]
  if np.prod(deit['pos_embed'].shape) != n * n * d:
    easy_mappings.update({
        'ToTokenSequence_0/cls': deit['cls_token'] + deit['pos_embed'][:, 0, :],
        'ToTokenSequence_0/posembed_input':
            deit['pos_embed'][:, 1:, :].reshape(1, n, n, -1)})
  else:
    easy_mappings.update({
        'ToTokenSequence_0/cls': deit['cls_token'],
        'ToTokenSequence_0/posembed_input': deit['pos_embed'].reshape(1, n,
                                                                      n, -1)})

  easy_mappings.update({
      'ToTokenSequence_0/embedding/kernel':
          deit['patch_embed.proj.weight'].transpose(2, 3, 1, 0),  # OIHW -> HWIO
      'ToTokenSequence_0/embedding/bias': deit['patch_embed.proj.bias'],
      'encoder_norm/scale': deit['norm.weight'],
      'encoder_norm/bias': deit['norm.bias']})
  if 'head.weight' in deit:
    easy_mappings.update({'pixel_classif/kernel': deit['head.weight'].T,
                          'pixel_classif/bias': deit['head.bias']})

  try:
    return easy_mappings[scenic_name]
  except KeyError:
    pass

  # Enumerated blocks.
  assert scenic_name.startswith('encoderblock_'), f'whats this? {scenic_name}'
  iblock = int(re.compile(r'encoderblock_(\d+)').match(scenic_name).group(1))

  prefix = f'encoderblock_{iblock}/'
  pt_prefix = f'blocks.{iblock}.'

  block = {
      prefix + 'LayerNorm_0/scale': deit[pt_prefix + 'norm1.weight'],
      prefix + 'LayerNorm_0/bias': deit[pt_prefix + 'norm1.bias'],
      prefix + 'MultiHeadDotProductAttention_0/query/kernel':
          split_qkv_w(deit[pt_prefix + 'attn.qkv.weight'], nheads)[0],
      prefix + 'MultiHeadDotProductAttention_0/query/bias':
          split_qkv_b(deit[pt_prefix + 'attn.qkv.bias'], nheads)[0],
      prefix + 'MultiHeadDotProductAttention_0/key/kernel':
          split_qkv_w(deit[pt_prefix + 'attn.qkv.weight'], nheads)[1],
      prefix + 'MultiHeadDotProductAttention_0/key/bias':
          split_qkv_b(deit[pt_prefix + 'attn.qkv.bias'], nheads)[1],
      prefix + 'MultiHeadDotProductAttention_0/value/kernel':
          split_qkv_w(deit[pt_prefix + 'attn.qkv.weight'], nheads)[2],
      prefix + 'MultiHeadDotProductAttention_0/value/bias':
          split_qkv_b(deit[pt_prefix + 'attn.qkv.bias'], nheads)[2],
      prefix + 'MultiHeadDotProductAttention_0/out/kernel':
          deit[pt_prefix + 'attn.proj.weight'].T.reshape(
              nheads, -1, deit[pt_prefix + 'attn.proj.weight'].shape[-1]),
      prefix + 'MultiHeadDotProductAttention_0/out/bias':
          deit[pt_prefix + 'attn.proj.bias'],
      prefix + 'LayerNorm_1/scale': deit[pt_prefix + 'norm2.weight'],
      prefix + 'LayerNorm_1/bias': deit[pt_prefix + 'norm2.bias'],
      prefix + 'MlpBlock_0/Dense_0/kernel':
          deit[pt_prefix + 'mlp.fc1.weight'].T,
      prefix + 'MlpBlock_0/Dense_0/bias': deit[pt_prefix + 'mlp.fc1.bias'],
      prefix + 'MlpBlock_0/Dense_1/kernel':
          deit[pt_prefix + 'mlp.fc2.weight'].T,
      prefix + 'MlpBlock_0/Dense_1/bias': deit[pt_prefix + 'mlp.fc2.bias'],
  }
  if pt_prefix + 'gamma_1' in deit:
    ls1 = deit[pt_prefix + 'gamma_1']
    ls2 = deit[pt_prefix + 'gamma_2']
    block.update({prefix + 'gamma_1': ls1, prefix + 'gamma_2': ls2})
    if absorbe_ls:
      k_key = prefix + 'MultiHeadDotProductAttention_0/out/kernel'
      b_key = prefix + 'MultiHeadDotProductAttention_0/out/bias'
      block[k_key] = block[k_key] * ls1
      block[b_key] = block[b_key] * ls1
      k_key = prefix + 'MlpBlock_0/Dense_1/kernel'
      b_key = prefix + 'MlpBlock_0/Dense_1/bias'
      block[k_key] = block[k_key] * ls2
      block[b_key] = block[b_key] * ls2
  return block[scenic_name]


def unflatten_dict(flattened: Dict[str, Any],
                   separator: str = '/',
                   leaf_idx: int = -1) -> Dict[str, Any]:
  """Convert dict."""
  unflattened = {}
  for k, v in flattened.items():
    subtree = unflattened
    if leaf_idx != 0:
      path = k.split(separator)[:leaf_idx]
    else:
      path = k.split(separator)
    for k2 in path[:-1]:
      if k2 not in subtree:
        subtree[k2] = {}
      subtree = subtree[k2]
    subtree[path[-1]] = v
  return unflattened


def load_augreg(checkpoint_path: str, params: Any) -> Any:
  """Load a supervised ViT checkpoint from the AugReg paper."""
  restored_params = None
  logging.info('Loading big_vision checkpoint from %s', checkpoint_path)
  checkpoint_data = np.load(gfile.GFile(checkpoint_path, 'rb'))
  tree = unflatten_dict(checkpoint_data, separator='/', leaf_idx=0)
  restored_params, name = {}, ''
  for name in tree['Transformer'].keys():
    if 'posembed_input' in name:
      continue
    restored_params[name] = tree['Transformer'][name]
  to_token = {}
  to_token['embedding'] = {}
  to_token['embedding']['bias'] = tree['embedding']['bias']
  to_token['embedding']['kernel'] = tree['embedding']['kernel']
  pos = tree['Transformer']['posembed_input']['pos_embedding']
  to_token['cls'] = tree['cls'] + pos[None, :, 0]
  p = np.sqrt(pos[:, 1:].shape[1]).astype(np.int32)
  assert p == np.sqrt(pos[:, 1:].shape[1]), '#patches should be a square number'
  pos_patches = pos[:, 1:].reshape(1, p, p, pos.shape[-1])  # Important: pos_patches.shape[-1]==pos.shape[-1] should hold! pylint: disable=line-too-long
  target_shape = params['ToTokenSequence_0']['posembed_input'].shape
  if pos_patches.shape != target_shape:
    logging.info('Resampling patch positional embedding to %s', target_shape)
    pos_patches = jax.image.resize(pos_patches, target_shape, 'bicubic')
  to_token['posembed_input'] = pos_patches
  restored_params['ToTokenSequence_0'] = to_token
  restored_params = checkpoints.convert_pre_linen(restored_params)
  restored_params = dict(restored_params)
  restored_params = pretrain_utils.inspect_params(
      expected_params=params,
      restored_params=restored_params,
      fail_if_extra=True,
      fail_if_missing=False,
      fail_if_shapes_mismatch=True)
  return restored_params


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  print(fname)
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # For other (remote) paths go via gfile+BytesIO as np.load requires seeks.
    with gfile.Open(fname, "rb") as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)

def _traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree_util.tree_map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, (list, tuple)):
    for idx in range(len(tree)):
      for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree

def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree_util.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def

def tree_map_with_names(f, tree, *rest):
  """Like jax.tree_util.tree_map but with a filter on the leaf path name.

  Args:
    f: A function with first parameter `name` (path-like "a/b/c") and remaining
      parameters values of `tree` and `*rest` corresponding to the given `name`
      Should return a new value for parameter `name`.
    tree: The tree of parameters `f` should be applied to.
    *rest: more trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves the
    result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
  """
  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)

def restore_params(checkpoint_name: str, checkpoint_path: str, params: Any,
                   params_key: str = 'teacher_weights',) -> Any:
  """Returns restored params."""
  restored_params = None
  # Option 1: We load weights from a pytorch like dict.
  if checkpoint_name in PYTORCH_STYLE_WEIGHTS.keys():
    params_key = ''
    checkpoint_path = PYTORCH_STYLE_WEIGHTS[checkpoint_name]
    logging.info('Loading pytorch style weights from %s', checkpoint_path)
    pt_weights = npload(checkpoint_path)
    # Pytorch param keys might have extra prefix 'module.'.
    pt_weights = {k.replace('module.', ''): v for k, v in pt_weights.items()}
    # Number of head.
    nh = params['encoderblock_0']['MultiHeadDotProductAttention_0']['key']['kernel'].shape[1]  # pylint: disable=line-too-long
    get_param = get_param_from_pytorch_style_dict
    if checkpoint_name.startswith('deit'):
      get_param = get_param_from_deit
    if checkpoint_name.startswith('beitv2'):
      # We load only the encoder.
      pt_weights = {k.replace('encoder.', ''): v for (
          k, v) in pt_weights.items() if k.startswith('encoder.')}
    def dontload(key):
      #Adding Lora
      if 'LoRA' in key or 'lora' in key:
        return True
      if 'projecti' in key or 'UperNet' in key or 'fpn' in key:
        return True
      if key.startswith('pixel_') or key.startswith('Dense_0'):
        return True
      return False
    restored_params = tree_map_with_names(
        lambda k, v: None if dontload(k) else get_param(pt_weights, k, nh),
        params)

  # In the remaining of this function we do param surgery.
  restored_params = flax.core.unfreeze(restored_params)
  if params_key in restored_params:
    logging.info('Taking "%s" in restored checkpoint dict...', params_key)
    restored_params = restored_params[params_key]

  if 'Transformer' in restored_params:
    logging.info('There is key `Transformer` in `restored_params`.')
    pos_emb, cls_token = None, None
    if 'cls' in restored_params:
      cls_token = restored_params['cls']
    if 'posembed_input' in restored_params:
      pos_emb = restored_params['posembed_input']['pos_embedding']
    elif 'pos_embedding' in restored_params:
      pos_emb = restored_params['pos_embedding']

    restored_params = {**restored_params['Transformer'],
                       'embedding': restored_params['embedding']}
    if pos_emb is not None:
      if pos_emb.shape[1] == 197:
        pos_emb = pos_emb[:, 1:]
        cls_token = cls_token + pos_emb[:, 0]
      n = int(pos_emb.shape[1]**0.5)
      pos_emb = pos_emb.reshape(1, n, n, pos_emb.shape[-1])
      restored_params['posembed_input'] = pos_emb
    if cls_token is not None:
      restored_params['cls'] = cls_token

  # Backward compatibility.
  tks = 'ToTokenSequence_0'
  if tks not in restored_params:
    logging.info('Backward compatibility.')
    keys = ['cls', 'embedding', 'posembed_input']
    to_token = {k: restored_params[k] for k in keys if k in restored_params}
    restored_params[tks] = to_token
    for k in keys:
      if k in restored_params: del restored_params[k]

  # Resampling of patch positional embeddings.
  target_shape = params[tks]['posembed_input'].shape
  if restored_params[tks]['posembed_input'].shape != target_shape:
    logging.info('Resampling patch positional embedding to %s', target_shape)
    pos_patches = jax.image.resize(restored_params[tks]['posembed_input'],
                                   target_shape, 'bilinear')
    restored_params[tks]['posembed_input'] = pos_patches
  return restored_params


def load_params(checkpoint_name: str, checkpoint_path: str, params: Any,
                params_key: str = 'teacher_weights',
                force_random_init: Optional[str] = None) -> Any:
  """Load parameters.

  Args:
    checkpoint_name: Name of the checkpoint.
    checkpoint_path: Path to checkpoint.
    params: A PyTree containing the random parameters.
    params_key: What key to fetch in restored checkpoint dict.
    force_random_init: We randomly initialized even if the param is in restored.

  Returns:
    restored parameters: PyTree
  """
  restored_params = restore_params(checkpoint_name, checkpoint_path, params,
                                   params_key)
  if restored_params is None:
    logging.info('Loading weights failed -> we use random weights...')
    return params

  # Flatten the tree.
  restored_params = flax.traverse_util.flatten_dict(restored_params, sep='/')
  params = flax.traverse_util.flatten_dict(params, sep='/')

  # Random init of the params not found in the restored dict.
  for k, v in params.items():
    if k not in restored_params or restored_params[k] is None:
      logging.info('%s not found in restored param -> random init.', k)
      restored_params[k] = v
    elif force_random_init and force_random_init in k:
      logging.info('We use random init for %s even if present in restored.', k)
      restored_params[k] = v
    else:
      logging.info('%s loaded from restored param.', k)

  # Removes unused loaded params.
  to_remove = []
  for k in restored_params:
    if k not in params:
      to_remove.append(k)
  for k in to_remove:
    logging.info('Removing unused %s from the restored params.', k)
    del restored_params[k]

  # Unflatten the tree.
  restored_params = flax.traverse_util.unflatten_dict(restored_params, sep='/')
  restored_params = flax.core.freeze(restored_params)
  return restored_params


# pylint: disable=line-too-long
PYTORCH_STYLE_WEIGHTS = {
    #'vessav2_vits14': '/home/jesimon/Documentos/mestrado/vessav2_vits14.npz',
    #'vessav2_vitb14': '/home/jesimon/Documentos/mestrado/vessav2_vitb14.npz',
    'vessav2_vits14': '/home/jesimonbarreto/vessav2_vits14.npz',
    'vessav2_vitb14': '/home/jesimonbarreto/vessav2_vitb14.npz',
    'vessav2_vitl14': '/home/jesimonbarreto/vessav2_vitl14.npz',
    'vessa_vitb16': '/home/jesimonbarreto/vessa_vitb16.npz',
    'vessa_vitdeits16': '/home/jesimonbarreto/vessa_deits16.npz',
    'tips_b14': '/home/jesimonbarreto/tips_b14.npz',
}
