# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for manipulating coordinate spaces and distances along rays."""

import numpy as np
import torch
from functorch import jvp, vmap


def contract(x):
  """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
  eps = torch.as_tensor(torch.finfo(torch.float16).eps, device = x.device)
  # Clamping to eps prevents non-finite gradients when x == 0.
  x_mag_sq = torch.maximum(eps, torch.sum(x**2, axis=-1, keepdims=True))
  z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
  return z


def inv_contract(z):
  """The inverse of contract()."""
  eps = torch.as_tensor(torch.finfo(torch.float16).eps, device = x.device)
  # Clamping to eps prevents non-finite gradients when z == 0.
  z_mag_sq = torch.maximum(eps, torch.sum(z**2, axis=-1, keepdims=True))
  x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq))
  return x

"""def linearize(f, params):
  def f_lin(p, *args, **kwargs):
    dparams = _sub(p, params)
    f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                                                    (params,), (dparams,))
    return f_params_x + proj
  return f_lin

def _sub(x, y):
  return tuple(x - y for (x, y) in zip(x, y))"""

def track_linearize(fn, mean, cov):
  """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: the function applied to the Gaussians parameterized by (mean, cov).
    mean: a tensor of means, where the last axis is the dimension.
    cov: a tensor of covariances, where the last two axes are the dimensions.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
  if (len(mean.shape) + 1) != len(cov.shape):
    raise ValueError('cov must be non-diagonal')
  #fn_mean, lin_fn = jvp(fn, (mean,), (torch.ones_like(mean),))
  #fn_cov = vmap(lin_fn, -1, -2)(vmap(lin_fn, -1, -2)(cov))
  _, fn_mean = jvp(fn, (mean,), (torch.zeros_like(mean),))
  _, fn_cov = jvp(fn, (cov,), (torch.zeros_like(cov),))
  # fn_cov = vmap(lin_fn, -1, -2)(vmap(lin_fn, -1, -2)(cov))
  return fn_mean, fn_cov


def construct_ray_warps(fn, t_near, t_far):
  """Construct a bijection between metric distances and normalized distances.

  See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
  detailed explanation.

  Args:
    fn: the function to ray distances.
    t_near: a tensor of near-plane distances.
    t_far: a tensor of far-plane distances.

  Returns:
    t_to_s: a function that maps distances to normalized distances in [0, 1].
    s_to_t: the inverse of t_to_s.
  """
  if fn is None:
    fn_fwd = lambda x: x
    fn_inv = lambda x: x
  elif fn == 'piecewise':
    # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
    fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
    fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
  else:
    inv_mapping = {
        'reciprocal': torch.reciprocal,
        'log': torch.exp,
        'exp': torch.log,
        'sqrt': torch.square,
        'square': torch.sqrt
    }
    fn_fwd = fn
    fn_inv = inv_mapping[fn.__name__]

  s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
  t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
  s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
  return t_to_s, s_to_t

def safe_trig_helper(x, fn, t=100):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  t = t * torch.as_tensor(np.pi, device=x.device)
  return fn(torch.where(torch.abs(x) < t, x, x % t))

def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)

def expected_sin(mean, var):
  """Compute the mean of sin(x), x ~ N(mean, var)."""
  return torch.exp(-0.5 * var) * safe_sin(mean)  # large var -> small value.


def lift_and_diagonalize(mean, cov, basis):
  """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
  fn_mean = torch.matmul(mean, basis)
  fn_cov_diag = torch.sum(basis * torch.matmul(cov, basis), axis=-2)
  return fn_mean, fn_cov_diag

'''
def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2**torch.arange(min_deg, max_deg)
  shape = x.shape[:-1] + (-1,)
  scaled_x = torch.reshape((x[..., None, :] * scales[:, None]), shape)
  # Note that we're not using safe_sin, unlike IPE.
  four_feat = torch.sin(
      torch.concatenate([scaled_x, scaled_x + 0.5 * torch.as_tensor(np.pi, device=x.device)], axis=-1))
  if append_identity:
    return torch.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat

def integrated_pos_enc(mean, var, min_deg, max_deg):
  """Encode `x` with sinusoids scaled by 2^[min_deg, max_deg).

  Args:
    mean: tensor, the mean coordinates to be encoded
    var: tensor, the variance of the coordinates to be encoded.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  scales = 2**torch.arange(min_deg, max_deg, device = mean.device)
  shape = mean.shape[:-1] + (-1,)
  scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
  scaled_var = torch.reshape(var[..., None, :] * scales[:, None]**2, shape)

  return expected_sin(
      torch.concatenate([scaled_mean, scaled_mean + 0.5 * torch.as_tensor(np.pi, device=mean.device)], axis=-1),
      torch.concatenate([scaled_var] * 2, axis=-1))'''

  # Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for constructing geodesic polyhedron, which are used as a basis."""

import itertools
import numpy as np


def compute_sq_dist(mat0, mat1=None):
  """Compute the squared Euclidean distance between all pairs of columns."""
  if mat1 is None:
    mat1 = mat0
  # Use the fact that ||x - y||^2 == ||x||^2 + ||y||^2 - 2 x^T y.
  sq_norm0 = np.sum(mat0**2, 0)
  sq_norm1 = np.sum(mat1**2, 0)
  sq_dist = sq_norm0[:, None] + sq_norm1[None, :] - 2 * mat0.T @ mat1
  sq_dist = np.maximum(0, sq_dist)  # Negative values must be numerical errors.
  return sq_dist


def compute_tesselation_weights(v):
  """Tesselate the vertices of a triangle by a factor of `v`."""
  if v < 1:
    raise ValueError(f'v {v} must be >= 1')
  int_weights = []
  for i in range(v + 1):
    for j in range(v + 1 - i):
      int_weights.append((i, j, v - (i + j)))
  int_weights = np.array(int_weights)
  weights = int_weights / v  # Barycentric weights.
  return weights


def tesselate_geodesic(base_verts, base_faces, v, eps=1e-4):
  """Tesselate the vertices of a geodesic polyhedron.

  Args:
    base_verts: tensor of floats, the vertex coordinates of the geodesic.
    base_faces: tensor of ints, the indices of the vertices of base_verts that
      constitute eachface of the polyhedra.
    v: int, the factor of the tesselation (v==1 is a no-op).
    eps: float, a small value used to determine if two vertices are the same.

  Returns:
    verts: a tensor of floats, the coordinates of the tesselated vertices.
  """
  if not isinstance(v, int):
    raise ValueError(f'v {v} must an integer')
  tri_weights = compute_tesselation_weights(v)

  verts = []
  for base_face in base_faces:
    new_verts = np.matmul(tri_weights, base_verts[base_face, :])
    new_verts /= np.sqrt(np.sum(new_verts**2, 1, keepdims=True))
    verts.append(new_verts)
  verts = np.concatenate(verts, 0)

  sq_dist = compute_sq_dist(verts.T)
  assignment = np.array([np.min(np.argwhere(d <= eps)) for d in sq_dist])
  unique = np.unique(assignment)
  verts = verts[unique, :]

  return verts


def generate_basis(base_shape,
                   angular_tesselation,
                   remove_symmetries=True,
                   eps=1e-4):
  """Generates a 3D basis by tesselating a geometric polyhedron.

  Args:
    base_shape: string, the name of the starting polyhedron, must be either
      'icosahedron' or 'octahedron'.
    angular_tesselation: int, the number of times to tesselate the polyhedron,
      must be >= 1 (a value of 1 is a no-op to the polyhedron).
    remove_symmetries: bool, if True then remove the symmetric basis columns,
      which is usually a good idea because otherwise projections onto the basis
      will have redundant negative copies of each other.
    eps: float, a small number used to determine symmetries.

  Returns:
    basis: a matrix with shape [3, n].
  """
  if base_shape == 'icosahedron':
    a = (np.sqrt(5) + 1) / 2
    verts = np.array([(-1, 0, a), (1, 0, a), (-1, 0, -a), (1, 0, -a), (0, a, 1),
                      (0, a, -1), (0, -a, 1), (0, -a, -1), (a, 1, 0),
                      (-a, 1, 0), (a, -1, 0), (-a, -1, 0)]) / np.sqrt(a + 2)
    faces = np.array([(0, 4, 1), (0, 9, 4), (9, 5, 4), (4, 5, 8), (4, 8, 1),
                      (8, 10, 1), (8, 3, 10), (5, 3, 8), (5, 2, 3), (2, 7, 3),
                      (7, 10, 3), (7, 6, 10), (7, 11, 6), (11, 0, 6), (0, 1, 6),
                      (6, 1, 10), (9, 0, 11), (9, 11, 2), (9, 2, 5),
                      (7, 2, 11)])
    verts = tesselate_geodesic(verts, faces, angular_tesselation)
  elif base_shape == 'octahedron':
    verts = np.array([(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0),
                      (1, 0, 0)])
    corners = np.array(list(itertools.product([-1, 1], repeat=3)))
    pairs = np.argwhere(compute_sq_dist(corners.T, verts.T) == 2)
    faces = np.sort(np.reshape(pairs[:, 1], [3, -1]).T, 1)
    verts = tesselate_geodesic(verts, faces, angular_tesselation)
  else:
    raise ValueError(f'base_shape {base_shape} not supported')

  if remove_symmetries:
    # Remove elements of `verts` that are reflections of each other.
    match = compute_sq_dist(verts.T, -verts.T) < eps
    verts = verts[np.any(np.triu(match), 1), :]

  basis = verts[:, ::-1].T #changed from source
  basis = np.ascontiguousarray(basis) #changed from source
  return basis
