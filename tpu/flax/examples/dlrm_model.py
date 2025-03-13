"""Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List

from flax import linen as nn
import jax
import jax.numpy as jnp

from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested


def uniform_init(bound: float):
  def init(key, shape, dtype=jnp.float_):
    return jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=-bound,
        maxval=bound
    )
  return init


class DLRMDCNV2(nn.Module):
  """DLRM DCN v2 model."""
  feature_specs: Nested[embedding_spec.FeatureSpec]
  mesh: jax.sharding.Mesh
  sharding_axis: str
  global_batch_size: int
  embedding_size: int
  bottom_mlp_dims: List[int]
  top_mlp_dims = [1024, 1024, 512, 256, 1]
  dcn_layers: int = 3
  projection_dim: int = 512

  def bottom_mlp(self, x):
    for dim in self.bottom_mlp_dims:
      previous_dim = x.shape[-1]
      bound = jnp.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
    return x

  def top_mlp(self, x):
    previous_dim = x.shape[-1]
    for dim in self.top_mlp_dims[:-1]:
      bound = jnp.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
      previous_dim = dim

    bound = jnp.sqrt(1.0 / previous_dim)
    x = nn.Dense(
        self.top_mlp_dims[-1],
        kernel_init=uniform_init(bound),
        bias_init=uniform_init(bound),
    )(x)
    x = nn.sigmoid(x)
    return x

  def dcn_layer(self, x0):
    xl = x0
    input_dim = x0.shape[-1]

    for i in range(self.dcn_layers):
      u_kernel = self.param(
          f'u_kernel_{i}',
          nn.initializers.xavier_normal(),
          (input_dim, self.projection_dim),
      )
      v_kernel = self.param(
          f'v_kernel_{i}',
          nn.initializers.xavier_normal(),
          (self.projection_dim, input_dim),
      )
      bias = self.param(f'bias_{i}', nn.initializers.zeros, (input_dim,))

      u_output = jnp.matmul(xl, u_kernel)
      v_output = jnp.matmul(u_output, v_kernel)
      v_output += bias

      xl = x0 * v_output + xl

    return xl

  @nn.compact
  def __call__(
      self, dense_features, embedding_lookups
  ):
    dense_outputs = self.bottom_mlp(dense_features)
    sparse_embeddings = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
    )(embedding_lookups)
    sparse_embeddings = jax.tree.flatten(sparse_embeddings)
    concatenated_embeddings = jnp.concatenate(sparse_embeddings[0], axis=1)
    # Concatenate dense features and embeddings. We're using global batch size
    # here because we're doing global view of the data in the training loop.
    interaction_args = jax.lax.concatenate(
        [
            dense_outputs.reshape(
                (self.global_batch_size, 1, self.embedding_size)
            ),
            concatenated_embeddings.reshape(
                (self.global_batch_size, 26, self.embedding_size)
            ),
        ],
        dimension=1,
    )
    interaction_args = interaction_args.reshape((self.global_batch_size, -1))
    interaction_outputs = self.dcn_layer(interaction_args)
    predictions = self.top_mlp(interaction_outputs)
    predictions = jnp.reshape(predictions, (-1,))

    return predictions

