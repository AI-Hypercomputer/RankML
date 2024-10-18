"""
Copyright 2024 Google LLC

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

from typing import Sequence, List
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


class MLP(nn.Module):
    """Multi-layer perceptron."""

    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        return x


class DenseArch(nn.Module):
    """Dense features architecture."""

    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, dense_features):
        return MLP(self.layer_sizes)(dense_features)


class EmbeddingArch(nn.Module):
    """Embedding architecture"""

    vocab_sizes: List[int]
    embedding_dim: int

    @nn.compact
    def __call__(self, embedding_ids):
        embeddings = []
        for i, vocab_size in enumerate(self.vocab_sizes):
            embedding_table = self.param(
                f"embedding_{i}",
                nn.initializers.uniform(),
                (vocab_size, self.embedding_dim),
            )
            embedding = jnp.take(embedding_table, embedding_ids[:, i], axis=0)
            embeddings.append(embedding)
        return embeddings


class InteractionArch(nn.Module):
    """Base interaction architecture."""

    @nn.compact
    def __call__(self, dense_output, embedding_outputs):
        return jnp.concatenate([dense_output] + embedding_outputs, axis=1)


class DotInteractionArch(InteractionArch):
    """Dot product interaction architecture."""

    @nn.compact
    def __call__(self, dense_output, embedding_outputs):
        combined_values = jnp.concatenate(
            [dense_output.reshape(dense_output.shape[0], 1, -1)]
            + [e.reshape(e.shape[0], 1, -1) for e in embedding_outputs],
            axis=1,
        )

        interactions = jnp.matmul(combined_values, combined_values.transpose((0, 2, 1)))

        num_features = combined_values.shape[1]
        triu_indices = jnp.triu_indices(num_features, num_features, k=1)

        interactions_flat = interactions[:, triu_indices[0], triu_indices[1]]

        return jnp.concatenate([dense_output, interactions_flat], axis=1)


class LowRankCrossNetInteractionArch(InteractionArch):
    """Low Rank Cross Network interaction architecture."""

    num_layers: int
    low_rank: int

    @nn.compact
    def __call__(self, dense_output, embedding_outputs):
        base_output = jnp.concatenate([dense_output] + embedding_outputs, axis=1)

        x_0 = base_output
        x_l = x_0
        in_features = x_0.shape[-1]

        for layer in range(self.num_layers):
            W = self.param(
                f"W_{layer}",
                nn.initializers.glorot_uniform(),
                (in_features, self.low_rank),
            )
            V = self.param(
                f"V_{layer}",
                nn.initializers.glorot_uniform(),
                (self.low_rank, in_features),
            )
            b = self.param(f"b_{layer}", nn.initializers.zeros, (in_features,))

            x_l_v = jnp.matmul(x_l, V.T)
            x_l_w = jnp.matmul(x_l_v, W.T)
            x_l = x_0 * (x_l_w + b) + x_l

        return x_l


class OverArch(nn.Module):
    """Over-architecture (top MLP)."""

    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = MLP(self.layer_sizes)(x)
        return nn.Dense(features=1)(x)
