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
from layers import MLP, DenseArch, EmbeddingArch, InteractionArch, OverArch

class DLRMV2(nn.Module):
    """DLRM V2 model."""
    vocab_sizes: List[int]
    embedding_dim: int
    bottom_mlp_dims: List[int]
    top_mlp_dims: List[int]

    @nn.compact
    def __call__(self, dense_features, embedding_ids):
        # Bottom MLP
        x = self.bottom_mlp(dense_features)

        # Embedding layer
        embeddings = []
        for i, vocab_size in enumerate(self.vocab_sizes):
            embedding = nn.Embed(vocab_size, self.embedding_dim)(embedding_ids[str(i)])
            embeddings.append(embedding)
        
        # Flatten and concatenate embeddings
        embedding_output = jnp.concatenate([e.reshape(-1, self.embedding_dim) for e in embeddings], axis=1)

        # Concatenate bottom MLP output and embedding output
        concatenated = jnp.concatenate([x, embedding_output], axis=1)

        # Top MLP
        y = self.top_mlp(concatenated)

        return y.squeeze(-1)

    def bottom_mlp(self, x):
        for dim in self.bottom_mlp_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return x

    def top_mlp(self, x):
        for dim in self.top_mlp_dims[:-1]:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.top_mlp_dims[-1])(x)
        return x