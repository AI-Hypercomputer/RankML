"""Copyright 2024 Google LLC

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

import os

os.environ['KERAS_BACKEND'] = 'jax'

import jax
import jax.numpy as jnp
import keras


def DLRM(config):
  # Extract model parameters from the configuration
  num_dense_features = config.model.num_dense_features
  vocab_sizes = config.model.vocab_sizes
  embedding_dim = config.model.embedding_dim
  bottom_mlp_units = config.model.bottom_mlp_dims
  top_mlp_units = config.model.top_mlp_dims

  num_sparse_features = len(vocab_sizes)

  # Inputs
  dense_input = keras.Input(shape=(num_dense_features,), name='dense_features')
  sparse_inputs = [
      keras.Input(shape=(), dtype='int32', name=f'sparse_feature_{i}')
      for i in range(num_sparse_features)
  ]

  # Bottom MLP for dense features
  x = dense_input
  for units in bottom_mlp_units:
    x = keras.layers.Dense(units, activation='relu')(x)
  dense_embedding = x  # Shape: (batch_size, embedding_dim)

  # Embedding layers for sparse features
  sparse_embeddings = []
  for i in range(num_sparse_features):
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_sizes[i],
        output_dim=embedding_dim,
        name=f'embedding_{i}',
    )
    sparse_embedding = embedding_layer(
        sparse_inputs[i]
    )  # Shape: (batch_size, embedding_dim)
    sparse_embeddings.append(sparse_embedding)

  # Interactions between dense and sparse embeddings
  all_embeddings = [
      dense_embedding
  ] + sparse_embeddings  # List of tensors with shape (batch_size, embedding_dim)
  # Concatenate interactions
  x = keras.layers.Concatenate()(
      all_embeddings
  )  # Shape: (batch_size, num_interactions)

  # Top MLP
  for units in top_mlp_units[:-1]:
    x = keras.layers.Dense(units, activation='relu')(x)
  output = keras.layers.Dense(top_mlp_units[-1], activation='sigmoid')(x)

  model = keras.Model(inputs=[dense_input] + sparse_inputs, outputs=output)
  return model
