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

from typing import Sequence
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


def accuracy(logits, labels):
  """Calculates the accuracy of predictions."""
  predictions = jax.nn.sigmoid(logits) > 0.5
  return jnp.mean(predictions == labels)


@jax.jit
def compute_metrics(logits, labels):
  """Computes all metrics at once."""
  return {
      'accuracy': accuracy(logits, labels),
  }
