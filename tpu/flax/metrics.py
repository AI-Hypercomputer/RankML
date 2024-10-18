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
from jax.scipy.integrate import trapezoid

def accuracy(logits, labels):
    """Calculates the accuracy of predictions."""
    predictions = jax.nn.sigmoid(logits) > 0.5
    return jnp.mean(predictions == labels)


def auc(logits, labels):
    """Calculates the Area Under the Receiver Operating Characteristic Curve (AUC)."""
    # Sort the data by predicted probabilities in descending order
    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Calculate the cumulative sum of positive and negative labels
    cumsum_pos = jnp.cumsum(sorted_labels)
    cumsum_neg = jnp.cumsum(1 - sorted_labels)

    # Calculate the total number of positive and negative labels
    total_pos = jnp.sum(sorted_labels)
    total_neg = len(sorted_labels) - total_pos

    # Calculate the AUC using the trapezoidal rule
    auc_score = trapezoid(cumsum_pos / total_pos, cumsum_neg / total_neg)

    return auc_score

@jax.jit
def compute_metrics(logits, labels):
    """Computes all metrics at once."""
    return {
        "accuracy": accuracy(logits, labels),
        "AUC": auc(logits, labels),
    }
