# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def auc(logits, labels):
    """Calculates the Area Under the ROC Curve (AUC)."""
    predictions = jax.nn.sigmoid(logits)
    sorted_indices = jnp.argsort(predictions)
    sorted_labels = labels[sorted_indices]
    tpr = jnp.cumsum(sorted_labels) / jnp.sum(sorted_labels)
    fpr = jnp.cumsum(1 - sorted_labels) / jnp.sum(1 - sorted_labels)
    return jnp.trapz(tpr, fpr)

def ndcg(logits, labels, k=10):
    """Calculates the Normalized Discounted Cumulative Gain (NDCG) at rank k."""
    predictions = jnp.argsort(-logits)[:k]
    relevance = labels[predictions]
    ideal_relevance = jnp.sort(labels)[::-1][:k]
    dcg = jnp.sum(relevance / jnp.log2(jnp.arange(2, len(relevance) + 2)))
    idcg = jnp.sum(ideal_relevance / jnp.log2(jnp.arange(2, len(ideal_relevance) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(logits, labels, k=10):
    """Calculates the Precision at rank k."""
    top_k_indices = jnp.argsort(-logits)[:k]
    return jnp.mean(labels[top_k_indices])

@jax.jit
def compute_metrics(logits, labels):
    """Computes all metrics at once."""
    return {
        'accuracy': accuracy(logits, labels),
        'auc': auc(logits, labels),
        'ndcg@10': ndcg(logits, labels, k=10),
        'precision@10': precision_at_k(logits, labels, k=10)
    }

