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

"""DLRM V2 Recommendation example.

Library file which executes the training and evaluation loop for DLRM V2.
The data is generated as fake input data.
"""

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from models import DLRMV2
from configs import get_config
from losses import bce_with_logits_loss
from metrics import accuracy
import ml_collections
from data_pipeline import train_input_fn, eval_input_fn
import tensorflow as tf

@jax.jit
def apply_model(state, dense_features, sparse_features, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, dense_features, sparse_features)
        loss = bce_with_logits_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    acc = accuracy(logits, labels)
    return grads, loss, acc

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    dlrm = DLRMV2(
        vocab_sizes=config.model.vocab_sizes,
        embedding_dim=config.model.embedding_dim,
        bottom_mlp_dims=config.model.bottom_mlp_dims,
        top_mlp_dims=config.model.top_mlp_dims
    )
    
    # Create dummy inputs for initialization
    dummy_dense = jnp.ones([1, config.model.num_dense_features])
    dummy_sparse = {str(i): jnp.ones([1], dtype=jnp.int32) for i in range(len(config.model.vocab_sizes))}
    
    params = dlrm.init(rng, dummy_dense, dummy_sparse)['params']
    tx = optax.adam(config.model.learning_rate)
    return train_state.TrainState.create(apply_fn=dlrm.apply, params=params, tx=tx)

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop."""
    train_ds = train_input_fn(config)
    # test_ds = eval_input_fn(config)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    print('start training')
    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        
        # Train loop
        epoch_loss = []
        epoch_accuracy = []
        for features, labels in train_ds.take(config.steps_per_epoch):  # Add this line\
            dense_features = jnp.array(features['dense_features'])
            sparse_features = {k: jnp.array(v) for k, v in features['sparse_features'].items()}
            labels = jnp.array(labels)
            grads, loss, accuracy = apply_model(state, dense_features, sparse_features, labels)
            state = update_model(state, grads)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        
        train_loss = jnp.mean(jnp.array(epoch_loss))
        train_accuracy = jnp.mean(jnp.array(epoch_accuracy))

        # # Evaluation loop
        # test_loss = []
        # test_accuracy = []
        # for features, labels in test_ds:
        #     dense_features = jnp.array(features['dense_features'])
        #     sparse_features = {k: jnp.array(v) for k, v in features['sparse_features'].items()}
        #     labels = jnp.array(labels)
        #     _, loss, accuracy = apply_model(state, dense_features, sparse_features, labels)
        #     test_loss.append(loss)
        #     test_accuracy.append(accuracy)
        
        # test_loss = jnp.mean(jnp.array(test_loss))
        # test_accuracy = jnp.mean(jnp.array(test_accuracy))

        # logging.info(
        #     'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        #     % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        # )
        
        # print('epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        #     % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))    
        print('epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100))    

    return state

if __name__ == "__main__":
    config = get_config()
    train_and_evaluate(config, '/tmp/dlrm_v2')
