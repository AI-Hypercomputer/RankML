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

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from models import DLRMV2
from configs import get_config, get_criteo_config
from losses import bce_with_logits_loss
from metrics import accuracy, compute_metrics
import ml_collections
from data_pipeline import train_input_fn, eval_input_fn
import tensorflow as tf
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import time


def create_train_state(rng, config, mesh):
    """Creates initial `TrainState` with sharding."""
    dlrm = DLRMV2(
        vocab_sizes=config.model.vocab_sizes,
        embedding_dim=config.model.embedding_dim,
        bottom_mlp_dims=config.model.bottom_mlp_dims,
        top_mlp_dims=config.model.top_mlp_dims,
    )

    dummy_dense = jnp.ones([1, config.model.num_dense_features])
    dummy_sparse = {
        str(i): jnp.ones([1], dtype=jnp.int32)
        for i in range(len(config.model.vocab_sizes))
    }

    params = dlrm.init(rng, dummy_dense, dummy_sparse)["params"]
    tx = optax.adam(config.model.learning_rate)

    return train_state.TrainState.create(
        apply_fn=dlrm.apply,
        params=jax.tree.map(
            lambda x: jax.device_put(x, NamedSharding(mesh, P())), params
        ),
        tx=tx,
    )


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, batch["dense_features"], batch["sparse_features"]
        )
        loss = bce_with_logits_loss(logits, batch["labels"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch["labels"])
    metrics["loss"] = loss
    return state, metrics


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(
        {"params": state.params}, batch["dense_features"], batch["sparse_features"]
    )
    loss = bce_with_logits_loss(logits, batch["labels"])
    metrics = compute_metrics(logits, batch["labels"])
    metrics["loss"] = loss
    return metrics


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("batch",))

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    with mesh:
        state = create_train_state(init_rng, config, mesh)

        train_ds = train_input_fn(config)
        eval_ds = eval_input_fn(config)

        batch_sharding = NamedSharding(
            mesh,
            P(
                "batch",
            ),
        )

        print("Start training")
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = []
            for features, labels in train_ds.take(config.steps_per_epoch):
                batch = {
                    "dense_features": jax.device_put(
                        np.array(features["dense_features"]), batch_sharding
                    ),
                    "sparse_features": jax.tree.map(
                        lambda x: jax.device_put(np.array(x), batch_sharding),
                        features["sparse_features"],
                    ),
                    "labels": jax.device_put(np.array(labels), batch_sharding),
                }

                # jax.debug.visualize_array_sharding(batch['labels'])

                state, metrics = train_step(state, batch)
                train_metrics.append(metrics)

            train_metrics = jax.tree.map(
                lambda *args: jnp.mean(jnp.array(args)), *train_metrics
            )

            print(f"Epoch {epoch}:")
            print(
                f'  Train loss: {train_metrics["loss"]:.4f}, accuracy: {train_metrics["accuracy"]:.4f}, auc: {train_metrics["auc"]: .4f}'
            )

    return state


if __name__ == "__main__":
    # config = get_config()
    config = get_criteo_config()
    train_and_evaluate(config, "/tmp/dlrm_v2")
