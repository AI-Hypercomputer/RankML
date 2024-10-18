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

os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf
import keras

from absl import app
from typing import Sequence
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Import the configuration and data pipeline modules
from configs import get_config
import data_pipeline
from models import DLRM


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = get_config()

  # Get training and validation datasets
  train_data = data_pipeline.train_input_fn(config)
  eval_data = data_pipeline.eval_input_fn(config)

  """
  ## Multi-Device Synchronous Training

  Now, we will set up the training loop to perform synchronous training across multiple devices using JAX sharding APIs.
  """

  # Configurations
  num_epochs = config.num_epochs
  batch_size = config.train_data.global_batch_size
  learning_rate = config.model.learning_rate

  model = DLRM(config)
  optimizer = keras.optimizers.Adam(learning_rate)
  loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

  # Initialize all state with .build()
  # Need to generate one batch of data to build the model
  (one_batch_inputs, one_batch_labels) = next(iter(train_data))
  model.build(one_batch_inputs)
  optimizer.build(model.trainable_variables)

  # This is the loss function that will be differentiated.
  def compute_loss(
      trainable_variables, non_trainable_variables, inputs, y_true
  ):
    y_pred, updated_non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs
    )
    loss_value = loss_fn(y_true, y_pred)
    return loss_value, updated_non_trainable_variables

  # Function to compute gradients
  compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)

  # Training step
  @jax.jit
  def train_step(train_state, inputs, y_true):
    trainable_variables, non_trainable_variables, optimizer_variables = (
        train_state
    )
    (loss_value, non_trainable_variables), grads = compute_gradients(
        trainable_variables, non_trainable_variables, inputs, y_true
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )

    return loss_value, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )

  # Replicate the model and optimizer variable on all devices
  def get_replicated_train_state(devices):
    # All variables will be replicated on all devices
    var_mesh = Mesh(devices, axis_names="_")
    # In NamedSharding, axes not mentioned are replicated (all axes here)
    var_replication = NamedSharding(var_mesh, P())

    # Apply the distribution settings to the model variables
    trainable_variables = jax.device_put(
        model.trainable_variables, var_replication
    )
    non_trainable_variables = jax.device_put(
        model.non_trainable_variables, var_replication
    )
    optimizer_variables = jax.device_put(optimizer.variables, var_replication)

    # Combine all state in a tuple
    return (trainable_variables, non_trainable_variables, optimizer_variables)

  num_devices = len(jax.local_devices())
  print(f"Running on {num_devices} devices: {jax.local_devices()}")
  devices = mesh_utils.create_device_mesh((num_devices,))

  # Data will be split along the batch axis
  data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
  data_sharding = NamedSharding(
      data_mesh,
      P(
          "batch",
      ),
  )  # naming axes of the sharded partition

  train_state = get_replicated_train_state(devices)

  # Custom training loop
  for epoch in range(num_epochs):
    data_iter = iter(train_data)
    loss_value = 0.0
    for step in range(config.steps_per_epoch):
      batch = next(data_iter)
      inputs, y_true = batch
      # Convert inputs to the expected format
      # inputs is a dict with 'dense_features' and 'sparse_features'
      dense_features = inputs["dense_features"].numpy()
      sparse_features = inputs["sparse_features"]
      sparse_features = [
          sparse_features[str(i)].numpy()
          for i in range(len(config.model.vocab_sizes))
      ]
      # Prepare the input list
      input_list = [dense_features] + sparse_features
      y_true = y_true.numpy()
      # Shard inputs
      sharded_inputs = [jax.device_put(x, data_sharding) for x in input_list]
      sharded_y_true = jax.device_put(y_true, data_sharding)
      loss_value, train_state = train_step(
          train_state, sharded_inputs, sharded_y_true
      )
    print(f"Epoch {epoch+1}, loss: {loss_value}")

  trainable_variables, non_trainable_variables, optimizer_variables = (
      train_state
  )
  for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
  for variable, value in zip(
      model.non_trainable_variables, non_trainable_variables
  ):
    variable.assign(value)


if __name__ == "__main__":
  app.run(main)
