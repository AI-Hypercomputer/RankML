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

import collections
import functools
import threading
import time
from typing import Any, List, Mapping

from absl import app
from absl import flags
from absl import logging
from dataloader import CriteoDataLoader
from dataloader import DataConfig
from dlrm_model import DLRMDCNV2
from dlrm_model import uniform_init
import jax
from jax.experimental.layout import DeviceLocalLayout as DLL
from jax.experimental.layout import Layout
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.flax import embed_optimizer
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import optax


partial = functools.partial
info = logging.info
shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested

FLAGS = flags.FLAGS

VOCAB_SIZES = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
MULTI_HOT_SIZES = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]

np.set_printoptions(threshold=np.inf)

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 135168, "Batch size.")

_FILE_PATTERN = flags.DEFINE_string(
    "file_pattern",
    None,
    "File pattern for the training data.",
)

_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")

_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    "num_dense_features", 13, "Number of dense features."
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 1000, "Number of steps to train for."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs",
    1,
    "Number of separate epochs.",
)

_EMBEDDING_SIZE = flags.DEFINE_integer("embedding_size", 128, "Embedding size.")

_ALLOW_ID_DROPPING = flags.DEFINE_bool(
    "allow_id_dropping",
    False,
    "If set, allow dropping ids during embedding lookup.",
)

_EMBEDDING_THRESHOLD = flags.DEFINE_integer(
    "embedding_threshold",
    0,
    "Embedding threshold for placing features on TensorCore or SparseCore.",
)

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    "jax_dlrm",
    "The name of the experiment. If not set the name of the fixture that"
    " returns the fiddle configuration will be used instead.",
)
_MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    "/tmp/dlrm_test",
    "Model working directory.",
)

_MODE = flags.DEFINE_enum(
    "mode",
    default=None,
    enum_values=[
        "train",
        "eval",
        "train_and_eval",
        "continuous_eval",
    ],
    help="Mode to run: `train`, `eval`, `train_and_eval`, `continuous_eval`.",
)

_BINARY_PATH = flags.DEFINE_string("binary_path", None, "Path to train binary.")

_PRIORITY = flags.DEFINE_integer(
    "priority", 200, "Priority for the training job."
)
_BORG_USER = flags.DEFINE_string("borg_user", None, "Borg user.")
_CELL = flags.DEFINE_string("cell", "yo", "Borg cell.")
_LOGS_ACCESS = flags.DEFINE_multi_string(
    "logs_access",
    [],
    "List of roles that can read the logs of the training job.",
)

_TRAIN_PKG_NAME = flags.DEFINE_string(
    "train_pkg_name",
    None,
    "MPM package name of the train binary. Must be passed with"
    " `train_pkg_binary` and `train_version`.",
)
_TRAIN_PKG_BINARY = flags.DEFINE_string(
    "train_pkg_binary",
    None,
    "Binary path of the train binary within the MPM. Must be passed with"
    " `train_pkg_name` and `train_version`.",
)
_TRAINER_VERSION = flags.DEFINE_string(
    "train_version",
    None,
    "Version of the train binary MPM. Must be passed with `train_pkg_name` and"
    " `train_pkg_binary`.",
)

_VIZIER_STUDY_CONFIG = flags.DEFINE_string(
    "vizier_study_config",
    None,
    "Path to a `vizier_pb2.StudyConfig` text proto. Must be set to use Vizier.",
)
_VIZIER_NUM_PARALLEL_RUNS = flags.DEFINE_integer(
    "vizier_num_parallel_runs",
    2,
    "Number of parallel trials to run with Vizier.",
)


def create_table_and_feature_specs(
    vocab_sizes: List[int],
) -> tuple[
    Mapping[str, embedding_spec.TableSpec],
    Mapping[str, embedding_spec.FeatureSpec],
]:
  """Creates the feature specs for the DLRM model."""
  table_specs = {}
  feature_specs = {}
  for i, vocab_size in enumerate(vocab_sizes):
    table_name = f"{i}"
    feature_name = f"{i}"
    bound = jnp.sqrt(1.0 / vocab_size)
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size,
        embedding_dim=_EMBEDDING_SIZE.value,
        initializer=uniform_init(bound),
        optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
        combiner="mean",
        name=table_name,
        max_ids_per_partition=2048,
        max_unique_ids_per_partition=512,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(_BATCH_SIZE.value, 1),
        output_shape=(
            _BATCH_SIZE.value,
            _EMBEDDING_SIZE.value,
        ),
        name=feature_name,
    )
    feature_specs[feature_name] = feature_spec
    table_specs[table_name] = table_spec
  return table_specs, feature_specs


class DLRMDataLoader:
  """Parallel data producer for the DLRM model."""

  def __init__(
      self,
      file_pattern: str,
      batch_size,
      num_workers=4,
      buffer_size=128,
      feature_specs=None,
      mesh=None,
      global_sharding=None,
  ):
    """Initialize the producer.

    Args:
        file_pattern: File pattern for the training data.
        batch_size: The size of each batch.
        num_workers: Number of threads generating batches.
        buffer_size: Maximum number of batches to store in the buffer.
        feature_specs: Specifications for features.
        mesh: Mesh configuration for distributed processing.
        global_sharding: Global sharding configuration.
    """
    # Here we pass global batch size. TF.data will shard it.
    self.data_config = DataConfig(
        global_batch_size=batch_size,
        is_training=True,
        use_cached_data=True if file_pattern is None else False,
    )
    self._dataloader = CriteoDataLoader(
        file_pattern=file_pattern,
        params=self.data_config,
        num_dense_features=_NUM_DENSE_FEATURES.value,
        vocab_sizes=VOCAB_SIZES,
        multi_hot_sizes=MULTI_HOT_SIZES,
        embedding_threshold=_EMBEDDING_THRESHOLD.value,
    )
    self._iterator = self._dataloader.get_iterator()
    self.batch_size = batch_size
    self.feature_specs = feature_specs
    self.mesh = mesh
    self.global_sharding = global_sharding

    self.buffer = collections.deque(maxlen=buffer_size)
    self._sync = threading.Condition()
    self._workers = []

    for _ in range(num_workers):
      worker = threading.Thread(target=self._worker_loop, daemon=True)
      worker.start()
      self._workers.append(worker)

  def process_inputs(self, feature_batch):
    """Process input features into the required format."""
    dense_features = feature_batch["dense_features"]
    sparse_features = feature_batch["sparse_features"]
    labels = feature_batch["clicked"]

    feature_weights = jax.tree_util.tree_map(
        lambda x: np.array(
            np.ones_like(x, shape=x.shape, dtype=np.float32)
        ),
        sparse_features
    )

    # Process sparse features
    processed_sparse = embedding.preprocess_sparse_dense_matmul_input(
        sparse_features,
        feature_weights,
        self.feature_specs,
        self.mesh.local_mesh.size,
        self.mesh.size,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        allow_id_dropping=_ALLOW_ID_DROPPING.value,
        static_buffer_size_multiplier=256,
    )[:-1]

    # Preprocess the inputs and build Jax global views of the data.
    # Global view is required for embedding lookup.
    make_global_view = lambda x: jax.tree.map(
        lambda y: jax.make_array_from_process_local_data(
            self.global_sharding, y
        ),
        x,
    )
    labels = make_global_view(labels)
    dense_features = make_global_view(dense_features)
    processed_sparse = map(make_global_view, processed_sparse)
    return [
        labels,
        dense_features,
        embed.EmbeddingLookups(*processed_sparse)
    ]

  def _worker_loop(self):
    """Worker thread that continuously generates and processes batches."""
    while True:
      batch = next(self._iterator)
      processed_batch = self.process_inputs(batch)
      # processed_batch = jax.device_put(processed_batch, self.global_sharding)

      with self._sync:
        self._sync.wait_for(
            lambda: len(self.buffer) < self.buffer.maxlen
        )
        self.buffer.append(processed_batch)
        self._sync.notify_all()

  def __iter__(self):
    return self

  def __next__(self):
    """Get next batch from the buffer."""
    with self._sync:
      self._sync.wait_for(lambda: self.buffer)
      item = self.buffer.popleft()
      self._sync.notify_all()
      return item

  def stop(self):
    """Stop all worker threads and clear the buffer."""
    for worker in self._workers:
      worker.join(timeout=5)

    with self._sync:
      self.buffer.clear()
      self._sync.notify_all()


def test_dlrm_dcnv2_model():
  """Test DLRM DCN v2 model."""

  # Define sharding.
  pd = P("x")
  global_devices = jax.devices()
  mesh = jax.sharding.Mesh(global_devices, "x")
  global_sharding = jax.sharding.NamedSharding(mesh, pd)

  _, feature_specs = create_table_and_feature_specs(VOCAB_SIZES)
  def _get_max_ids_per_partition(name: str, batch_size: int) -> int:
    """Reference implementation for calculating max ids per partition on the fly.

    Args:
        name: Name of the feature.
        batch_size: The size of each batch.

    Returns:
        Estimated maximum number of ids per partition.

    """
    logging.info("max_ids_per_partition: %s : # of ids: %s", name, batch_size)
    return 4096

  def _get_max_unique_ids_per_partition(name: str, batch_size: int) -> int:
    """Reference implementation for calculating max unique ids per partition on the fly.

    Args:
        name: Name of the feature.
        batch_size: The size of each batch.

    Returns:
        Estimated maximum number of unique ids per partition.
    """
    logging.info(
        "max_unique_ids_per_partition: %s : # of ids: %s", name, batch_size
    )
    return 2048

  # Table stacking
  embedding.auto_stack_tables(
      feature_specs,
      global_device_count=jax.device_count(),
      stack_to_max_ids_per_partition=_get_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=_get_max_unique_ids_per_partition,
  )

  embedding.prepare_feature_specs_for_training(
      feature_specs,
      global_device_count=jax.device_count(),
  )

  # Construct the model.
  model = DLRMDCNV2(
      feature_specs=feature_specs,
      mesh=mesh,
      sharding_axis="x",
      global_batch_size=_BATCH_SIZE.value,
      embedding_size=_EMBEDDING_SIZE.value,
      bottom_mlp_dims=[512, 256, _EMBEDDING_SIZE.value],
  )

  producer = DLRMDataLoader(
      file_pattern=None,
      batch_size=_BATCH_SIZE.value,
      num_workers=32,
      buffer_size=256,
      feature_specs=feature_specs,
      mesh=mesh,
      global_sharding=global_sharding,
  )

  _, dense_features, embedding_lookups = next(producer)
  params = model.init(jax.random.key(42), dense_features, embedding_lookups)

  tx = embed_optimizer.create_optimizer_for_sc_model(
      params,
      optax.adagrad(learning_rate=_LEARNING_RATE.value),
  )
  opt_state = tx.init(params)

  dense_out_shardings = {
      f"Dense_{i}": {
          "bias": Layout(
              DLL(major_to_minor=(0,)),
              NamedSharding(mesh, P()),
          ),
          "kernel": Layout(
              DLL(major_to_minor=(0, 1)),
              NamedSharding(mesh, P()),
          ),
      }
      for i in range(8)
  }

  stacked_table_names = set([
      f.table_spec.setting_in_stack.stack_name
      for _, f in feature_specs.items()
  ])
  embedding_out_shardings = {
      "SparseCoreEmbed_0": {
          "sc_embedding_variables": embed.WithSparseCoreLayout(
              value={
                  f"{key}": Layout(
                      DLL(
                          major_to_minor=(0, 1),
                          _tiling=((8,),),
                      ),
                      NamedSharding(mesh, P("x", None)),
                  )
                  for key in stacked_table_names
              },
              names=("x",),
              mesh=mesh,
          )
      }
  }

  dcn_out_shardings = {
      f"bias_{i}": Layout(
          DLL(major_to_minor=(0,)), NamedSharding(mesh, P())
      )
      for i in range(3)
  }
  dcn_out_shardings.update({
      f"u_kernel_{i}": Layout(
          DLL(major_to_minor=(0, 1)), NamedSharding(mesh, P())
      )
      for i in range(3)
  })
  dcn_out_shardings.update({
      f"v_kernel_{i}": Layout(
          DLL(major_to_minor=(0, 1)), NamedSharding(mesh, P())
      )
      for i in range(3)
  })

  out_shardings = {
      "params": {
          **dense_out_shardings,
          **embedding_out_shardings,
          **dcn_out_shardings,
      }
  }

  @partial(
      jax.jit,
      out_shardings=(out_shardings, None, None),
      donate_argnums=(0,),
  )
  def train_step(
      params: Any,
      labels: jax.Array,
      dense_features: jax.Array,
      embedding_lookups: embed.EmbeddingLookups,
      opt_state,
  ):
    def forward_pass(params, labels, dense_features, embedding_lookups):
      logits = model.apply(params, dense_features, embedding_lookups)
      xentropy = optax.sigmoid_binary_cross_entropy(
          logits=logits, labels=labels
      )
      return jnp.mean(xentropy), logits

    # Run model forward/backward pass.
    train_step_fn = jax.value_and_grad(
        forward_pass, has_aux=True, allow_int=True
    )

    (loss_val, unused_logits), grads = train_step_fn(
        params, labels, dense_features, embedding_lookups
    )

    updates, opt_state = tx.update(grads, opt_state)
    params = embed_optimizer.apply_updates_for_sc_model(params, updates)

    return params, opt_state, loss_val

  start_time = time.time()

  for step in range(_NUM_STEPS.value):
    with jax.profiler.StepTraceAnnotation("step", step_num=step):
      # ------------------------------------------------------------------------
      # Step 1: SC input processing.
      # ------------------------------------------------------------------------
      labels, dense_features, embedding_lookups = next(producer)
      # ------------------------------------------------------------------------
      # Step 2: run model.
      # ------------------------------------------------------------------------
      params, opt_state, loss_val = train_step(
          params, labels, dense_features, embedding_lookups, opt_state
      )

      if step % 100 == 0:
        end_time = time.time()

        logging.info(
            "Step %s: %lf, mean_step_time = %lf",
            step,
            loss_val,
            (end_time - start_time) / 100,
        )
        start_time = time.time()
  producer.stop()


def main(argv):
  del argv
  test_dlrm_dcnv2_model()


if __name__ == "__main__":
  app.run(main)

