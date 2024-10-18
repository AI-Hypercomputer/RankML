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

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model configuration
  config.model = ml_collections.ConfigDict()
  config.model.vocab_sizes = [1000, 1000, 1000]  # Example vocab sizes
  config.model.num_dense_features = 13  # Example number of dense features
  config.model.embedding_dim = 32  # Add this line
  config.model.bottom_mlp_dims = [64, 32, 16]  # Add this line
  config.model.top_mlp_dims = [64, 32, 1]  # Add this line
  config.model.learning_rate = 0.001  # Add this line

  # Data configuration
  config.train_data = ml_collections.ConfigDict()
  config.train_data.input_path = 'path/to/train/data/*.tsv'
  config.train_data.global_batch_size = 1024
  config.train_data.is_training = True
  config.train_data.sharding = True
  config.train_data.num_shards_per_host = 8
  config.train_data.cycle_length = 8
  config.train_data.use_synthetic_data = True

  config.validation_data = ml_collections.ConfigDict()
  config.validation_data.input_path = 'path/to/validation/data/*.tsv'
  config.validation_data.global_batch_size = 1024
  config.validation_data.is_training = False
  config.validation_data.sharding = False
  config.validation_data.use_synthetic_data = True

  # Global configuration
  config.num_epochs = 10  # Make sure this is defined
  config.steps_per_epoch = (
      100  # Adjust this value based on your dataset size and batch size
  )

  return config
