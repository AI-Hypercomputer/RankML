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

import enum
import ml_collections

class DatasetFormat(enum.Enum):
    """Defines the dataset format."""
    TSV = "tsv"
    TFRECORD = "tfrecord"
    SYNTHETIC = "synthetic"


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
    config.train_data.dataset_format = DatasetFormat.SYNTHETIC

    config.validation_data = ml_collections.ConfigDict()
    config.validation_data.input_path = 'path/to/validation/data/*.tsv'
    config.validation_data.global_batch_size = 1024
    config.validation_data.is_training = False
    config.validation_data.sharding = False
    config.validation_data.dataset_format = DatasetFormat.SYNTHETIC

    # Global configuration
    config.num_epochs = 10  # Make sure this is defined
    config.steps_per_epoch = 100  # Adjust this value based on your dataset size and batch size

    return config

def get_criteo_config():
    """Get the configuration for the Criteo dataset."""
    config = ml_collections.ConfigDict()

    # Model configuration
    config.model = ml_collections.ConfigDict()
    config.model.vocab_sizes = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]  # Example vocab sizes
    config.model.num_dense_features = 13  # Example number of dense features
    config.model.embedding_dim = 16  # TODO(qinyiyan): Use larger embedding vector when ready.
    config.model.bottom_mlp_dims = [512, 256, 128]  # Add this line
    config.model.top_mlp_dims = [1024, 1024, 512, 256, 1]  # Add this line
    config.model.learning_rate = 0.025  # Add this line

    # Data configuration
    config.train_data = ml_collections.ConfigDict()
    config.train_data.input_path = 'gs://rankml-datasets/criteo/train/day_0/*00100*'
    config.train_data.global_batch_size = 1024
    config.train_data.is_training = True
    config.train_data.sharding = True
    config.train_data.num_shards_per_host = 8
    config.train_data.cycle_length = 8
    config.train_data.multi_hot_sizes = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
    config.train_data.dataset_format = DatasetFormat.TFRECORD

    config.validation_data = ml_collections.ConfigDict()
    config.validation_data.input_path = 'gs://rankml-datasets/criteo/eval/day_23/*00000*'
    config.validation_data.global_batch_size = 1024
    config.validation_data.is_training = False
    config.validation_data.sharding = False
    config.validation_data.cycle_length = 8
    config.validation_data.multi_hot_sizes = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
    config.validation_data.dataset_format = DatasetFormat.TFRECORD


    # Global configuration
    config.num_epochs = 10  # Make sure this is defined
    config.steps_per_epoch = 100  # Adjust this value based on your dataset size and batch size

    return config