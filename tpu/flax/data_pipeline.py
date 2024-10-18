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

from typing import List
import tensorflow as tf
from configs import DatasetFormat
import ml_collections


class CriteoTFRecordReader(object):
    """Input reader fn for TFRecords that have been serialized in batched form."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        is_training: bool,
        use_cached_data: bool = False,
    ):
        self._params = config.train_data if is_training else config.validation_data
        self._num_dense_features = config.model.num_dense_features
        self._vocab_sizes = config.model.vocab_sizes
        self._use_cached_data = use_cached_data

        self.label_features = "label"
        self.dense_features = ["dense-feature-%d" % x for x in range(1, 14)]
        self.sparse_features = ["sparse-feature-%d" % x for x in range(14, 40)]

    def __call__(self) -> tf.data.Dataset:
        params = self._params
        # Per replica batch size.
        batch_size = params.global_batch_size

        def _get_feature_spec():
            feature_spec = {}
            feature_spec[self.label_features] = tf.io.FixedLenFeature(
                [], dtype=tf.int64
            )
            for dense_feat in self.dense_features:
                feature_spec[dense_feat] = tf.io.FixedLenFeature(
                    [],
                    dtype=tf.float32,
                )
            for i, sparse_feat in enumerate(self.sparse_features):
                feature_spec[sparse_feat] = tf.io.FixedLenFeature(
                    [params.multi_hot_sizes[i]], dtype=tf.int64
                )
            return feature_spec

        def _parse_fn(serialized_example):
            feature_spec = _get_feature_spec()
            parsed_features = tf.io.parse_single_example(
                serialized_example, feature_spec
            )
            label = parsed_features[self.label_features]
            features = {}
            int_features = []
            for dense_ft in self.dense_features:
                int_features.append(parsed_features[dense_ft])
            features["dense_features"] = tf.stack(int_features)

            features["sparse_features"] = {}
            for i, sparse_ft in enumerate(self.sparse_features):
               features['sparse_features'][str(i)] = parsed_features[sparse_ft]
            return features, label

        # TODO(qinyiyan): Enable sharding.
        filenames = tf.data.Dataset.list_files(self._params.input_path, shuffle=False)

        num_shards_per_host = 1
        if params.sharding:
            num_shards_per_host = params.num_shards_per_host

        def make_dataset(shard_index):
            filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
            dataset = tf.data.TFRecordDataset(filenames_for_shard)
            if params.is_training:
                dataset = dataset.repeat()
            dataset = dataset.map(
                _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            return dataset

        indices = tf.data.Dataset.range(num_shards_per_host)
        dataset = indices.interleave(
            map_func=make_dataset,
            cycle_length=params.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if self._use_cached_data:
            dataset = dataset.take(1).cache().repeat()

        return dataset


class CriteoTsvReader:
    """Input reader callable for pre-processed Criteo data."""

    def __init__(self, config: ml_collections.ConfigDict, is_training: bool):
        self._params = config.train_data if is_training else config.validation_data
        self._model_config = config.model

    def __call__(self) -> tf.data.Dataset:
        if self._params.dataset_format == DatasetFormat.SYNTHETIC:
            return self._generate_synthetic_data()

        @tf.function
        def _parse_fn(example: tf.Tensor):
            """Parser function for pre-processed Criteo TSV records."""
            label_defaults = [[0.0]]
            dense_defaults = [
                [0.0] for _ in range(self._model_config.num_dense_features)
            ]
            num_sparse_features = len(self._model_config.vocab_sizes)
            categorical_defaults = [[0] for _ in range(num_sparse_features)]
            record_defaults = label_defaults + dense_defaults + categorical_defaults
            fields = tf.io.decode_csv(
                example, record_defaults, field_delim="\t", na_value="-1"
            )

            label = tf.reshape(fields[0], [1])
            features = {}
            dense_features = fields[1 : self._model_config.num_dense_features + 1]
            features["dense_features"] = tf.stack(dense_features, axis=0)
            features["sparse_features"] = {
                str(i): fields[i + self._model_config.num_dense_features + 1]
                for i in range(num_sparse_features)
            }
            return features, label

        filenames = tf.data.Dataset.list_files(self._params.input_path, shuffle=False)
        dataset = tf.data.TextLineDataset(filenames)
        if self._params.is_training:
            dataset = dataset.repeat()
        dataset = dataset.batch(self._params.global_batch_size, drop_remainder=True)
        dataset = dataset.map(_parse_fn, num_parallel_calls=1)
        dataset = dataset.prefetch(10)

        return dataset

    def _generate_synthetic_data(self) -> tf.data.Dataset:
        """Creates synthetic data based on the parameter batch size."""
        num_dense = self._model_config.num_dense_features
        dataset_size = 100 * self._params.global_batch_size

        dense_tensor = tf.random.uniform(
            shape=(dataset_size, num_dense), maxval=1.0, dtype=tf.float32
        )
        sparse_tensors = [
            tf.random.uniform(shape=(dataset_size,), maxval=int(size), dtype=tf.int32)
            for size in self._model_config.vocab_sizes
        ]
        sparse_tensor_elements = {
            str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
        }

        dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)
        sparse_tensors_mean = tf.math.reduce_sum(
            tf.stack(sparse_tensors, axis=-1), axis=1
        )
        sparse_tensors_mean = tf.cast(sparse_tensors_mean, dtype=tf.float32) / sum(
            self._model_config.vocab_sizes
        )
        label_tensor = tf.cast(
            (dense_tensor_mean + sparse_tensors_mean) / 2.0 + 0.5, tf.int32
        )

        input_elem = {
            "dense_features": dense_tensor,
            "sparse_features": sparse_tensor_elements,
        }, label_tensor
        dataset = tf.data.Dataset.from_tensor_slices(input_elem)
        dataset = dataset.cache()
        if self._params.is_training:
            dataset = dataset.repeat()
        dataset = dataset.batch(self._params.global_batch_size, drop_remainder=True)

        return dataset


def train_input_fn(config: ml_collections.ConfigDict) -> tf.data.Dataset:
    """Returns dataset of batched training examples."""
    if config.train_data.dataset_format in {DatasetFormat.SYNTHETIC, DatasetFormat.TSV}:
        return CriteoTsvReader(config, is_training=True)()
    return CriteoTFRecordReader(config, is_training=True)()


def eval_input_fn(config: ml_collections.ConfigDict) -> tf.data.Dataset:
    """Returns dataset of batched eval examples."""
    if config.validation_data.dataset_format in {DatasetFormat.SYNTHETIC, DatasetFormat.TSV}:
        return CriteoTsvReader(config, is_training=True)()
    return CriteoTFRecordReader(config, is_training=False)()
