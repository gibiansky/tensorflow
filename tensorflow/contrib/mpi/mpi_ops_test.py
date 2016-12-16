# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Tests for tensorflow.contrib.mpi.mpi_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

import tensorflow.contrib.mpi as mpi
from tensorflow.python.platform import resource_loader

MPI_ENV_RANK = "PMI_RANK"
MPI_ENV_SIZE = "PMI_SIZE"


class MPITests(tf.test.TestCase):
    def test_mpi_rank(self):
        """Test that the rank returned by mpi.rank() is correct."""
        true_rank = int(os.environ.get(MPI_ENV_RANK, "0"))
        with self.test_session() as session:
            rank = session.run(mpi.rank())
            self.assertEqual(true_rank, rank)

    def test_mpi_size(self):
        """Test that the size returned by mpi.size() is correct."""
        true_size = int(os.environ.get(MPI_ENV_SIZE, "1"))
        with self.test_session() as session:
            size = session.run(mpi.size())
            self.assertEqual(true_size, size)

    def test_mpi_allreduce_float(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D float tensors."""
        with self.test_session() as session:
            size = session.run(mpi.size())

            for dim in [1, 2, 3]:
                tf.set_random_seed(1234)
                tensor = tf.random_uniform([17] * dim, -1.0, 1.0)
                summed = mpi.allreduce(tensor)
                multiplied = tensor * tf.cast(size, tf.float32)
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))
                self.assertTrue(session.run(max_difference) < 1e-4,
                                "mpi.allreduce produces incorrect results")

    def test_mpi_allreduce_int(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D int32 tensors."""
        with self.test_session() as session:
            size = session.run(mpi.size())

            for dim in [1, 2, 3]:
                tf.set_random_seed(1234)
                tensor = tf.random_uniform([17] * dim, -100, 100,
                                           dtype=tf.int32)
                summed = mpi.allreduce(tensor)
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))
                self.assertTrue(session.run(max_difference) == 0,
                                "mpi.allreduce produces incorrect results")

    def test_mpi_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different rank or dimension."""
        with self.test_session() as session:
            size = session.run(mpi.size())
            rank = session.run(mpi.rank())

            # Same rank, different dimension
            tf.set_random_seed(1234)
            dims = [17 + rank] * 3
            tensor = tf.random_uniform(dims, -1.0, 1.0)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(mpi.allreduce(tensor))

            # Same number of elements, different rank
            tf.set_random_seed(1234)
            if rank == 0:
                dims = [17, 23 * 57]
            else:
                dims = [17, 23, 57]
            tensor = tf.random_uniform(dims, -1.0, 1.0)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(mpi.allreduce(tensor))

    def test_mpi_allreduce_type_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different type."""
        with self.test_session() as session:
            size = session.run(mpi.size())
            rank = session.run(mpi.rank())

            # Same rank, different dimension
            dims = [17 + rank] * 3
            tensor = tf.ones(dims,
                             dtype=tf.int32 if rank % 2 == 0 else tf.float32)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(mpi.allreduce(tensor))

    def test_mpi_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        with self.test_session() as session:
            size = session.run(mpi.size())
            rank = session.run(mpi.rank())

            for dtype in [tf.int32, tf.float32]:
                for dim in [1, 2, 3]:
                    tensor = tf.ones([17] * dim, dtype=dtype) * rank
                    gathered = mpi.allgather(tensor)

                    gathered_tensor = session.run(gathered)
                    self.assertEqual(list(gathered_tensor.shape),
                                     [17 * size] + [17] * (dim - 1))

                    rank_tensors = session.run(
                        tf.split(value=gathered, axis=0,
                                 num_or_size_splits=size))

                    for i, rank_tensor in enumerate(rank_tensors):
                        self.assertEqual(list(rank_tensor.shape),[17] * dim)
                        self.assertTrue(
                            session.run(tf.reduce_all(rank_tensor == i)),
                            "mpi.allgather produces incorrect gathered tensor")

    def test_mpi_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        with self.test_session() as session:
            size = session.run(mpi.size())
            rank = session.run(mpi.rank())

            for dtype in [tf.int32, tf.float32]:
                for dim in [1, 2, 3]:
                    tensor_sizes = [17, 32, 81, 12, 15, 23, 22][:size]
                    tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1),
                                     dtype=dtype) * rank
                    gathered = mpi.allgather(tensor)

                    gathered_tensor = session.run(gathered)
                    expected_size = sum(tensor_sizes)
                    self.assertEqual(list(gathered_tensor.shape),
                                     [expected_size] + [17] * (dim - 1))

                    rank_tensors = session.run(
                        tf.split_v(value=gathered, axis=0,
                                   num_or_size_splits=tensor_sizes))

                    for i, rank_tensor in enumerate(rank_tensors):
                        rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                        self.assertEqual(list(rank_tensor.shape), rank_size)
                        self.assertTrue(
                            session.run(tf.reduce_all(rank_tensor == i)),
                            "mpi.allgather produces incorrect gathered tensor")

    def test_mpi_allgather_error(self):
        """Test that the allgather returns an error if any dimension besides
        the first is different among the tensors being gathered."""
        with self.test_session() as session:
            rank = session.run(mpi.rank())

            tensor_size = [17] * dim
            tensor_size[rank] = 10 * (rank + 1)
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(mpi.allgather(tensor))

    def test_mpi_allgather_type_error(self):
        """Test that the allgather returns an error if the types being gathered
        differ among the processes"""
        with self.test_session() as session:
            rank = session.run(mpi.rank())

            tensor_size = [17] * dim
            dtype = tf.int32 if rank % 2 == 0 else tf.float32
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(mpi.allgather(tensor))


if __name__ == '__main__':
  tf.test.main()
