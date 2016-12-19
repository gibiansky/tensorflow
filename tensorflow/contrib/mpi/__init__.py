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
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation
"""## Communicating Between Processes with MPI

TensorFlow natively provides inter-device communication through send and
receive ops and inter-node communication through Distributed TensorFlow, based
on the same send and receive abstractions. On HPC clusters where Infiniband or
other high-speed node interconnects are available, these can end up being
insufficient for synchronous data-parallel training (without asynchronous
gradient descent). This module implements a variety of MPI ops which can take
advantage of hardware-specific MPI libraries for efficient communication.

In order to use this module, TensorFlow must be built with an MPI library,
which can be provided to the `./configure` script at build time. As a user of
TensorFlow, you will need to build TensorFlow yourself to select the MPI
library to use; to do so, follow the [instructions for building TensorFlow from
source](https://www.tensorflow.org/get_started/os_setup#installing_from_sources).

### Utility Ops

In addition to reductions and gathers, this module provides utility operations
for detecting the running MPI configuration.

Example:

```python
from tensorflow.contrib import mpi

with tf.Session() as session:
    rank = session.run(mpi.rank())
    print("My MPI Rank:", rank)

    if rank == 0:
        print("MPI Size:", session.run(mpi.size()))
```

@@rank
@@size

### Ring Allreduce and Allgather

When summing or averaging tensors across many processes, communication can
easily become a bottleneck. A naive implementation will send all the tensor
values to the same process, perform the reduction, and then broadcast the
values back to all other processes, effectively creating a synchronous
parameter server in one process. However, the process responsible for
performing the reduction will have to receive and send a massive amount of data
which scales with the number of processes *and* the number of parameters in the
model.

Instead of centralizing the reduction and having one primary reducer, we can
implement a distributed allreduce or allgather. A bandwidth-optimal allreduce
will end up sending 2(N - 1) values for every value in the input tensor [1],
and can be implemented with a ring allreduce [2]. This module implements
bandwidth-optimal ring allreduce and ring allgather operations using MPI; by
choosing a hardware-appropriate MPI implementation (such as OpenMPI with
CUDA-IPC support), you can train large models with synchronous gradient descent
with minimal communication overhead.

In addition to the `allreduce` and `allgather` functions, a convenience
`DistributedOptimizer` wrapper is provided to simplify using these functions
for reducing model gradients.

Example:

```python
import tensorflow as tf
from tensorflow.contrib import mpi

# Construct a simple linear regression model to optimize
W = tf.get_variable("W", shape=[20, 1], dtype=tf.float32)
B = tf.get_variable("B", shape=[1, 1], dtype=tf.float32)
inputs = tf.placeholder("Inputs", shape=[None, 20])
outputs = tf.placeholder("Outputs", shape=[None, 1])
loss = tf.nn.l2_loss(tf.matmul(inputs, W) + B - outputs)

# Training using MPI allreduce with DistributedOptimizer
optimizer = mpi.DistributedOptimizer(tf.train.AdamOptimizer())
train = optimizer.minimize(loss)

# Average loss over all ranks, for printing.
# Do not pass this to an optimizer!
avg_loss = mpi.allreduce(loss)

# On different ranks, feed different input data.
with tf.Session() as session:
    rank = session.run(mpi.rank())
    batch_inputs, batch_outputs = construct_batch_for_rank(rank)
    feed_dict = {inputs: batch_inputs, outputs: batch_outputs}
    _, l = session.run([train, avg_loss], feed_dict=feed_dict)
    print("Average Loss:", l)
```

@@DistributedOptimizer
@@allreduce
@@allgather
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.train import Optimizer

from tensorflow.contrib.mpi.mpi_ops import size
from tensorflow.contrib.mpi.mpi_ops import rank
from tensorflow.contrib.mpi.mpi_ops import allgather
import tensorflow.contrib.mpi.mpi_ops as mpi_ops

def allreduce(tensor, average=True):
    """Perform an MPI allreduce on a tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
        The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.
    """
    if isinstance(tensor, tf.IndexedSlices):
        # For IndexedSlices, do two allgathers intead of an allreduce.
        mpi_size = tf.cast(size(), tensor.values.dtype)
        values = allgather(tensor.values)
        indices = allgather(tensor.indices)

        # To make this operation into an average, divide all gathered values by
        # the MPI size.
        new_values = values / mpi_size if average else values
        return tf.IndexedSlices(values / mpi_size, indices,
                                dense_shape=tensor.dense_shape)
    else:
        mpi_size = tf.cast(size(), tensor.dtype)
        summed_tensor = mpi_ops.allreduce(tensor)
        new_tensor = summed_tensor / mpi_size if average else summed_tensor
        return new_tensor / mpi_size


class DistributedOptimizer(Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an MPI allreduce to
    average gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, name=None, use_locking=False):
        """Construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged
        across all the MPI ranks.

        Args:
          optimizer:
            Optimizer to use for computing gradients and applying updates.
          name:
            Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
          use_locking:
            Whether to use locking when updating variables.
            See Optimizer.__init__ for more info.
        """
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = (super(DistributedOptimizer, self)
                     .compute_gradients(*args, **kwargs))
        return [(allreduce(gradient), var) for (gradient, var) in gradients]

    def _apply_dense(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._apply_dense(*args, **kwargs)

    def _apply_sparse(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._apply_sparse(*args, **kwargs)

    def _prepare(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._prepare(*args, **kwargs)

    def _create_slots(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._create_slots(*args, **kwargs)

    def _valid_dtypes(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._valid_dtypes(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._finish(*args, **kwargs)
