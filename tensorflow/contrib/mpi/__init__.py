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

TensorFlow provides Ops to communicate with other TensorFlow processes using
MPI.

Example:

```python
from tensorflow.contrib import mpi

# Get MPI Size (like MPI_Comm_size())
mpi_size = mpi.size()

# Get MPI Rank (like MPI_Comm_rank())
mpi_rank = mpi.rank()

# Allreduce a tensor
reduced_tensor = mpi.allreduce([1, 2, 3])
```

@@size
@@rank
@@allreduce
@@allgather
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.mpi.mpi_ops import size
from tensorflow.contrib.mpi.mpi_ops import rank
from tensorflow.contrib.mpi.mpi_ops import allreduce
from tensorflow.contrib.mpi.mpi_ops import allgather
