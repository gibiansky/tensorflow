// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <stdlib.h>

#include <cstdio>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "third_party/mpi/mpi.h"

namespace tensorflow {
namespace mpi {

namespace {
    // Whether MPI has been initialized. Making this atomic ensures
    // MPI gets initialized exactly once.
    std::atomic_flag mpi_initialized = ATOMIC_FLAG_INIT;

    Status InitializeMPIOnce() {
        // Ensure MPI is only initialized once.
        if(mpi_initialized.test_and_set())
            return Status::OK();

        auto init_result = MPI_Init(NULL, NULL);
        if(init_result != MPI_SUCCESS) {
            return errors::Unknown("Could not initialize MPI; MPI_Init() failed.");
        }
        return Status::OK();
    }
}

// Op to get the current MPI Size.
class MPISizeOp : public OpKernel {
 public:
  explicit MPISizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }


  void Compute(OpKernelContext* context) override {
    // Get the number of processes
    int world_size;
    OP_REQUIRES(context, MPI_Comm_size(MPI_COMM_WORLD, &world_size) == MPI_SUCCESS,
            errors::Unknown("Could not get MPI Comm size; MPI_Comm_size() failed."));

    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0) = world_size;
  }
};

REGISTER_KERNEL_BUILDER(Name("MPISize").Device(DEVICE_CPU), MPISizeOp);

REGISTER_OP("MPISize")
    .Output("size: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the number of running MPI processes.

More precisely, returns the number of MPI processes in the group associated
with the MPI_COMM_WORLD communicator.

size:   Size of the MPI group.
)doc");

// Op to get the current MPI Rank.
class MPIRankOp : public OpKernel {
 public:
  explicit MPIRankOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }

  void Compute(OpKernelContext* context) override {
    // Get the processor index
    int rank;
    OP_REQUIRES(context, MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
            errors::Unknown("Could not get MPI Comm rank; MPI_Comm_rank() failed."));

    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0) = rank;
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIRank").Device(DEVICE_CPU), MPIRankOp);

REGISTER_OP("MPIRank")
    .Output("rank: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the index of the current process in the MPI group.

More precisely, returns the rank of the calling process in the MPI_COMM_WORLD
communicator.

rank:   Rank of the calling process.
)doc");

}  // namespace mpi
}  // namespace tensorflow
