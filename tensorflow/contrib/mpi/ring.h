#ifndef TENSORFLOW_CONTRIB_MPI_H_
#define TENSORFLOW_CONTRIB_MPI_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define TAG_TENSOR  12

namespace tensorflow {
namespace mpi {

Status RingAllreduce(OpKernelContext* context, Tensor& input, Tensor* output);

}
}

#undef TENSORFLOW_CONTRIB_MPI_H_
#endif // TENSORFLOW_CONTRIB_MPI_H_
