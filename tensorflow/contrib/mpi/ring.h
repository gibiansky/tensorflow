#ifndef TENSORFLOW_CONTRIB_MPI_H_
#define TENSORFLOW_CONTRIB_MPI_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define TAG_TENSOR  12

namespace tensorflow {
namespace mpi {

template <typename T>
Status RingAllreduce(OpKernelContext* context, Tensor& input, Tensor* output);

template<typename T>
Status RingAllgather(OpKernelContext* context, Tensor& input, Tensor* output,
                     std::vector<size_t>& sizes);

}
}

#undef TENSORFLOW_CONTRIB_MPI_H_
#endif // TENSORFLOW_CONTRIB_MPI_H_
