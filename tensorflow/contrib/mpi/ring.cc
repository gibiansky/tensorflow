#define EIGEN_USE_THREADS

#include "tensorflow/contrib/mpi/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

extern template MPI_Datatype MPIType<float>();
extern template MPI_Datatype MPIType<int>();
extern template DataType TensorFlowDataType<float>();
extern template DataType TensorFlowDataType<int>();


// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<CPUDevice, int>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<CPUDevice, float>(OpKernelContext*, Tensor&, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<CPUDevice, int>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<CPUDevice, float>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);

}
}
}
