
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/mpi/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

template<> MPI_Datatype MPIType<float>() { return MPI_FLOAT; };
template<> MPI_Datatype MPIType<int>() { return MPI_INT; };

template<> DataType TensorFlowDataType<float>() { return DT_FLOAT; };
template<> DataType TensorFlowDataType<int>() { return DT_INT32; };

// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<GPUDevice, int>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<GPUDevice, float>(OpKernelContext*, Tensor&, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<GPUDevice, int>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<GPUDevice, float>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);

}
}
}
#endif
