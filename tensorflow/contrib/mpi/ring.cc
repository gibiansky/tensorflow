#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/contrib/mpi/ring.h"
#include "third_party/mpi/mpi.h"

namespace tensorflow {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

// Perform a ring allreduce on the data. Allocate the necessary output tensor and
// store it in the output parameter.
//
// Assumes that all MPI processes are doing an allreduce of the same tensor,
// with the same dimensions.
Status RingAllreduce(OpKernelContext* context, Tensor& input, Tensor* output) {
    // Acquire MPI size and rank
    int n, r;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    // Allocate a new output tensor and copy data to it. 
    Status status = context->allocate_temp(DT_FLOAT, input.shape(), output);
    if(!status.ok()) {
        return status;
    }
    const CPUDevice& device = context->eigen_device<CPUDevice>();
    output->flat<float>().device(device) = input.flat<float>(); 

    float* buffer = (float*) output->tensor_data().data();

    // Calculate segment sizes and segment ends
    const size_t elements_to_reduce = input.NumElements();
    const size_t segment_size = elements_to_reduce / n;
    std::vector<size_t> segment_sizes(n, segment_size);

    const size_t residual = elements_to_reduce % n;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    std::vector<size_t> segment_ends(n);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i-1];
    }

    assert(segment_ends[n-1] == elements_to_reduce);

    // Allocate temporary buffer - we know the first segment size is the
    // largest.
    tensorflow::TensorShape shape;
    tensorflow::Tensor temp;
    shape.AddDim(segment_sizes[0]);
    status = context->allocate_temp(tensorflow::DT_FLOAT, shape, &temp);
    if(!status.ok()) {
        return status;
    }
    float* segment_recv = (float*) temp.tensor_data().data();

    // Receive from your left neighbor with wrap-around
    const size_t recv_from = ((r - 1) + n) % n;

    // Send to your right neighbor with wrap-around
    const size_t send_to = (r + 1) % n;

    MPI_Status recv_status;
    MPI_Request recv_req;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, rank r, sends segment (r-i) and receives
    // segment (r-i-1).
    for (int i = 0; i < n - 1; i++) {
        float* segment_send = &(buffer[segment_ends[((r-i) + n) % n] -
                                       segment_sizes[((r-i) + n) % n]]);

        MPI_Irecv(segment_recv, segment_sizes[((r-i-1) + n) % n],
                  MPI_FLOAT, recv_from, TAG_TENSOR, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[((r-i) + n) % n],
                 MPI_FLOAT, send_to, TAG_TENSOR, MPI_COMM_WORLD);

        float *segment_update = &(buffer[segment_ends[((r-i-1) + n) % n] -
                                         segment_sizes[((r-i-1) + n) % n]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

        const int N = segment_sizes[((r-i-1) + n) % n];
        auto eigen_recv = temp.flat<float>();
        auto eigen_update = typename TTypes<float>::Flat(
            (float*) segment_update, N);
        eigen_update.device(device) += eigen_recv;
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (r+1-i) and
    // receives segment (r-i).
    for (size_t i = 0; i < n - 1; ++i) {
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(buffer[segment_ends[((r+1-i) + n) % n] -
                                       segment_sizes[((r+1-i) + n) % n]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(buffer[segment_ends[((r-i) + n) % n] -
                                       segment_sizes[((r-i) + n) % n]]);
        MPI_Sendrecv(segment_send, segment_sizes[((r+1-i) + n) % n],
                 MPI_FLOAT, send_to, TAG_TENSOR, segment_recv,
                 segment_sizes[((r-i) + n) % n], MPI_FLOAT, recv_from,
                 TAG_TENSOR, MPI_COMM_WORLD, &recv_status);
    }

    return Status::OK();
}

}
}
