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


// Convert from templated types to values we can pass to MPI.
template<typename T>
MPI_Datatype MPIType();

template<> MPI_Datatype MPIType<float>() { return MPI_FLOAT; };
template<> MPI_Datatype MPIType<int>() { return MPI_INT; };


// Convert from templated types to TensorFlow data types.
template<typename T>
DataType TensorFlowDataType();

template<> DataType TensorFlowDataType<float>() { return DT_FLOAT; };
template<> DataType TensorFlowDataType<int>() { return DT_INT32; };


// TODO: Make it so that this doesn't have to allocate temp and then copy to output

// Perform a ring allreduce on the data. Allocate the necessary output tensor and
// store it in the output parameter.
//
// Assumes that all MPI processes are doing an allreduce of the same tensor,
// with the same dimensions.
template<typename T>
Status RingAllreduce(OpKernelContext* context, Tensor& input, Tensor* output) {
    // Acquire MPI size and rank
    int n, r;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    // Allocate a new output tensor and copy data to it.
    Status status = context->allocate_temp(TensorFlowDataType<T>(), input.shape(), output);
    if(!status.ok()) {
        return status;
    }
    const CPUDevice& device = context->eigen_device<CPUDevice>();
    output->flat<T>().device(device) = input.flat<T>();

    T* buffer = (T*) output->tensor_data().data();

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
    status = context->allocate_temp(TensorFlowDataType<T>(), shape, &temp);
    if(!status.ok()) {
        return status;
    }
    T* segment_recv = (T*) temp.tensor_data().data();

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
        T* segment_send = &(buffer[segment_ends[((r-i) + n) % n] -
                                   segment_sizes[((r-i) + n) % n]]);

        MPI_Irecv(segment_recv, segment_sizes[((r-i-1) + n) % n],
                  MPIType<T>(), recv_from, TAG_TENSOR, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[((r-i) + n) % n],
                 MPIType<T>(), send_to, TAG_TENSOR, MPI_COMM_WORLD);

        T *segment_update = &(buffer[segment_ends[((r-i-1) + n) % n] -
                                     segment_sizes[((r-i-1) + n) % n]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

        const int N = segment_sizes[((r-i-1) + n) % n];
        auto eigen_recv = temp.flat<T>();
        auto eigen_update = typename TTypes<T>::Flat(
            (T*) segment_update, N);
        eigen_update.device(device) += eigen_recv;
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (r+1-i) and
    // receives segment (r-i).
    for (size_t i = 0; i < n - 1; ++i) {
        // Segment to send - at every iteration we send segment (r+1-i)
        T* segment_send = &(buffer[segment_ends[((r+1-i) + n) % n] -
                                   segment_sizes[((r+1-i) + n) % n]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        T* segment_recv = &(buffer[segment_ends[((r-i) + n) % n] -
                                   segment_sizes[((r-i) + n) % n]]);
        MPI_Sendrecv(segment_send, segment_sizes[((r+1-i) + n) % n],
                 MPIType<T>(), send_to, TAG_TENSOR, segment_recv,
                 segment_sizes[((r-i) + n) % n], MPIType<T>(), recv_from,
                 TAG_TENSOR, MPI_COMM_WORLD, &recv_status);
    }

    return Status::OK();
}

template Status RingAllreduce<int>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<float>(OpKernelContext*, Tensor&, Tensor*);

template<typename T>
Status RingAllgather(OpKernelContext* context, Tensor& input, Tensor* output,
                     std::vector<size_t>& sizes) {
    // Acquire MPI size and rank
    int n, r;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    assert(sizes.size() == n);
    assert(input.dim_size(0) == sizes[r]);

    // Compute output shape: all dimensions identical, except first, which is
    // the sum of all the input tensor sizes.
    size_t total_dimension_size = 0;
    for(auto dim : sizes) {
        total_dimension_size += dim;
    }

    tensorflow::TensorShape output_shape;
    output_shape.AddDim(total_dimension_size);
    for(int i = 1; i < input.shape().dims(); i++) {
        output_shape.AddDim(input.dim_size(i));
    }

    // Compute number of elements in every "row". We can't compute number of
    // elements in every chunks, because those chunks are variable length.
    size_t elements_per_row = 1;
    for(int i = 1; i < input.shape().dims(); i++) {
        elements_per_row *= input.dim_size(i);
    }

    Status status = context->allocate_temp(TensorFlowDataType<T>(), output_shape, output);
    if(!status.ok()) {
        return status;
    }

    // Copy data from input tensor to correct place in output tensor.
    std::vector<size_t> segment_starts(sizes.size());
    segment_starts[0] = 0;
    for(int i = 1; i < r; i++) {
        segment_starts[i] = segment_starts[i - 1] + elements_per_row * sizes[i - 1];
    }
    size_t offset = segment_starts[r];
    // COPY TO OFFSET

    T* buffer = (T*) output->tensor_data().data();

    // Receive from your left neighbor with wrap-around
    const size_t recv_from = ((r - 1) + n) % n;

    // Send to your right neighbor with wrap-around
    const size_t send_to = (r + 1) % n;

    // Perform a ring allgather. At every step, for every rank, we iterate
    // through segments with wraparound and send and recv from our neighbors.
    // At the i'th iteration, rank r, sends segment (r-i) and receives segment (r-1-i).
    MPI_Status recv_status;
    for (size_t i = 0; i < n - 1; ++i) {
        // Segment to send - at every iteration we send segment (r-i)
        size_t offset_send = segment_starts[(r - i + n) % n];
        size_t rows_send = sizes[(r - i + n) % n];
        T* segment_send = &(buffer[offset_send]);

        // Segment to recv - at every iteration we receive segment (r-1-i)
        size_t offset_recv = segment_starts[(r - i - 1 + n) % n];
        size_t rows_recv = sizes[(r - i - 1 + n) % n];
        T* segment_recv = &(buffer[offset_recv]);

        int result = MPI_Sendrecv(segment_send, elements_per_row * rows_send,
                                  MPIType<T>(), send_to, TAG_TENSOR, segment_recv,
                                  elements_per_row * rows_recv, MPIType<T>(), recv_from,
                                  TAG_TENSOR, MPI_COMM_WORLD, &recv_status);
        if(result != MPI_SUCCESS) {
            return errors::Unknown("MPI_Sendrecv failed in allgather.");
        }
    }
}

template Status RingAllgather<int>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<float>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);

}
}
