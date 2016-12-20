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

#include <unordered_map>
#include <queue>
#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#define OMPI_SKIP_MPICXX
#include "third_party/mpi/mpi.h"
#include "tensorflow/contrib/mpi/ring.h"
#include "tensorflow/contrib/mpi/mpi_message.pb.h"

/*
 * MPI Allreduce and Allgather Ops for TensorFlow.
 *
 * TensorFlow natively provides inter-device communication through send and
 * receive ops and inter-node communication through Distributed TensorFlow,
 * based on the same send and receive abstractions. These end up being
 * insufficient for synchronous data-parallel training on HPC clusters where
 * Infiniband or other high-speed interconnects are available.  This module
 * implements MPI ops for allgather and allreduce, which do bandwidth-optimal
 * gathers and reductions and can take advantage of hardware-optimized
 * communication libraries through the MPI implementation.
 *
 * The primary logic of the allreduce and allgather are in RingAllgather() and
 * RingAllreduce(). The background thread which facilitates MPI operations is
 * run in BackgroundThreadLoop(). The provided MPI ops are:
 *      – MPISize:
 *          Get the number of MPI processes in the global communicator.
 *      – MPIRank:
 *          Get the rank of the current MPI process in the global communicator.
 *      – MPIAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all MPI processes in the global communicator.
 *      – MPIAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MPI processes in the
 *          global communicator.
 *
 */

template<class T>
using StatusOr = perftools::gputools::port::StatusOr<T>;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow {
namespace contrib {
namespace mpi {

namespace {

// Return true if the templated type is GPUDevice, otherwise false.
template<typename T> bool IsGPUDevice();
template<> bool IsGPUDevice<GPUDevice>() { return true; };
template<> bool IsGPUDevice<CPUDevice>() { return false; };

// A callback to call after the MPI communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
typedef std::function<void(StatusOr<Tensor>)> CommunicationDoneCallback;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction:
//  - Tensor: The tensor data.
//  - OpKernelContext*: A context used to allocate the output or temporary values.
//  - CommunicationDoneCallback: A callback to call with the result.
typedef std::unordered_map<std::string, std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> > TensorTable;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking and size calculations, as well as determining when a reduction is
// ready to be done (when all nodes are ready to do it).
typedef std::unordered_map<std::string, std::vector<MPIRequest> > MessageTable;

// The global state required for the MPI ops.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct MPIGlobalState {
    // An atomic boolean which is set to true when MPI is initialized.
    // This ensures that MPI_Init is never called twice.
    std::atomic_flag initialized_flag;

    // A mutex that needs to be used whenever MPI operations are done.
    std::mutex mutex;

    // Tensors waiting to be allreduced or allgathered.
    TensorTable tensor_table;

    // Queue of MPI requests waiting to be sent to the coordinator node.
    std::queue<MPIRequest> message_queue;

    // Background thread running MPI communication.
    std::thread background_thread;

    // Whether the background thread should shutdown.
    bool shut_down;

    // Only exists on the coordinator node (rank zero). Maintains a count of
    // how many nodes are ready to allreduce every tensor (keyed by tensor
    // name).
    std::unique_ptr<MessageTable> message_table;

    // Whether MPI_Init has been completed on the background thread.
    bool initialization_done;

    // Whether MPI_Init succeeded on the background thread.
    Status init_status;

    // The MPI rank and size.
    int rank;
    int size;

    ~MPIGlobalState() {
        // Make sure that the destructor of the background thread is safe to
        // call. If a thread is still joinable (not detached or complete) its
        // destructor cannot be called.
        if(background_thread.joinable()) {
            shut_down = true;
            background_thread.join();
        }
    }
};

// All the MPI state that must be stored globally per-process.
static MPIGlobalState mpi_global = {
    .initialized_flag = ATOMIC_FLAG_INIT,
    .initialization_done = false
};

// For clarify in argument lists.
#define RANK_ZERO   0

// A tag used for all coordinator messaging.
#define TAG_NOTIFY  1

// Store the MPIRequest for a name, and return whether the total count of
// MPIRequests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(
        std::unique_ptr<MessageTable>& message_table,
        MPIRequest msg, int mpi_size) {
    auto name = msg.tensor_name();
    auto table_iter = message_table->find(name);
    if(table_iter == message_table->end()) {
        message_table->emplace(name, std::vector<MPIRequest>({msg}));
        table_iter = message_table->find(name);
    } else {
        table_iter->second.push_back(msg);
    }

    int count = table_iter->second.size();
    return count == mpi_size;
}

// Once a tensor is ready to be reduced, the coordinator sends an MPIResponse
// instructing all ranks to start the reduction to all ranks. The MPIResponse
// also contains error messages in case the submitted MPIRequests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the MPIResponse, thus, requires a whole lot of error checking.
MPIResponse ConstructMPIResponse(std::unique_ptr<MessageTable>& message_table, std::string name) {
    bool error = false;
    auto it =  message_table->find(name);
    assert(it != message_table->end());

    std::vector<MPIRequest> requests = it->second;
    assert(requests.size() > 0);

    std::ostringstream error_message_stream;

    // Check that all data types being reduced or gathered are identical
    auto data_type = requests[0].tensor_type();
    for(unsigned int i = 1; i < requests.size(); i++) {
        auto request_type = requests[i].tensor_type();
        if(data_type != request_type) {
            error = true;
            error_message_stream 
                << "Mismatched data types: One rank had type "
                << MPIDataType_Name(data_type)
                << ", but another rank had type "
                << MPIDataType_Name(request_type)
                << ".";
            break;
        }
    }

    // Check that all requested operations are the same
    auto message_type = requests[0].request_type();
    for(unsigned int i = 1; i < requests.size(); i++) {
        if(error) {
            break;
        }

        auto request_type = requests[i].request_type();
        if(message_type != request_type) {
            error = true;
            error_message_stream 
                << "Mismatched MPI operations: One rank did an "
                << message_type 
                << ", but another rank did an "
                << request_type
                << ".";
            break;
        }
    }

    // If we are doing an allreduce, check that all tensor shapes are identical
    if(message_type == MPIRequest::ALLREDUCE) {
        TensorShape tensor_shape;
        for(auto it = requests[0].tensor_shape().begin();
            it != requests[0].tensor_shape().end(); it++) {
            tensor_shape.AddDim(*it);
        }
        for(unsigned int i = 1; i < requests.size(); i++) {
            if(error) {
                break;
            }

            TensorShape request_shape;
            for(auto it = requests[i].tensor_shape().begin();
                it != requests[i].tensor_shape().end(); it++) {
                request_shape.AddDim(*it);
            }
            if(tensor_shape != request_shape) {
                error = true;
                error_message_stream 
                    << "Mismatched allreduce tensor shapes: "
                    << "One rank reduced a tensor of shape "
                    << tensor_shape.DebugString()
                    << ", but another rank sent a tensor of shape "
                    << request_shape.DebugString()
                    << ".";
                break;
            }
        }
    } 

    // If we are doing an allgather, make sure all but the first dimension are
    // the same. The first dimension may be different and the output tensor is
    // the sum of the first dimension. Collect the sizes by rank.
    std::vector<size_t> tensor_sizes(requests.size());
    if(message_type == MPIRequest::ALLGATHER) {
        TensorShape tensor_shape;
        for(auto it = requests[0].tensor_shape().begin();
            it != requests[0].tensor_shape().end(); it++) {
            tensor_shape.AddDim(*it);
        }

        if(tensor_shape.dims() == 0) {
            error = true;
            error_message_stream 
                << "Rank zero tried to gather a rank-zero tensor.";
        } else {
            tensor_sizes[requests[0].request_rank()] = size_t(tensor_shape.dim_size(0));
        }

        for(unsigned int i = 1; i < requests.size(); i++) {
            if(error) {
                break;
            }

            TensorShape request_shape;
            for(auto it = requests[i].tensor_shape().begin();
                it != requests[i].tensor_shape().end(); it++) {
                request_shape.AddDim(*it);
            }
            if(tensor_shape.dims() != request_shape.dims()) {
                error = true;
                error_message_stream 
                    << "Mismatched allgather tensor shapes: "
                    << "One rank gathered a tensor of rank "
                    << tensor_shape.dims()
                    << ", but another rank sent a tensor of rank "
                    << request_shape.dims()
                    << ".";
                break;
            }

            bool dim_mismatch = false;
            for(unsigned int dim = 1; dim < tensor_shape.dims(); dim++) {
                if(tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
                    error = true;
                    error_message_stream 
                        << "Mismatched allgather tensor shapes: "
                        << "One rank gathered a tensor with dimension "
                        << dim << " equal to " << tensor_shape.dim_size(dim)
                        << ", but another rank sent a tensor with dimension "
                        << dim << " equal to " << request_shape.dim_size(dim)
                        << ".";
                    dim_mismatch = true;
                    break;
                }
            }
            if(dim_mismatch) {
                break;
            }

            tensor_sizes[requests[i].request_rank()] = size_t(request_shape.dim_size(0));
        }
    }

    MPIResponse response;
    response.set_tensor_name(name);
    if(error) {
        std::string error_message = error_message_stream.str();
        response.set_response_type(MPIResponse::ERROR);
        response.set_error_message(error_message);
    } else if(message_type == MPIRequest::ALLGATHER) {
        response.set_response_type(MPIResponse::ALLGATHER);
        for(auto dim : tensor_sizes) {
            response.add_tensor_sizes(dim);
        }
    } else if(message_type == MPIRequest::ALLREDUCE) {
        response.set_response_type(MPIResponse::ALLREDUCE);
    }

    // Clear all queued up requests for this name. They are now taken care of
    // by the constructed MPI response.
    message_table->erase(it);

    return response;
}

// Process an MPIResponse by doing a reduction, a gather, or raising an error.
void PerformReductionOrGather(TensorTable& tensor_table, MPIResponse response) {
    // We should never fail at finding this key in the tensor table.
    auto name = response.tensor_name();
    auto iter = tensor_table.find(name);
    assert(iter != tensor_table.end());

    assert(response.response_type() == MPIResponse::ALLREDUCE ||
           response.response_type() == MPIResponse::ALLGATHER ||
           response.response_type() == MPIResponse::ERROR);

    Tensor tensor;
    OpKernelContext* context;
    CommunicationDoneCallback callback;
    bool on_gpu;
    std::tie(tensor, context, on_gpu, callback) = iter->second;

    // Clear the tensor table of this tensor and its callbacks; the rest of
    // this function takes care of it.
    tensor_table.erase(iter);

    Tensor output;
    Status status;
    if(response.response_type() == MPIResponse::ALLGATHER) {
        // Copy tensor sizes from the MPI response into a vector of size_t
        std::vector<size_t> tensor_sizes;
        for(auto it = response.tensor_sizes().begin();
            it != response.tensor_sizes().end(); it++) {
            tensor_sizes.push_back(size_t(*it));
        }

        if(tensor.dtype() == DT_FLOAT) {
            status = on_gpu ? RingAllgather<GPUDevice, float>(context, tensor, &output, tensor_sizes)
                            : RingAllgather<CPUDevice, float>(context, tensor, &output, tensor_sizes);
        } else if(tensor.dtype() == DT_INT32) {
            status = on_gpu ? RingAllgather<GPUDevice, int>(context, tensor, &output, tensor_sizes)
                            : RingAllgather<CPUDevice, int>(context, tensor, &output, tensor_sizes);
        } else {
            status = errors::Unknown("Invalid tensor type for MPI allgather.");
        }
    } else if(response.response_type() == MPIResponse::ALLREDUCE) {
        if(tensor.dtype() == DT_FLOAT) {
            status = on_gpu ? RingAllreduce<GPUDevice, float>(context, tensor, &output)
                            : RingAllreduce<CPUDevice, float>(context, tensor, &output);
        } else if(tensor.dtype() == DT_INT32) {
            status = on_gpu ? RingAllreduce<GPUDevice, int>(context, tensor, &output)
                            : RingAllreduce<CPUDevice, int>(context, tensor, &output);
        } else {
            status = errors::Unknown("Invalid tensor type for MPI allreduce.");
        }
    } else if(response.response_type() == MPIResponse::ERROR) {
        status = errors::FailedPrecondition(response.error_message());
    }

    if(status.ok()) {
        callback(StatusOr<Tensor>(output));
    } else {
        callback(StatusOr<Tensor>(status));
    }
}

// The MPI background thread loop coordinates all the MPI processes and the
// tensor reductions. The design of the communicator mechanism is limited by a few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a single thread.
//      Since TensorFlow may use several threads for graph processing, this means we must have
//      our own dedicated thread for dealing with MPI.
//      2. We want to gracefully handle errors, when MPI processes do not properly agree upon
//      what should happen (such as mismatched types or shapes). To do so requires the MPI processes
//      to know about the shapes and types of the relevant tensors on the other processes.
//      3. The MPI reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the MPI processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never make
//      progress if we have a thread pool limit.
//
// The coordinator currently follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each rank runs its own background thread which progresses in ticks.
// In each tick, the following actions happen:
//
//      a) The workers send an MPIRequest to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the MPIRequests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table. The
//      coordinator continues to receive MPIRequest messages until it has
//      received MPI_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of those,
//      it sends an MPIResponse to all the workers. When no more MPIResponses
//      are available, it sends an empty "DONE" response to the workers.
//
//      e) The workers listen for MPIResponse messages, processing each one by
//      doing the required reduce or gather, until they receive an empty "DONE"
//      message from the coordinator. At that point, the tick ends.
void BackgroundThreadLoop(MPIGlobalState& state) {
    // Initialize MPI. This must happen on the background thread, since not all
    // MPI implementations support being called from multiple threads.
    auto init_result = MPI_Init(NULL, NULL);
    if(init_result != MPI_SUCCESS) {
        state.init_status = errors::Unknown("Could not initialize MPI; MPI_Init() failed.");
    } else {
        state.init_status = Status::OK();
    }

    // Get MPI rank to determine if we are rank zero.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool is_coordinator = rank == 0;

    // Get MPI size to determine how many tensors to wait for before reducing.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    state.rank = rank;
    state.size = size;
    state.initialization_done = true;

    // Initialize the tensor count table. No tensors are available yet.
    if(is_coordinator) {
        state.message_table =
            std::unique_ptr<MessageTable>(new MessageTable());
    }

    // The coordinator sends a SHUTDOWN message to trigger shutdown.
    bool should_shut_down = false;
    do {
        // This delay determines thread frequency and MPI message latency
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // The rest of this loop happens under the MPI lock
        std::lock_guard<std::mutex> guard(state.mutex);

        // Collect all tensors that are ready to be reduced. Record them in the
        // tensor count table (rank zero) or send them to rank zero to be
        // recorded (everyone else).
        std::vector<std::string> ready_to_reduce;
        while(!state.message_queue.empty()) {
            // Pop the first available message message
            MPIRequest message = state.message_queue.front();
            state.message_queue.pop();

            if(is_coordinator) {
                bool reduce = IncrementTensorCount(state.message_table, message, size);
                if(reduce) {
                    ready_to_reduce.push_back(message.tensor_name());
                }
            } else {
                std::string encoded_message;
                message.SerializeToString(&encoded_message);
                MPI_Send(encoded_message.c_str(), encoded_message.length() + 1,
                         MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
            }
        }

        // Rank zero has put all its own tensors in the tensor count table.
        // Now, it should count all the tensors that are coming from other
        // ranks at this tick. It should keep getting tensors until it gets a
        // DONE message from all the other ranks.
        if(is_coordinator) {
            // Count of DONE messages. Keep receiving messages until the number
            // of messages is equal to the number of processes. Initialize to
            // one since the coordinator is effectively done.
            int completed_ranks = 1;
            while(completed_ranks != size) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, TAG_NOTIFY, MPI_COMM_WORLD, &status);

                // Find number of characters in message (including zero byte).
                int source_rank = status.MPI_SOURCE;
                int msg_length;
                MPI_Get_count(&status, MPI_BYTE, &msg_length);

                // If the length is zero, this is a DONE message.
                if(msg_length == 0) {
                    completed_ranks++;
                    MPI_Recv(NULL, 0, MPI_BYTE, source_rank, TAG_NOTIFY, MPI_COMM_WORLD, &status);
                    continue;
                }

                // Get tensor name from MPI into an std::string.
                char* buffer = new char[msg_length];
                MPI_Recv(buffer, msg_length, MPI_BYTE, source_rank,
                         TAG_NOTIFY, MPI_COMM_WORLD, &status);
                std::string received_data(buffer);
                delete[] buffer;

                MPIRequest received_message;
                received_message.ParseFromString(received_data);
                auto received_name = received_message.tensor_name();

                bool reduce = IncrementTensorCount(
                        state.message_table, received_message, size);
                if(reduce) {
                    ready_to_reduce.push_back(received_name);
                }
            }

            // At this point, rank zero should have a fully updated tensor count
            // table and should know all the tensors that need to be reduced or
            // gathered, and everyone else should have sent all their information
            // to rank zero. We can now do reductions and gathers; rank zero will
            // choose which ones and in what order, and will notify the other ranks
            // before doing each reduction.
            for(int i = 0; i < ready_to_reduce.size(); i++) {
                // Notify all nodes which tensor we'd like to reduce at this step.
                auto name = ready_to_reduce[i];
                MPIResponse response = ConstructMPIResponse(state.message_table, name);

                std::string encoded_response;
                response.SerializeToString(&encoded_response);
                for(int r = 1; r < size; r++) {
                    MPI_Send(encoded_response.c_str(), encoded_response.length() + 1,
                             MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
                }

                // Perform the reduction. All nodes should end up performing the same reduction.
                PerformReductionOrGather(state.tensor_table, response);
            }

            // Notify all nodes that we are done with the reductions for this tick.
            MPIMessage done_response;
            should_shut_down = state.shut_down;
            done_response.set_response_type(
                     should_shut_down ? MPIResponse::SHUTDOWN : MPIResponse::DONE);
            std::string encoded_response;
            done_response.SerializeToString(&encoded_response);
            for(int r = 1; r < size; r++) {
                MPI_Send(encoded_response.c_str(), encoded_response.length() + 1,
                         MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
            }
        } else {
            // Notify the coordinator that this node is done sending messages.
            // A DONE message is encoded as a zero-length message.
            MPI_Send(NULL, 0, MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);

            // Receive names for tensors to reduce from rank zero.
            // Once we receive a empty DONE message, stop waiting for more names.
            while(true) {
                MPI_Status status;
                MPI_Probe(0, TAG_NOTIFY, MPI_COMM_WORLD, &status);

                // Find number of characters in message (including zero byte).
                int msg_length;
                MPI_Get_count(&status, MPI_BYTE, &msg_length);

                // Get tensor name from MPI into an std::string.
                char* buffer = new char[msg_length];
                MPI_Recv(buffer, msg_length, MPI_BYTE, 0,
                         TAG_NOTIFY, MPI_COMM_WORLD, &status);
                std::string received_message(buffer);
                delete[] buffer;

                MPIResponse response;
                response.ParseFromString(received_message);
                if(response.response_type() == MPIMessage::DONE) {
                    // No more messages this tick
                    break;
                } else if(response.response_type() == MPIMessage::SHUTDOWN) {
                    // No more messages this tick, and the background thread should shut down
                    should_shut_down = true;
                } else {
                    // Process the current message
                    PerformReductionOrGather(state.tensor_table, response);
                }
            }
        }
    } while(!should_shut_down);

    MPI_Finalize();
}

// Initialize MPI and start the MPI background thread. Ensure that this is
// only done once no matter how many times this function is called.
Status InitializeMPIOnce() {
    // Ensure MPI is only initialized once.
    if(mpi_global.initialized_flag.test_and_set())
        return mpi_global.init_status;

    // Start the MPI background thread, which assumes MPI is initialized
    mpi_global.background_thread = std::thread(BackgroundThreadLoop, std::ref(mpi_global));

    // Wait to ensure that the background thread has finished initializing MPI. 
    while(!mpi_global.initialization_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return mpi_global.init_status;
}

// Convert a TensorFlow DataType to our MPIDataType.
Status DataTypeToMPIType(DataType tf_dtype, MPIDataType* mpi_dtype) {
    if(tf_dtype == DT_FLOAT) {
        *mpi_dtype = TF_MPI_FLOAT32;
    } else if(tf_dtype == DT_INT32) {
        *mpi_dtype = TF_MPI_INT32;
    } else {
        return errors::FailedPrecondition("Invalid tensor type passed.");
    }
    return Status::OK();
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllreduce(
        OpKernelContext* context,
        const Tensor& tensor,
        const std::string name,
        const bool on_gpu,
        CommunicationDoneCallback callback) {
    MPIDataType dtype;
    Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
    if(!status.ok()) {
        callback(StatusOr<Tensor>(status));
        return;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPIRequest message;
    message.set_request_rank(rank);
    message.set_tensor_name(name);
    message.set_tensor_type(dtype);
    message.set_request_type(MPIRequest::ALLREDUCE);
    for(int i = 0; i < tensor.shape().dims(); i++) {
        message.add_tensor_shape(tensor.shape().dim_size(i));
    }

    std::lock_guard<std::mutex> guard(mpi_global.mutex);
    std::tuple<Tensor, OpKernelContext*, bool, CommunicationDoneCallback> record(tensor, context, on_gpu, callback);
    mpi_global.tensor_table.emplace(name, record);
    mpi_global.message_queue.push(message);
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllgather(
        OpKernelContext* context,
        const Tensor& tensor,
        const std::string name,
        const bool on_gpu,
        CommunicationDoneCallback callback) {
    MPIDataType dtype;
    Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
    if(!status.ok()) {
        callback(StatusOr<Tensor>(status));
        return;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPIRequest message;
    message.set_request_rank(rank);
    message.set_tensor_name(name);
    message.set_tensor_type(dtype);
    message.set_request_type(MPIRequest::ALLGATHER);
    for(int i = 0; i < tensor.shape().dims(); i++) {
        message.add_tensor_shape(tensor.shape().dim_size(i));
    }

    std::lock_guard<std::mutex> guard(mpi_global.mutex);
    std::tuple<Tensor, OpKernelContext*, bool, CommunicationDoneCallback> record(tensor, context, on_gpu, callback);
    mpi_global.tensor_table.emplace(name, record);
    mpi_global.message_queue.push(message);
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
    int world_size = mpi_global.size;

    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0) = world_size;
  }
};

REGISTER_KERNEL_BUILDER(Name("MPISize").Device(DEVICE_CPU), MPISizeOp);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPISize").Device(DEVICE_GPU), MPISizeOp);
#endif

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
    int rank = mpi_global.rank;

    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<int>();
    flat(0) = rank;
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIRank").Device(DEVICE_CPU), MPIRankOp);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPIRank").Device(DEVICE_GPU), MPIRankOp);
#endif

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

// Op to get the current MPI Rank.
template <typename Device>
class MPIAllreduceOp : public AsyncOpKernel {
 public:
  explicit MPIAllreduceOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
      bool on_gpu = IsGPUDevice<Device>();
      auto device_context = context->op_device_context();
      auto node_name = name();
      auto callback = [node_name, done, context] {
        auto tensor = context->input(0);
        EnqueueTensorAllreduce(context, tensor, node_name, on_gpu,
                               [node_name, done, context](StatusOr<Tensor> status) {
            if(status.ok()) {
                Tensor output = status.ValueOrDie();
                context->set_output(0, output);
            }
            context->SetStatus(status.status());
            done();
        });
      };

      // If we are on a CPU, our device context will be null and we can't
      // get a stream to enqueue this on. On a CPU this op is called when the
      // data is already available, so we can just immediately do the allreduce;
      // we don't have to wait for the data to get populated.
      if(device_context == nullptr) {
          callback();
      } else {
        auto stream = device_context->stream();
        stream->ThenDoHostCallback(callback);
      }
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIAllreduce").Device(DEVICE_CPU), MPIAllreduceOp<CPUDevice>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPIAllreduce").Device(DEVICE_GPU), MPIAllreduceOp<GPUDevice>);
#endif

REGISTER_OP("MPIAllreduce")
    .Attr("T: {int32, float32}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

template <typename Device>
class MPIAllgatherOp : public AsyncOpKernel {
 public:
  explicit MPIAllgatherOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
      bool on_gpu = IsGPUDevice<Device>();
      auto device_context = context->op_device_context();
      auto node_name = name();
      auto callback = [node_name, done, context] {
        auto tensor = context->input(0);
        EnqueueTensorAllgather(context, tensor, node_name, on_gpu,
                               [node_name, done, context](StatusOr<Tensor> status) {
            if(status.ok()) {
                Tensor output = status.ValueOrDie();
                context->set_output(0, output);
            }
            context->SetStatus(status.status());
            done();
        });
      };

      // If we are on a CPU, our device context will be null and we can't
      // get a stream to enqueue this on. On a CPU this op is called when the
      // data is already available, so we can just immediately do the allgather;
      // we don't have to wait for the data to get populated.
      if(device_context == nullptr) {
          callback();
      } else {
        auto stream = device_context->stream();
        stream->ThenDoHostCallback(callback);
      }
  }
};

REGISTER_KERNEL_BUILDER(Name("MPIAllgather").Device(DEVICE_CPU), MPIAllgatherOp<CPUDevice>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPIAllgather").Device(DEVICE_GPU), MPIAllgatherOp<GPUDevice>);
#endif

REGISTER_OP("MPIAllgather")
    .Attr("T: {int32, float32}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.

Output
    gathered:    A tensor with the same shape as `tensor` except for the first dimension.
)doc");

}  // namespace mpi
}  // namespace contrib
}  // namespace tensorflow
