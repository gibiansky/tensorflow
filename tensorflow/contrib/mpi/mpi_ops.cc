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

#include "third_party/mpi/mpi.h"
#include "tensorflow/contrib/mpi/ring.h"
#include "tensorflow/contrib/mpi/mpi_message.pb.h"

template<class T>
using StatusOr = perftools::gputools::port::StatusOr<T>;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow {
namespace contrib {
namespace mpi {

namespace {

// A callback to call after the MPI communication completes.
typedef std::function<void(StatusOr<Tensor>)> CommunicationDoneCallback;

// Table storing Tensors to be reduced, keyed by unique name
typedef std::unordered_map<std::string, std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> > TensorTable;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking and size calculations.
typedef std::unordered_map<std::string, std::vector<MPIRequest> > MessageTable;

struct MPIGlobalState {
    // An atomic boolean which is set to true when MPI is initialized.
    // This ensures that MPI_Init is never called twice.
    std::atomic_flag initialized_flag;

    // A mutex that needs to be used whenever MPI operations are done.
    std::mutex mutex;

    // Tensors waiting to be allreduced or allgathered
    TensorTable tensor_table;

    // Queue of MPI messages waiting to be sent
    std::queue<MPIRequest> message_queue;

    // Background thread running MPI communication
    std::thread background_thread;

    // Whether the background thread should shutdown.
    bool shut_down;

    // Only exists on the coordinator node (rank zero). Maintains a count of
    // how many nodes are ready to allreduce every tensor (keyed by tensor
    // name).
    std::unique_ptr<MessageTable> message_table;

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
    .initialized_flag = ATOMIC_FLAG_INIT
};


#define RANK_ZERO   0
#define TAG_NOTIFY  1

// Increment the tensor count for a name (or set it to one if it doesn't
// exist), and return whether the total count is now equal to the MPI size (and
// thus we are ready to reduce the tensor).
bool IncrementTensorCount(
        std::unique_ptr<MessageTable>& message_table,
        MPIRequest msg, int mpi_size) {
    auto name = msg.tensor_name();
    auto table_iter = message_table->find(name);
    if(table_iter == message_table->end()) {
        message_table->emplace(name, std::vector<MPIRequest>({msg}));
        table_iter = message_table->find(name);
    }

    table_iter->second.push_back(msg);
    int count = table_iter->second.size();

    return count == mpi_size;
}

MPIResponse ConstructMPIResponse(
        std::unique_ptr<MessageTable>& message_table,
        std::string name, bool* error) {
    auto it =  message_table->find(name);
    assert(it != message_table->end());

    std::vector<MPIRequest> requests = it->second;
    assert(requests.size() > 0);

    *error = false;
    std::ostringstream error_message_stream;

    // Check that all data types being reduced or gathered are identical
    auto data_type = requests[0].tensor_type();
    for(unsigned int i = 1; i < requests.size(); i++) {
        auto request_type = requests[i].tensor_type();
        if(data_type != request_type) {
            *error = true;
            error_message_stream 
                << "Mismatched data types: Rank 0 had type "
                << data_type 
                << ", but rank "
                << i
                << "had type "
                << request_type
                << ".";
            break;
        }
    }

    // Check that all requested operations are the same
    auto message_type = requests[0].request_type();
    for(unsigned int i = 1; i < requests.size(); i++) {
        auto request_type = requests[i].request_type();
        if(message_type != request_type) {
            *error = true;
            error_message_stream 
                << "Mismatched MPI operations: Rank 0 did an "
                << message_type 
                << ", but rank "
                << i
                << "did an "
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
            TensorShape request_shape;
            for(auto it = requests[i].tensor_shape().begin();
                it != requests[i].tensor_shape().end(); it++) {
                tensor_shape.AddDim(*it);
            }
            if(tensor_shape != request_shape) {
                *error = true;
                error_message_stream 
                    << "Mismatched allreduce tensor shapes: "
                    << "Rank 0 reduced a tensor of shape "
                    << tensor_shape.DebugString()
                    << ", but rank "
                    << i
                    << "sent a tensor of shape "
                    << request_shape.DebugString()
                    << ".";
                break;
            }
        }
    } 
    // If we are doing an allgather, make sure all but the first dimension are
    // the same. The first dimension may be different and the output tensor is
    // the sum of the first dimension. Collect the sizes by rank.
    std::vector<size_t> tensor_sizes;
    if(message_type == MPIRequest::ALLGATHER) {
        TensorShape tensor_shape;
        for(auto it = requests[0].tensor_shape().begin();
            it != requests[0].tensor_shape().end(); it++) {
            tensor_shape.AddDim(*it);
        }

        if(tensor_shape.dims() == 0) {
            *error = true;
            error_message_stream 
                << "Rank zero tried to gather a rank-zero tensor.";
        } else {
            tensor_sizes.push_back(size_t(tensor_shape.dim_size(0)));
        }

        for(unsigned int i = 1; i < requests.size(); i++) {
            if(*error) {
                break;
            }

            TensorShape request_shape;
            for(auto it = requests[i].tensor_shape().begin();
                it != requests[i].tensor_shape().end(); it++) {
                tensor_shape.AddDim(*it);
            }
            if(tensor_shape.dims() != request_shape.dims()) {
                *error = true;
                error_message_stream 
                    << "Mismatched allgather tensor shapes: "
                    << "Rank 0 gathered a tensor of rank "
                    << tensor_shape.dims()
                    << ", but rank "
                    << i
                    << "sent a tensor of rank "
                    << request_shape.dims()
                    << ".";
                break;
            }

            bool dim_mismatch = false;
            for(unsigned int dim = 1; dim < tensor_shape.dims(); dim++) {
                if(tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
                    *error = true;
                    error_message_stream 
                        << "Mismatched allgather tensor shapes: "
                        << "Rank 0 gathered a tensor with dimension "
                        << dim << " equal to " << tensor_shape.dim_size(dim)
                        << ", but rank "
                        << i
                        << "sent a tensor with dimension "
                        << dim << " equal to " << request_shape.dim_size(dim)
                        << ".";
                    dim_mismatch = true;
                    break;
                }
            }
            if(dim_mismatch) {
                break;
            }

            tensor_sizes.push_back(size_t(tensor_shape.dim_size(0)));
        }
    }

    std::string error_message = error_message_stream.str();

    MPIResponse response;
    response.set_tensor_name(name);
    if(*error) {
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


    return response;
}

void ReportReductionError(TensorTable& tensor_table, MPIResponse response) {
    // We should never fail at finding this key in the tensor table.
    auto name = response.tensor_name();
    auto iter = tensor_table.find(name);
    assert(iter != tensor_table.end());

    Tensor tensor;
    OpKernelContext* context;
    CommunicationDoneCallback callback;
    std::tie(tensor, context, callback) = iter->second;

    callback(StatusOr<Tensor>(errors::Unknown(response.error_message())));
}

void PerformReductionOrGather(TensorTable& tensor_table, MPIResponse response) {
    // We should never fail at finding this key in the tensor table.
    auto name = response.tensor_name();
    auto iter = tensor_table.find(name);
    assert(iter != tensor_table.end());

    assert(response.response_type() == MPIResponse::ALLREDUCE ||
           response.response_type() == MPIResponse::ALLGATHER);

    Tensor tensor;
    OpKernelContext* context;
    CommunicationDoneCallback callback;
    std::tie(tensor, context, callback) = iter->second;

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
            status = RingAllgather<CPUDevice, float>(context, tensor, &output, tensor_sizes);
        } else if(tensor.dtype() == DT_INT32) {
            status = RingAllgather<CPUDevice, int>(context, tensor, &output, tensor_sizes);
        } else {
            status = errors::Unknown("Invalid tensor type for MPI allgather.");
        }
    } else if(response.response_type() == MPIResponse::ALLREDUCE) {
        if(tensor.dtype() == DT_FLOAT) {
            status = RingAllreduce<CPUDevice, float>(context, tensor, &output);
        } else if(tensor.dtype() == DT_INT32) {
            status = RingAllreduce<CPUDevice, int>(context, tensor, &output);
        } else {
            status = errors::Unknown("Invalid tensor type for MPI allreduce.");
        }
    }

    if(status.ok()) {
        callback(StatusOr<Tensor>(output));
    } else {
        callback(StatusOr<Tensor>(status));
    }
}

// The MPI background thread loop coordinates all the MPI processes and the
// tensor reductions.
void BackgroundThreadLoop(MPIGlobalState& state) {
    // Get MPI rank to determine if we are rank zero.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool is_coordinator = rank == 0;

    // Get MPI size to determine how many tensors to wait for before reducing.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the tensor count table. No tensors are available yet.
    if(is_coordinator) {
        state.message_table =
            std::unique_ptr<MessageTable>(new MessageTable());
    }

    while(!state.shut_down) {
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

        // Notify the coordinator that this node is done sending messages.
        if(!is_coordinator) {
            // A DONE message is encoded as a zero-length message.
            MPI_Send(NULL, 0, MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
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
        }

        // At this point, rank zero should have a fully updated tensor count
        // table and should know all the tensors that need to be reduced or
        // gathered, and everyone else should have sent all their information
        // to rank zero. We can now do reductions and gathers; rank zero will
        // choose which ones and in what order, and will notify the other ranks
        // before doing each reduction.
        if(is_coordinator) {
            for(int i = 0; i < ready_to_reduce.size(); i++) {
                // Notify all nodes which tensor we'd like to reduce at this step.
                auto name = ready_to_reduce[i];
                bool error;
                MPIResponse response = ConstructMPIResponse(
                        state.message_table, name, &error);

                std::string encoded_response;
                response.SerializeToString(&encoded_response);
                for(int r = 1; r < size; r++) {
                    MPI_Send(encoded_response.c_str(), encoded_response.length() + 1,
                             MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
                }

                // Perform the reduction. All nodes should end up performing
                // the same reduction.
                if(!error) {
                    PerformReductionOrGather(state.tensor_table, response);
                }
            }

            // Notify all nodes that we are done with the reductions for this tick.
            for(int r = 1; r < size; r++) {
                MPI_Send(NULL, 0, MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
            }
        } else {
            // Receive names for tensors to reduce from rank zero.
            // Once we receive a empty DONE message, stop waiting for more names.
            while(true) {
                MPI_Status status;
                MPI_Probe(0, TAG_NOTIFY, MPI_COMM_WORLD, &status);

                // Find number of characters in message (including zero byte).
                int msg_length;
                MPI_Get_count(&status, MPI_BYTE, &msg_length);

                // If the length is zero, this is a DONE message.
                if(msg_length == 0) {
                    break;
                }

                // Get tensor name from MPI into an std::string.
                char* buffer = new char[msg_length];
                MPI_Recv(buffer, msg_length, MPI_BYTE, 0,
                         TAG_NOTIFY, MPI_COMM_WORLD, &status);
                std::string received_message(buffer);
                delete[] buffer;

                MPIResponse response;
                response.ParseFromString(received_message);

                if(response.response_type() == MPIResponse::ERROR) {
                    ReportReductionError(state.tensor_table, response);
                } else {
                    PerformReductionOrGather(state.tensor_table, response);
                }
            }
        }
    }
}

// Initialize MPI and start the MPI background thread. Ensure that this is
// only done once no matter how many times this function is called.
Status InitializeMPIOnce() {
    // Ensure MPI is only initialized once.
    if(mpi_global.initialized_flag.test_and_set())
        return Status::OK();

    auto init_result = MPI_Init(NULL, NULL);
    if(init_result != MPI_SUCCESS) {
        return errors::Unknown("Could not initialize MPI; MPI_Init() failed.");
    }

    // Start the MPI background thread, which assumes MPI is initialized
    mpi_global.background_thread = std::thread(BackgroundThreadLoop, std::ref(mpi_global));

    return Status::OK();
}

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

void EnqueueTensorAllreduce(
        OpKernelContext* context,
        const Tensor& tensor,
        const std::string name,
        CommunicationDoneCallback callback) {
    // Ensure that the MPI thread is running
    InitializeMPIOnce();

    MPIDataType dtype;
    Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
    if(!status.ok()) {
        callback(StatusOr<Tensor>(status));
        return;
    }

    MPIRequest message;
    message.set_tensor_name(name);
    message.set_tensor_type(dtype);
    message.set_request_type(MPIRequest::ALLREDUCE);
    for(int i = 0; i < tensor.shape().dims(); i++) {
        message.add_tensor_shape(tensor.shape().dim_size(i));
    }

    std::lock_guard<std::mutex> guard(mpi_global.mutex);
    std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> record(tensor, context, callback);
    mpi_global.tensor_table.emplace(name, record);
    mpi_global.message_queue.push(message);
}

void EnqueueTensorAllgather(
        OpKernelContext* context,
        const Tensor& tensor,
        const std::string name,
        CommunicationDoneCallback callback) {
    // Ensure that the MPI thread is running
    InitializeMPIOnce();

    MPIDataType dtype;
    Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
    if(!status.ok()) {
        callback(StatusOr<Tensor>(status));
        return;
    }

    MPIRequest message;
    message.set_tensor_name(name);
    message.set_tensor_type(dtype);
    message.set_request_type(MPIRequest::ALLGATHER);
    for(int i = 0; i < tensor.shape().dims(); i++) {
        message.add_tensor_shape(tensor.shape().dim_size(i));
    }

    std::lock_guard<std::mutex> guard(mpi_global.mutex);
    std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> record(tensor, context, callback);
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

// Op to get the current MPI Rank.
class MPIAllreduceOp : public AsyncOpKernel {
 public:
  explicit MPIAllreduceOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
      auto device_context = context->op_device_context();
      auto node_name = name();
      auto callback = [node_name, done, context] {
        auto tensor = context->input(0);
        EnqueueTensorAllreduce(context, tensor, node_name, [node_name, done, context](StatusOr<Tensor> status) {
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

REGISTER_KERNEL_BUILDER(Name("MPIAllreduce").Device(DEVICE_CPU), MPIAllreduceOp);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPIAllreduce").Device(DEVICE_GPU), MPIAllreduceOp);
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

class MPIAllgatherOp : public AsyncOpKernel {
 public:
  explicit MPIAllgatherOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, InitializeMPIOnce());
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
      auto device_context = context->op_device_context();
      auto node_name = name();
      auto callback = [node_name, done, context] {
        auto tensor = context->input(0);
        EnqueueTensorAllgather(context, tensor, node_name, [node_name, done, context](StatusOr<Tensor> status) {
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

REGISTER_KERNEL_BUILDER(Name("MPIAllgather").Device(DEVICE_CPU), MPIAllgatherOp);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MPIAllgather").Device(DEVICE_GPU), MPIAllgatherOp);
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
