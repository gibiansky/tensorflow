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

template<class T>
using StatusOr = perftools::gputools::port::StatusOr<T>;

namespace tensorflow {
namespace mpi {

namespace {
// What type of communication is being done between the nodes.
enum MPIMessageType {
    MPIAllreduce = 0,
    MPIAllgather = 1
};

// What type of communication is being done between the nodes.
enum MPIMessageDataType {
    MPIFloat = 0,
    MPIInt = 1
};

// A MPI communication summary
struct MPIMessage {
    MPIMessageType message_type;
    MPIMessageDataType data_type;
    TensorShape data_shape;
    std::string tensor_name;
};

// A callback to call after the MPI communication completes.
typedef std::function<void(StatusOr<Tensor>)> CommunicationDoneCallback;

// Table storing Tensors to be reduced, keyed by unique name
typedef std::unordered_map<std::string, std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> > TensorTable;

struct MPIGlobalState {
    // An atomic boolean which is set to true when MPI is initialized.
    // This ensures that MPI_Init is never called twice.
    std::atomic_flag initialized_flag;

    // A mutex that needs to be used whenever MPI operations are done.
    std::mutex mutex;

    // Tensors waiting to be allreduced or allgathered
    TensorTable tensor_table;

    // Queue of MPI messages waiting to be sent
    std::queue<MPIMessage> message_queue;

    // Background thread running MPI communication
    std::thread background_thread;

    // Whether the background thread should shutdown.
    bool shut_down;

    // Only exists on the coordinator node (rank zero). Maintains a count of
    // how many nodes are ready to allreduce every tensor (keyed by tensor
    // name).
    std::unique_ptr<std::unordered_map<std::string, int> > tensor_counts;

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


// Convert Tensorflow data type into MPIMessageDataType
Status TensorMPIDataType(const Tensor& tensor, MPIMessageDataType* data_type) {
    auto type = tensor.dtype();
    switch(type) {
        case DT_FLOAT:
            *data_type = MPIFloat;
            return Status::OK();
        default:
            return errors::FailedPrecondition("MPI operation on unsupported data type");
    }
}

#define RANK_ZERO   0
#define TAG_NOTIFY  1

// Increment the tensor count for a name (or set it to one if it doesn't
// exist), and return whether the total count is now equal to the MPI size (and
// thus we are ready to reduce the tensor).
bool IncrementTensorCount(
        std::unique_ptr<std::unordered_map<std::string, int> >& tensor_counts,
        std::string name, int mpi_size) {
    auto count_iter = tensor_counts->find(name);
    int count = count_iter == tensor_counts->end() ? 1 : count_iter->second + 1;
    (*tensor_counts)[name] = count;

    return count == mpi_size;
}

void PerformReductionOrGather(TensorTable& tensor_table, std::string name) {
    // We should never fail at finding this key in the tensor table.
    auto iter = tensor_table.find(name);
    assert(iter != tensor_table.end());

    Tensor tensor;
    OpKernelContext* context;
    CommunicationDoneCallback callback;
    std::tie(tensor, context, callback) = iter->second;

    Tensor output;
    Status status = RingAllreduce<float>(context, tensor, &output);
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
        state.tensor_counts =
            std::unique_ptr<std::unordered_map<std::string, int> >(
                    new std::unordered_map<std::string, int>());
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
            MPIMessage message = state.message_queue.front();
            state.message_queue.pop();

            auto name = message.tensor_name;
            if(is_coordinator) {
                bool reduce = IncrementTensorCount(state.tensor_counts, name, size);
                if(reduce) {
                    ready_to_reduce.push_back(name);
                }
            } else {
                MPI_Send(name.c_str(), name.length() + 1, MPI_BYTE,
                         RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
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
                int name_length;
                MPI_Get_count(&status, MPI_BYTE, &name_length);

                // If the length is zero, this is a DONE message.
                if(name_length == 0) {
                    completed_ranks++;
                    continue;
                }

                // Get tensor name from MPI into an std::string.
                char* buffer = new char[name_length];
                MPI_Recv(buffer, name_length, MPI_BYTE, source_rank,
                        TAG_NOTIFY, MPI_COMM_WORLD, &status);
                std::string received_name(buffer);
                delete[] buffer;

                bool reduce = IncrementTensorCount(state.tensor_counts, received_name, size);
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
                for(int r = 1; r < size; r++) {
                    MPI_Send(name.c_str(), name.length() + 1, MPI_BYTE,
                             r, TAG_NOTIFY, MPI_COMM_WORLD);
                }

                // Perform the reduction. All nodes should end up performing the same reduction.
                PerformReductionOrGather(state.tensor_table, name);
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
                int name_length;
                MPI_Get_count(&status, MPI_BYTE, &name_length);

                // If the length is zero, this is a DONE message.
                if(name_length == 0) {
                    break;
                }

                // Get tensor name from MPI into an std::string.
                char* buffer = new char[name_length];
                MPI_Recv(buffer, name_length, MPI_BYTE, 0,
                         TAG_NOTIFY, MPI_COMM_WORLD, &status);
                std::string received_name(buffer);
                delete[] buffer;

                PerformReductionOrGather(state.tensor_table, received_name);
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

void EnqueueTensorAllreduce(
        OpKernelContext* context,
        const Tensor& tensor,
        const std::string name,
        CommunicationDoneCallback callback) {
    // Ensure that the MPI thread is running
    InitializeMPIOnce();

    MPIMessage message;
    message.data_shape = tensor.shape();
    message.message_type = MPIAllreduce;
    message.tensor_name = name;

    auto status = TensorMPIDataType(tensor, &message.data_type);
    if(!status.ok()) {
        callback(status);
        return;
    }

    std::lock_guard<std::mutex> guard(mpi_global.mutex);
    std::tuple<Tensor, OpKernelContext*, CommunicationDoneCallback> record(tensor, context, callback);
    mpi_global.tensor_table.emplace(name, record);
    mpi_global.message_queue.push(message);
}

void EnqueueTensorAllgather(
        const Tensor& tensor,
        const std::string name,
        int dimension,
        CommunicationDoneCallback callback) {
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

REGISTER_OP("MPIAllreduce")
    .Input("tensor: float32")
    .Output("reduced: float32")
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
    reduced:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

}  // namespace mpi
}  // namespace tensorflow
