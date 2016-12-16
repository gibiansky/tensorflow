# Ops that communicate with other processes via MPI.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

cc_library(
    name = "mpi_cc",
    srcs = ["mpi_ops.cc", "ring.cc", "ring.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        "@protobuf//:protobuf",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "//third_party/eigen3",
        "//third_party/mpi:mpi",
        ":mpi_message_proto_cc",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "mpi.so",
    deps = [
        ":mpi_cc",
    ],
)

cc_library(
    name = "mpi_op_lib",
    deps = [
        ":mpi_cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "mpi_op_wrapper_py",
    require_shape_functions = True,
    visibility = ["//visibility:private"],
    deps = [
        ":mpi_cc",
    ],
)

tf_py_test(
    name = "mpi_test",
    srcs = ["mpi_op_test.py"],
    additional_deps = [
        ":mpi_op_wrapper_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    tags = ["manual"],
)

py_library(
    name = "mpi_ops_py",
    srcs = [
        "__init__.py",
        "mpi_ops.py",
    ],
    data = [":mpi.so"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":mpi_op_wrapper_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_proto_library(
    name = "mpi_message_proto",
    srcs = ["mpi_message.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
