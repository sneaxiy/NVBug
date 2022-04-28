// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "cudnn.h"

static bool IsError(bool err_code) { return !err_code; }

static std::string ToErrorCode(bool err_code) {
  return err_code ? "true" : "false";
}

static bool IsError(cudaError_t err_code) { return err_code != cudaSuccess; }

static std::string ToErrorCode(cudaError_t err_code) {
  return cudaGetErrorString(err_code);
}

static bool IsError(cudnnStatus_t err_code) {
  return err_code != CUDNN_STATUS_SUCCESS;
}

static std::string ToErrorCode(cudnnStatus_t err_code) {
  return cudnnGetErrorString(err_code);
}

#define ASSERT_CHECK(__cond)                                               \
  do {                                                                     \
    auto __err_code = (__cond);                                            \
    if (::IsError(__err_code)) {                                           \
      auto __err_msg = "`" + ::std::string(#__cond) + "` check failed: " + \
                       __FILE__ + ":" + ::std::to_string(__LINE__) +       \
                       " error code: " + ToErrorCode(__err_code);          \
      throw std::runtime_error(__err_msg);                                 \
    }                                                                      \
  } while (0)

template <typename T>
static T *CudaMalloc(size_t n) {
  T *ptr = nullptr;
  ASSERT_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  return ptr;
}

static void CudaFree(void *ptr) {
  if (ptr) {
    ASSERT_CHECK(cudaFree(ptr));
  }
}

enum Layout { kNCHW = 0, kNHWC = 1 };

template <typename T>
struct CuDNNDataType;

template <>
struct CuDNNDataType<float> {
  static constexpr auto kType = CUDNN_DATA_FLOAT;
  using MT = float;
};

template <>
struct CuDNNDataType<half> {
  static constexpr auto kType = CUDNN_DATA_HALF;
  using MT = float;
};

template <typename T>
static void BatchNormTest(
    int N, int C, int H, int W, Layout layout, cudnnBatchNormOps_t op) {
  constexpr auto kCudnnDType = CuDNNDataType<T>::kType;
  using MT = typename CuDNNDataType<T>::MT;

  size_t numel = static_cast<size_t>(N) * C * H * W;
  auto *x = CudaMalloc<T>(numel);
  auto *y = CudaMalloc<T>(numel);
  T *z = nullptr;
  if (op == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    z = CudaMalloc<T>(numel);
  }

  auto *scale = CudaMalloc<MT>(C);
  auto *bias = CudaMalloc<MT>(C);

  auto *saved_mean = CudaMalloc<MT>(C);
  auto *saved_inv_var = CudaMalloc<MT>(C);
  auto *running_mean = CudaMalloc<MT>(C);
  auto *running_var = CudaMalloc<MT>(C);

  MT alpha = 1.0;
  MT beta = 0.0;

  double factor = 0.5;
  double epsilon = 1e-6;
  epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

  cudnnActivationDescriptor_t act_desc = nullptr;
  if (op == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION ||
      op == CUDNN_BATCHNORM_OPS_BN_ACTIVATION) {
    ASSERT_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    ASSERT_CHECK(cudnnSetActivationDescriptor(
        act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }

  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  cudnnTensorDescriptor_t x_desc, param_desc;
  ASSERT_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  ASSERT_CHECK(cudnnCreateTensorDescriptor(&param_desc));

  auto y_desc = x_desc;
  cudnnTensorDescriptor_t z_desc = nullptr;
  if (op == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    z_desc = x_desc;
  }

  std::vector<int> dims;
  std::vector<int> strides;
  int D = 1;
  if (layout == kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }

  ASSERT_CHECK(cudnnSetTensorNdDescriptor(
      x_desc, kCudnnDType, 4, dims.data(), strides.data()));

  ASSERT_CHECK(cudnnDeriveBNTensorDescriptor(param_desc, x_desc, mode));

  cudnnHandle_t handle;
  cudaStream_t stream;
  ASSERT_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  ASSERT_CHECK(cudnnCreate(&handle));
  ASSERT_CHECK(cudnnSetStream(handle, stream));

  size_t workspace_size = 0;
  ASSERT_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle,
      mode,
      op,
      x_desc,
      z_desc,
      y_desc,
      param_desc,
      act_desc,
      &workspace_size));

  void *workspace = CudaMalloc<uint8_t>(workspace_size);

  size_t reserve_size = 0;
  ASSERT_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      handle, mode, op, act_desc, x_desc, &reserve_size));
  void *reserve_space = CudaMalloc<uint8_t>(reserve_size);

  ASSERT_CHECK(cudnnBatchNormalizationForwardTrainingEx(handle,
                                                        mode,
                                                        op,
                                                        &alpha,
                                                        &beta,
                                                        x_desc,
                                                        x,
                                                        z_desc,
                                                        z,
                                                        y_desc,
                                                        y,
                                                        param_desc,
                                                        scale,
                                                        bias,
                                                        factor,
                                                        running_mean,
                                                        running_var,
                                                        epsilon,
                                                        saved_mean,
                                                        saved_inv_var,
                                                        act_desc,
                                                        workspace,
                                                        workspace_size,
                                                        reserve_space,
                                                        reserve_size));

  ASSERT_CHECK(cudaStreamSynchronize(stream));
  CudaFree(x);
  CudaFree(y);
  CudaFree(z);
  CudaFree(scale);
  CudaFree(bias);
  CudaFree(running_mean);
  CudaFree(running_var);
  CudaFree(saved_mean);
  CudaFree(saved_inv_var);
  CudaFree(workspace);
  CudaFree(reserve_space);

  ASSERT_CHECK(cudnnDestroy(handle));
  ASSERT_CHECK(cudaStreamDestroy(stream));
  ASSERT_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  ASSERT_CHECK(cudnnDestroyTensorDescriptor(param_desc));
  if (act_desc) {
    ASSERT_CHECK(cudnnDestroyActivationDescriptor(act_desc));
  }
}

static std::string LayoutToString(Layout layout) {
  switch (layout) {
    case kNHWC:
      return "NHWC";
    case kNCHW:
      return "NCHW";
  }
  ASSERT_CHECK(false);
  return "UNKNOWN";
}

static std::string BNOpToString(cudnnBatchNormOps_t op) {
  switch (op) {
    case CUDNN_BATCHNORM_OPS_BN:
      return "CUDNN_BATCHNORM_OPS_BN";
    case CUDNN_BATCHNORM_OPS_BN_ACTIVATION:
      return "CUDNN_BATCHNORM_OPS_BN_ACTIVATION";
    case CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:
      return "CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION";
  }
  ASSERT_CHECK(false);
  return "UNKNOWN";
}

int main() {
  int N = 256;
  int C = 4;
  int H = 224;
  int W = 224;

  auto layouts = {kNHWC, kNCHW};
  auto ops = {CUDNN_BATCHNORM_OPS_BN,
              CUDNN_BATCHNORM_OPS_BN_ACTIVATION,
              CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION};
  for (const auto layout : layouts) {
    for (const auto op : ops) {
      try {
        BatchNormTest<float>(N, C, H, W, layout, op);
      } catch (std::exception &ex) {
        std::cerr << "Float Test Failed: " << LayoutToString(layout) << " "
                  << BNOpToString(op) << " | message: " << ex.what()
                  << std::endl;
        cudaGetLastError();
      }

      try {
        BatchNormTest<half>(N, C, H, W, layout, op);
      } catch (std::exception &ex) {
        std::cerr << "Half Test Failed: " << LayoutToString(layout) << " "
                  << BNOpToString(op) << " | message: " << ex.what()
                  << std::endl;
        cudaGetLastError();
      }
    }
  }
  return 0;
}
