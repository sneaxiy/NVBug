# How to Run

```shell
bash compile_and_run.sh
```

You will get the following error.

```
Float Test Failed: NCHW CUDNN_BATCHNORM_OPS_BN_ACTIVATION | message: `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize( handle, mode, op, x_desc, z_desc, y_desc, param_desc, act_desc, &workspace_size)` check failed: test_cudnn_bn.cu:157 error code: CUDNN_STATUS_NOT_SUPPORTED
Half Test Failed: NCHW CUDNN_BATCHNORM_OPS_BN_ACTIVATION | message: `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize( handle, mode, op, x_desc, z_desc, y_desc, param_desc, act_desc, &workspace_size)` check failed: test_cudnn_bn.cu:157 error code: CUDNN_STATUS_NOT_SUPPORTED
Float Test Failed: NCHW CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION | message: `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize( handle, mode, op, x_desc, z_desc, y_desc, param_desc, act_desc, &workspace_size)` check failed: test_cudnn_bn.cu:157 error code: CUDNN_STATUS_NOT_SUPPORTED
Half Test Failed: NCHW CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION | message: `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize( handle, mode, op, x_desc, z_desc, y_desc, param_desc, act_desc, &workspace_size)` check failed: test_cudnn_bn.cu:157 error code: CUDNN_STATUS_NOT_SUPPORTED
```
