#ifndef _GPU_COMMONS_CUH_
#define _GPU_COMMONS_CUH_

#include "cuda_runtime_api.h" // For CUDA
#include "math.h"

#include "Floating_Data_Type.h"

/* Conditional Compilation Constants */
#define _MATLAB_DISPLAY_CUDA_SETTING_INFO_ // Display the CUDA configuration

/* Macros for CUDA threads and blocks indices */
#define tidx threadIdx.x
#define bidx blockIdx.x
#define bdim blockDim.x
#define gdim gridDim.x

/* **********************************************************************************************************
		                                                              Commonly used functions 
*************************************************************************************************************/
extern bool CheckCudaError(cudaError, char *, const char *);
extern bool CuDeviceInit(struct cudaDeviceProp *, int, char *);
extern bool Get_Kernel_Basic_Config(int, struct cudaDeviceProp, struct cudaFuncAttributes, int *, size_t *, char *);

#endif //_GPU_COMMONS_H_