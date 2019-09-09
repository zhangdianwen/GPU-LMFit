#ifndef _GPU_COMMONS_CU_
#define _GPU_COMMONS_CU_

#include "Apps.cuh"
#include "GPU_Commons.cuh"

#if defined(_MATLAB_DISPLAY_CUDA_SETTING_INFO_)
#include "mex.h"
#endif

/* **********************************************************************************************************
		                                                              Commonly used functions 
*************************************************************************************************************/
/* Check Cuda errors */
bool CheckCudaError(cudaError errnum, char *ErrMsg, const char *InfoString)
{
	/*
	Parameters description:
		errnum (input) - the error code of a CUDA function;
		ErrMsg (output) - a string pointer for the text of the error description;
		InfoString (input) - a string pointer for the text of some additional information (e.g., the location of the error). 
		return value - true if an error is identified, or false otherwise . 
	*/

	if (errnum != cudaSuccess){
		sprintf(ErrMsg, "%s : %s - '%s'.", __FUNCTION__, InfoString, cudaGetErrorString(errnum));
		return true;
	}
	else
		return false;
}


/* Initialize cuda device */
bool CuDeviceInit(struct cudaDeviceProp *deviceProp, int DeviceID, char * ErrMsg)
{    
	/*
	Parameters description:
		deviceProp (input) - a struct for CUDA device properties;
		DeviceID (input) - user-specified GPU device ID number;
		ErrMsg (output) - a string pointer for the text of the error description;
		return value - true if the user-specified GPU device is initialized successfully, or
						false otherwise. 
	*/

	int deviceCount, targetDevice;

	/* Get the number of CUDA enabled GPU devices */
	if(CheckCudaError(cudaGetDeviceCount(&deviceCount), ErrMsg, "cudaGetDeviceCount")) 
		return(false);
	/* Select and reset target device */
	targetDevice = DeviceID;
	if(CheckCudaError(cudaSetDevice(targetDevice), ErrMsg, "cudaSetDevice")) return(false);
	if(CheckCudaError(cudaDeviceReset(), ErrMsg, "cudaDeviceReset")) return(false);
	/* Query target device */
	if(CheckCudaError(cudaGetDeviceProperties(deviceProp, targetDevice), 
		ErrMsg, "cudaGetDeviceProperties")) 
		return(false);

	return(true);
}


/* 
	Following the instruction of CUDA_Occupancy_Calculator.xls which is provided with CUDA,
	the below function Get_Kernel_Basic_Config is to determine the limit of the CUDA block size 
	and the limit of the shared memory size per block for a kernel function. 
*/
bool Get_Kernel_Basic_Config(int userBlkSize, struct cudaDeviceProp deviceProp, 
								struct cudaFuncAttributes KernelAttrib, int *MaxThreadsPer1DBlk, 
								size_t *MaxExtSharedMemPerBlk, char *ErrMsg)
{
	/*
	Parameters description:
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		deviceProp (input) - a struct for CUDA device properties;
		KernelAttrib (input) - a struct for the properties of the kernel function;
		MaxThreadsPer1DBlk (output) - the maximum number of threads per CUDA block 
			after considering many factors as in CUDA_Occupancy_Calculator.xls; 
		MaxExtSharedMemPerBlk (output) - the maximum size of the externally allocable 
			shared memory per CUDA block for MaxThreadsPer1DBlk threads;
		ErrMsg - a string pointer for the text of the error description if an error is found.
		return value - true if the user-specified GPU device is initialized successfully, or
						false otherwise. 

	Note:
		To understand this function, please refer to CUDA_Occupancy_Calculator.xls, which is usually 
		provided with the CUDA software. 
	*/
	int RegisterLimitedBlks;			// Registers limited Blocks per Multiprocessor
	int MaxWarpsLimitedBlks;		// Maximum warps limited Blocks per Multiprocessor
	int KernelWarpsPerBlock;		// Number of Warps per Block
	int MaxWarpsPerMP;			// Maximum allowed Warps per Multiprocessor
	int RegsWarpsLimitedBlks;		// Registers and Maximum allowed Warps limited Blocks per Multiprocessor
	
	size_t MaxSharedMemPerBlk;	/* Maximum available shared memory per block 
						(including the static shared variables in the kernel function). */
	
	*MaxThreadsPer1DBlk = deviceProp.maxThreadsPerBlock;
	if (userBlkSize>0) *MaxThreadsPer1DBlk = MIN(userBlkSize, *MaxThreadsPer1DBlk);
	*MaxThreadsPer1DBlk = MIN(*MaxThreadsPer1DBlk, KernelAttrib.maxThreadsPerBlock); 
	RegisterLimitedBlks = (int)floor((deviceProp.regsPerBlock+zero)/(*MaxThreadsPer1DBlk)/KernelAttrib.numRegs);
	KernelWarpsPerBlock = (int)ceil((*MaxThreadsPer1DBlk+zero)/deviceProp.warpSize);
	MaxWarpsPerMP = (int)floor((deviceProp.maxThreadsPerBlock+zero)/deviceProp.warpSize);
	MaxWarpsLimitedBlks = (int)floor((MaxWarpsPerMP+zero)/KernelWarpsPerBlock);
	RegsWarpsLimitedBlks = MIN(RegisterLimitedBlks, MaxWarpsLimitedBlks);
	MaxSharedMemPerBlk = (int)floor((deviceProp.sharedMemPerBlock+zero)/RegsWarpsLimitedBlks);
	if(KernelAttrib.sharedSizeBytes>MaxSharedMemPerBlk){
		sprintf(ErrMsg,  
			"%s : not enough shared memory for kernel static shared memory!", __FUNCTION__);
		return false;
	}
	// MaxExtSharedMemPerBlk is maximum externally available shared memory per block
	*MaxExtSharedMemPerBlk = MaxSharedMemPerBlk-KernelAttrib.sharedSizeBytes;

	return(true);
}


#endif // _GPU_COMMONS_CU_
