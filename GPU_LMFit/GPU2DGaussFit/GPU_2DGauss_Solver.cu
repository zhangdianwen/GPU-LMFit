#include "Apps.cuh"

#include "GPU_2DGauss_Solver.cuh"
#include "GPU_2DGauss_FitFunctions.cuh"

#if defined(_MATLAB_DISPLAY_CUDA_SETTING_INFO_) 
#include "mex.h"
#endif

/* Conditional Compilation Constants */
#define _RETURN_INFONUM_ // return the GPU-LMFit output code - InfoNum instead of Chi squares

/* 2*sizeof(_real_) for sp_smax and sp_smin, 1*sizeof(int) for sp_smax_idx. */
#define GPU_2DGau_Init_Extern_Shared_Mem_Size_Per_Thread (2*(2*sizeof(_real_)+1*sizeof(int)))

/* Variables in GPU constant memory */
#ifdef _SINGLE_FLOAT_ 
LM_CRITERIA GPU_2DGaussFit_LM_Criteria = {1e-4f, 1e-4f, 1e-5f, 1.0e-07f, 20.0f, 50}; // Good enough for GPU 2D Gauss fit
#else
LM_CRITERIA GPU_2DGaussFit_LM_Criteria = {1e-4, 1e-4, 1e-5, 1.0e-07, 20.0, 50}; // Good enough for GPU 2D Gauss fit
#endif

/* External shared memory pointer used by all kernels */
extern __shared__ char SVec[]; 


/***********************************************************************************************************
		                                                               Device  __global__ functions 
*************************************************************************************************************/
__global__ void GPU_2DGau_Init_Kernel(const int n, _real_ *x, _real_ *GPUImgDataBuffer, const int ImgDim, const _real_ init_s, _real_*B_Corr, 
	int NumOfImgs, int *WaitingIdx)
{
	/*
	Function: to calcuate the initial values of all fitting parameters
	Parameters description:
		n (input) - the number of fitting parameters;
		x (input and output) - the fitting parameters array in global memory of GPU;
		GPUImgDataBuffer (input) - the images data in global memory of GPU;
		ImgDim (input) - Image dimension size (assume square images);
		init_s (input) - user specified initial value of Gaussian waist width s;
		B_Corr (output) - background level correction factor. 

	Note:
		Each block completes one fit, so both TwoDGauss_GPU_LMFit_Kernel & GPU_2DGau_Init_Kernel 
		must use the same gdim. 
	*/

	__shared__ _real_ *sp_smax, *sp_smin;
	__shared__ int *sp_smax_idx;
	__shared__ _real_ *sv_ImgData;
	__shared__ int sv_m;
	__shared__ int CurrentIdx;

	int i0;

	if (tidx == 0) {
		sp_smax = (_real_ *)SVec;
		sp_smin = &sp_smax[2 * bdim];
		sp_smax_idx = (int *)&sp_smin[2 * bdim];
		sv_m = ImgDim*ImgDim;
		CurrentIdx = bidx;
	}
	__syncthreads();

	while (CurrentIdx < NumOfImgs) {
		if (tidx == 0)
			sv_ImgData = &GPUImgDataBuffer[CurrentIdx*sv_m];
		__syncthreads();

		sp_smax[tidx] = zero, sp_smin[tidx] = E10;
		sp_smax[tidx + bdim] = zero, sp_smin[tidx + bdim] = E10;
		__syncthreads();

		for (i0 = 0; i0 < sv_m; i0 += 2 * bdim) {
			if (i0 + tidx < sv_m) {
				sp_smin[tidx] = MIN(sp_smin[tidx], sv_ImgData[i0 + tidx]);
				if (sv_ImgData[i0 + tidx] > sp_smax[tidx]) {
					sp_smax[tidx] = sv_ImgData[i0 + tidx];
					sp_smax_idx[tidx] = i0 + tidx;
				}
			}
			if (i0 + tidx + bdim < sv_m) {
				sp_smin[bdim + tidx] = MIN(sp_smin[bdim + tidx], sv_ImgData[i0 + bdim + tidx]);
				if (sv_ImgData[i0 + bdim + tidx] > sp_smax[bdim + tidx]) {
					sp_smax[bdim + tidx] = sv_ImgData[i0 + bdim + tidx];
					sp_smax_idx[bdim + tidx] = i0 + bdim + tidx;
				}
			}
		}
		__syncthreads();

		for (i0 = bdim; i0 >= 1; i0 >>= 1) { 	//Warp Reduction 
			if (tidx < i0) {
				sp_smin[tidx] = MIN(sp_smin[tidx], sp_smin[i0 + tidx]);
				if (sp_smax[i0 + tidx] > sp_smax[tidx]) {
					sp_smax[tidx] = sp_smax[i0 + tidx];
					sp_smax_idx[tidx] = sp_smax_idx[i0 + tidx];
				}
			}
			//if (i0>=warpSize) // tested - not work even when using volatile for pointers
			__syncthreads();
		} // smdata[0] are the real value of s2 in CPU LM_enorm

		for (i0 = 0; i0 < sv_m; i0 += 2 * bdim) {
			if (i0 + tidx < sv_m) sv_ImgData[i0 + tidx] -= (sp_smin[0] - one);
			if (i0 + bdim + tidx < sv_m) sv_ImgData[i0 + bdim + tidx] -= (sp_smin[0] - one);
		}
		__syncthreads();

		if (tidx == 0) {
			B_Corr[CurrentIdx] = sp_smin[0] - one;
			x[CurrentIdx*n + 0] = one;
			x[CurrentIdx*n + 1] = sp_smax[0] - sp_smin[0];
			x[CurrentIdx*n + 2] = sp_smax_idx[0] % ImgDim;
			x[CurrentIdx*n + 3] = floorf((sp_smax_idx[0] + zero) / ImgDim);
			x[CurrentIdx*n + 4] = init_s;
			// Check if all fitting functions are initialized.
			CurrentIdx = atomicAdd(WaitingIdx, 1); // This function returns old value in WaitingIdx.
		}
		__syncthreads();
	} // while-loop
}


__global__ void TwoDGauss_GPU_LMFit_Kernel(TWODGAUSS_GPU_LMFIT_INS GPU_LMFit_Ins, _real_ *x, 
	_real_ *GPUFitDataBuffer, int *InfoNum, _real_ *Chisq, int SVecSize, int NumOfImgs, int *WaitingIdx)
{
	/*
	Function: perform two-dimensional Gaussian fit using GPU_LMFit
	Parameters description:
		GPU_LMFit_Ins (input) - some basic parameters, see the type definition in GPU_2DGauss_Solver.cuh;
		x (input and output) - the fitting parameters array in global memory of GPU;
		GPUImgDataBuffer (input) - the images data in global memory of GPU;
		InfoNum (output) - if it is not NULL, it returns the exist code from GPU_LMFit ;
		Chisq (output) - if it is not NULL, it returns Chi square of the fit;
		SVecSize (input) - the size of externally allocated shared memory for each CUDA block;

	Note:
		Each block completes one fit, so both TwoDGauss_GPU_LMFit_Kernel & GPU_2DGau_Init_Kernel 
		must use the same gdim. 
	*/

	/*Fit Function variabls */
	__shared__ GPU_FUNC_CONSTS GPU_f_Cs;
	__shared__ int CurrentIdx;

	if(tidx==0){ 
		GPU_f_Cs.sv_n = GPU_LMFit_Ins.n;
		GPU_f_Cs.sv_m = GPU_LMFit_Ins.m;
		GPU_f_Cs.sv_ImgDim = GPU_LMFit_Ins.ImgDim;
		GPU_f_Cs.sv_JacMethod = GPU_LMFit_Ins.JacMethod;
		GPU_f_Cs.sv_FitMethod = GPU_LMFit_Ins.FitMethod;
		GPU_f_Cs.sp_buffer = (_real_ *)SVec; 
		CurrentIdx = bidx;
	}
	__syncthreads(); 

	while (CurrentIdx < NumOfImgs) {
		if (tidx == 0)  GPU_f_Cs.sp_CurrData = &GPUFitDataBuffer[CurrentIdx*GPU_LMFit_Ins.m];
		__syncthreads();

		if (!GPU_f_Cs.sv_JacMethod)
			GPU_LMFit(GPU_FitFunction, &GPU_f_Cs, NULL, NULL, &x[CurrentIdx*GPU_LMFit_Ins.n],
				NULL, NULL, NULL, NULL,
				InfoNum == NULL ? NULL : &InfoNum[CurrentIdx], Chisq == NULL ? NULL : &Chisq[CurrentIdx],
				GPU_f_Cs.sp_buffer, // The shared buffer in GPU_LMFit starts at sp_smem_left.
				SVecSize, // The total size of shared buffer for GPU_LMFit. 
				bdim * sizeof(_real_)); // The maximum size of shared memory used in user-defined fit function or Jacobian function. 
		else {
			GPU_LMFit(GPU_FitFunction, &GPU_f_Cs, GPU_AnalyticalJacobian, &GPU_f_Cs, &x[CurrentIdx*GPU_LMFit_Ins.n],
				NULL, NULL, NULL, NULL,
				InfoNum == NULL ? NULL : &InfoNum[CurrentIdx], Chisq == NULL ? NULL : &Chisq[CurrentIdx],
				GPU_f_Cs.sp_buffer, // The shared buffer in GPU_LMFit starts at sp_smem_left.
				SVecSize, // The total size of shared buffer for GPU_LMFit. 
				bdim * sizeof(_real_)); // The maximum size of shared memory used in user-defined fit function or Jacobian function. 
		}
		__syncthreads();

		// Check if all fitting functions are done.
		if (tidx == 0) CurrentIdx = atomicAdd(WaitingIdx, 1); // This function returns old value in WaitingIdx.
		__syncthreads();
	}
}


__global__ void GPU_B_Corr_Kernel(const int n, const int NumOfImgs, _real_ *x, _real_*B_Corr)
{
	/*
	Function: correct the fitting parameter B - background level.
	Parameters description:
		n (input) - the number of fitting parameters;
		NumOfImgs (input) - the number of images to run;
		x (input and output) - the fitting parameters array in global memory of GPU;
		B_Corr (input) - it is background level correction factors returned from the function GPU_2DGau_Init_Kernel.
	*/
	const int idx = bidx*bdim+tidx;
	if(idx < NumOfImgs)
		x[idx*n] += B_Corr[idx];
}


/* **********************************************************************************************************
		                                                                    CPU host functions
*************************************************************************************************************/
bool Get_GPU_LMFit_Kernel_Config(int n, int m, int userBlkSize, struct cudaDeviceProp deviceProp, 
							struct cudaFuncAttributes KernelAttrib, int *NumOfThreadsPer1DBLK, int *SVecSize, 
							int *NumOfGPULMFitSharednVec, int *NumOfGPULMFitSharedmVec, 
							char *ErrMsg)
{
	/*
	Function: set up the configuration of GPU for the kernel function - TwoDGauss_GPU_LMFit_Kernel.
	Parameters description:
		n (input) - the number of fitting parameters;
		m (input) - the number of data points;
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		deviceProp (input) - a struct for CUDA device properties;
		KernelAttrib (input) - a struct for the properties of the kernel function - TwoDGauss_GPU_LMFit_Kernel;
		NumOfThreadsPer1DBLK (output) - the optimized number of threads per CUDA block; 
		SVecSize (output) - the size of the shared memory required to externally allocate for each CUDA block 
			with NumOfThreadsPer1DBLK threads;
		NumOfGPULMFitSharednVec (output) - the number of n-length vectors which can be in shared memory
			in GPU_LMFit;
		NumOfGPULMFitSharedmVec (output) - the number of m-length vectors which can be in shared memory 
			in GPU_LMFit;
		ErrMsg (output) - a string pointer for the text of the error description if an error is found.
		return value - true if the user-specified GPU device is initialized successfully, or
						false otherwise. 

	Note:
		All the above three parameters - NumOfGPULMFitSharednVec, NumOfGPULMFitSharedmVec and 
		GPU_LMFit_Num_Of_SMEM_Vecs (see also GPU_LMFit.cuh) as shown below. 
	*/

	size_t MaxExtSharedMemPerBlk;
	int MaxThreadsPer1DBlk;

	if(!Get_Kernel_Basic_Config(userBlkSize, deviceProp, KernelAttrib, &MaxThreadsPer1DBlk, 
		&MaxExtSharedMemPerBlk, ErrMsg)) return(false);

	int SMEMAllowedBlkDim; // Max blockDim.x limited the available external shared memory per CUDA block
	SMEMAllowedBlkDim = (int)floor((MaxExtSharedMemPerBlk+zero)/
		(GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread+zero)); 
	*NumOfThreadsPer1DBLK = MIN((int)nearest_pow2(SMEMAllowedBlkDim), (int)nearest_pow2(m));  
	*NumOfThreadsPer1DBLK = MIN(*NumOfThreadsPer1DBLK, (int)nearest_pow2(MaxThreadsPer1DBlk)); 
	
	/* Minimal SVec size required GPU_LMFit for internal warp reduction. */
	size_t BasicSVecSize = GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread*(*NumOfThreadsPer1DBLK); 
	if(BasicSVecSize>MaxExtSharedMemPerBlk) {
		sprintf(ErrMsg,   
			"%s : not enough shared memory for basic shared SVec!", __FUNCTION__);
		return false;
	}
	
	/* 
	Use the function GPU_LMFit_Num_Of_SMEM_Vecs declared in GPU_LMFit.cuh to determine 
	how many n- and m-length vectors can be in shared memory for GPU_LMFit.
	*/
	GPU_LMFit_Num_Of_SMEM_Vecs(m, n, (int)(MaxExtSharedMemPerBlk-BasicSVecSize), 
		NumOfGPULMFitSharedmVec, NumOfGPULMFitSharednVec);
	
	// The size of external SVec
	*SVecSize = (int)(BasicSVecSize+(*NumOfGPULMFitSharednVec)*n*sizeof(_real_)+
		(*NumOfGPULMFitSharedmVec)*m*sizeof(_real_));

	return true;
}


bool Get_2DGau_Init_Kernel_Config(int m, int userBlkSize, struct cudaDeviceProp deviceProp, 
							struct cudaFuncAttributes KernelAttrib, int *NumOfThreadsPer1DBLK, 
							int *SVecSize, char *ErrMsg)
{
	/*
	Function: set up the configuration of GPU for the kernel function - GPU_2DGau_Init_Kernel.
	Parameters description:
		m (input) - the number of data points;
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		deviceProp (input) - a struct for CUDA device properties;
		KernelAttrib (input) - a struct for the properties of the kernel function - GPU_2DGau_Init_Kernel;
		NumOfThreadsPer1DBLK (output) - the optimized number of threads per CUDA block; 
		SVecSize (output) - the size of the shared memory required to externally allocate for each CUDA block 
			with NumOfThreadsPer1DBLK threads;
		ErrMsg - a string pointer for the text of the error description if an error is found.
		return value - true if the user-specified GPU device is initialized successfully, or
						false otherwise. 
	*/

	size_t MaxExtSharedMemPerBlk;
	int MaxThreadsPer1DBlk;

	if(!Get_Kernel_Basic_Config(userBlkSize, deviceProp, KernelAttrib, &MaxThreadsPer1DBlk, 
		&MaxExtSharedMemPerBlk, ErrMsg)) return(false);

	int SMEMAllowedBlkDim; // Max blockDim.x limited the available external shared memory per CUDA block
	SMEMAllowedBlkDim = (int)floor((MaxExtSharedMemPerBlk+zero)/
		(GPU_2DGau_Init_Extern_Shared_Mem_Size_Per_Thread+zero)); 
	*NumOfThreadsPer1DBLK = MIN((int)nearest_pow2(SMEMAllowedBlkDim), (int)nearest_pow2(m));  
	*NumOfThreadsPer1DBLK = MIN(*NumOfThreadsPer1DBLK, (int)nearest_pow2(MaxThreadsPer1DBlk)); 

	/* Minimal SVec size required GPU_2DGau_Init_Kernel for internal warp reduction. */
	*SVecSize = GPU_2DGau_Init_Extern_Shared_Mem_Size_Per_Thread*(*NumOfThreadsPer1DBLK); 
	if((size_t)(*SVecSize)>MaxExtSharedMemPerBlk) {
		sprintf(ErrMsg, 
			"%s : not enough shared memory for basic shared SVec!", __FUNCTION__);
		return false;
	}

	return true;
}


int GPU_LMFIT_Solver(int n, int m, int NumOfImgs, int ImgDim, int JacMethod, int FitMethod, _real_ init_s, 
	int DeviceID, _real_ *ImgsBufferPtr,  int *userBlkSize, int *userGridSize, _real_ *outx, _real_ *Info, char *ErrMsg)
{
	/*
	Function: set up the configuration of GPU and implement two-dimensional Gaussian image fittings.
	Parameters description:
		n (input) - the number of fitting parameters;
		m (input) - the number of data points;
		NumOfImgs (input) - the number of images to be fit;
		ImgDim (input) - image dimension size (assume square images);
		JacMethod (input) - if it is 0, a user-defined analytical Jacobian function will be used if it is available,  
			or GPU_LMFit uses its integrated numerical Jacobian function;
		FitMethod (input) - if it is 0, use maximum likelihood estimator in the fit function, or use unweighted 
			least squares;
		init_s (input) - user specified initial value of Gaussian waist width s;
		DeviceID (input) - user-specified GPU device ID number;
		ImgsBufferPtr (input) - the images data in host memory;
		userBlkSize (input) - user specified maximum number of threads per CUDA block;
		userGridSize (input) - user specified maximum number of blocks per CUDA grid;
		outx (output) - fitting parameters in host memory;
		Info (output) - its value depends on the definition of _RETURN_INFONUM_ can be either Chi squares 
			of all fittings or the exit codes of the GPU_LMFit;
		ErrMsg (output) - a string pointer for the text of the error description if an error is found.
		return value - non-zeros if an error is found, or otherwise a zero. 
	*/

	/* CPU Variables */
	memsize_t GPU_Global_Mem_Size;			// The size of total global memory on GPU.
	int GPU_Img_Data_Size, GPU_x_Size, GPU_B_Corr_Size, GPU_Info_Size, GPU_LMFit_WA_Size;
	int ii;
	int BLKsPer1DGrid;
	int GPU_LMFit_BLK_Size, GPU_LMFit_SVecSize, GPU_2DGau_Init_BLK_Size, GPU_2DGau_Init_SVecSize;
	int NumOfGPULMFitSharednVec, NumOfGPULMFitSharedmVec;
	TWODGAUSS_GPU_LMFIT_INS GPU_LMFit_Paras;
	long long TotalGlobalBufferSize;				// The size of the global memory used on GPU

	// for device
	dim3 dimGPULMFitBlk1D, dimGPULMFitGrid1D;
	dim3 dim2DGaussianInitBlk1D, dim2DGaussianInitGrid1D;
	dim3 dimBCorrBlk1D, dimBCorrGrid1D;
	struct cudaDeviceProp deviceProp;
	struct cudaFuncAttributes GPULMFitKernel_funcAttrib;
	struct cudaFuncAttributes GPU2DGauInitKernel_funcAttrib;

	/* GPU Variables */
	int *GPU_WaitingIdx;
	_real_ *GPU_ImgsBuffer;			// Images data buffer in the global memory on GPU.
	_real_ *GPU_x, *GPU_B_Corr;		// Fitting parameters and background level correction factors in the global memory on GPU.
#ifdef _RETURN_INFONUM_
	int *GPU_info;						// GPU_LMFit exit codes in the global memory on GPU.
#else
	_real_ *GPU_info;					// Chi squares in the global memory on GPU.
#endif

	/* Initialize cuda device */
	 if(!CuDeviceInit(&deviceProp, DeviceID, ErrMsg)) return(-2);
		
     /* Get kernels' properties */
	if(CheckCudaError(cudaFuncGetAttributes(&GPULMFitKernel_funcAttrib, 
		TwoDGauss_GPU_LMFit_Kernel), ErrMsg, "cudaFuncGetAttributes")) return(-3);
	if(CheckCudaError(cudaFuncGetAttributes(&GPU2DGauInitKernel_funcAttrib, 
		GPU_2DGau_Init_Kernel), ErrMsg, "GPU_2DGau_Init_Kernel : cudaFuncGetAttributes")) 
		return(__LINE__);

     /* 
     How to determine gridDim.x:
		Although deviceProp.maxGridSize[0] allows 65535 blocks for CUDA Capability 2.0,
		it has been tested and found too many blocks on GPU will cause longer GPU 
		computation time, and windows can automatically restart GPU device after the time 
		defined in Windows (tested in Win7) Regitory Editor -> HKEY_Local_Machine -> 
		System -> CurrentControlSet -> Control -> GraphicsDrivers -> TdrDelay. 
		If GPU is occupied too int by the program and Windows restarts it, 
		then the computation results are wrong, so User parameter userGridSize is important 
		for user to determine a proper gridDim.x for FLIM_GPU_LMFit_Kernel.
	*/
	BLKsPer1DGrid = MIN(NumOfImgs, deviceProp.maxGridSize[0]);
	BLKsPer1DGrid = MIN(BLKsPer1DGrid, *userGridSize); 
	
	/* Buffers' sizes */
	GPU_Global_Mem_Size = deviceProp.totalGlobalMem;
	// Parameters' sizes
	GPU_x_Size = NumOfImgs*n*sizeof(_real_);
	GPU_B_Corr_Size = MIN(BLKsPer1DGrid, NumOfImgs)*sizeof(_real_);
	GPU_Img_Data_Size = NumOfImgs*m*sizeof(_real_);
#ifdef _RETURN_INFONUM_
	if(Info) GPU_Info_Size = NumOfImgs*sizeof(int);
	else GPU_Info_Size = 0;
#else
	if(Info) GPU_Info_Size = NumOfImgs*sizeof(_real_);
	else GPU_Info_Size = 0;
#endif

	/* Set up kernel's configuration */
	//TwoDGauss_GPU_LMFit_Kernel
	if(!Get_GPU_LMFit_Kernel_Config(n, m, 
		*userBlkSize, deviceProp, GPULMFitKernel_funcAttrib, &GPU_LMFit_BLK_Size, 
		&GPU_LMFit_SVecSize, &NumOfGPULMFitSharednVec, &NumOfGPULMFitSharedmVec, 
		ErrMsg)) 
		return(__LINE__);
	dimGPULMFitBlk1D.x = GPU_LMFit_BLK_Size, dimGPULMFitBlk1D.y = 1, dimGPULMFitBlk1D.z = 1;
	dimGPULMFitGrid1D.x = BLKsPer1DGrid, dimGPULMFitGrid1D.y = 1, dimGPULMFitGrid1D.z = 1;
	// GPU_2DGau_Init_Kernel
	if(!Get_2DGau_Init_Kernel_Config(m, *userBlkSize, deviceProp, GPU2DGauInitKernel_funcAttrib, 
		&GPU_2DGau_Init_BLK_Size, &GPU_2DGau_Init_SVecSize, ErrMsg)) 
		return(__LINE__);
	dim2DGaussianInitBlk1D.x = GPU_2DGau_Init_BLK_Size;
	dim2DGaussianInitGrid1D.x = BLKsPer1DGrid;
	//GPU_B_Corr_Kernel
	dimBCorrBlk1D.x = MIN(BLKsPer1DGrid, deviceProp.maxThreadsPerBlock);
	dimBCorrGrid1D.x = (unsigned int)ceil((BLKsPer1DGrid+zero)/dimBCorrBlk1D.x);

#ifdef _MATLAB_DISPLAY_CUDA_SETTING_INFO_
	mexPrintf("\n\tThe dimensions of CUDA block and grid are, respectively, \n\t\t(%d, %d, %d) and (%d, %d, %d) for GPU_2DGau_Init_Kernel.", 
		dim2DGaussianInitBlk1D.x, dim2DGaussianInitBlk1D.y, dim2DGaussianInitBlk1D.z, 
		dim2DGaussianInitGrid1D.x, dim2DGaussianInitGrid1D.y, dim2DGaussianInitGrid1D.z);
	mexPrintf("\n\tThe dimensions of CUDA block and grid are, respectively, \n\t\t(%d, %d, %d) and (%d, %d, %d) for GPU_LMFit_Kernel.", 
		dimGPULMFitBlk1D.x, dimGPULMFitBlk1D.y, dimGPULMFitBlk1D.z, 
		dimGPULMFitGrid1D.x, dimGPULMFitGrid1D.y, dimGPULMFitGrid1D.z);
	mexPrintf("\n\tThe dimensions of CUDA block and grid are, respectively, \n\t\t(%d, %d, %d) and (%d, %d, %d) for GPU_B_Corr_Kernel.\n", 
		dimBCorrBlk1D.x, dimBCorrBlk1D.y, dimBCorrBlk1D.z, 
		dimBCorrGrid1D.x, dimBCorrGrid1D.y, dimBCorrGrid1D.z);
#endif

	/* Determine GPU memory buffer size */
	GPU_LMFit_WA_Size = GPU_LMFit_Config(m, n, GPU_2DGaussFit_LM_Criteria, BLKsPer1DGrid,
		NumOfGPULMFitSharedmVec, NumOfGPULMFitSharednVec, ErrMsg);
	if(GPU_LMFit_WA_Size<0) {
		mexPrintf("\n\nFailed in GPU_LMFit_Config() with a code of %d.\n\n", GPU_LMFit_WA_Size);
		return(GPU_LMFit_WA_Size);
	}
	TotalGlobalBufferSize = GPU_Img_Data_Size+GPU_x_Size+GPU_B_Corr_Size+GPU_Info_Size+ 
							GPU_LMFit_WA_Size; 
#ifdef _MATLAB_DISPLAY_CUDA_SETTING_INFO_
	mexPrintf("\n\tMax available global memory on GPU is %g MByte;", 
		(GPU_Global_Mem_Size+zero)/1024/1024);
	mexPrintf("\n\tTotal used global memory on GPU is %g MByte (%g%% of Max).\n", 
		(TotalGlobalBufferSize+zero)/1024/1024, 100.0*(TotalGlobalBufferSize+zero)/GPU_Global_Mem_Size);
#endif
	if((2.0*GPU_Global_Mem_Size/3.0) < TotalGlobalBufferSize){
		// Not support for requiring global mem size > 2/3 of total global mem.
		sprintf(ErrMsg, 
			"%s : Not enough global memory on GPU for required %g MBytes!!!", 
			__FILE__, (TotalGlobalBufferSize+zero)/1024.0/1024.0);
#ifdef _MY_DEBUG_DISPLAY_
		mexErrMsgTxt(ErrMsg);
#endif
		return(-4);
	} 

	/* Allocate GPU memory buffers */
	cudaMalloc((void **)&GPU_WaitingIdx, sizeof(int));
	if(CheckCudaError(cudaMalloc((void **)&GPU_x, GPU_x_Size), 
		ErrMsg, "GPU_x : cudaMalloc")) return(__LINE__);	
	if(CheckCudaError(cudaMalloc((void **)&GPU_B_Corr, GPU_B_Corr_Size), 
		ErrMsg, "GPU_B_Corr : cudaMalloc")) return(__LINE__);
	if(GPU_Info_Size){
		if(CheckCudaError(cudaMalloc((void **)&GPU_info, GPU_Info_Size), 
			ErrMsg, "GPU_info : cudaMalloc")) return(__LINE__);
	}
	else GPU_info = NULL;
	if(CheckCudaError(cudaMalloc((void **)&GPU_ImgsBuffer, GPU_Img_Data_Size), 
		ErrMsg, "GPU_ImgsBuffer : cudaMalloc")) return(__LINE__);

	/* Copy the input CPU data to the device global buffer */
	if(CheckCudaError(cudaMemcpy(GPU_ImgsBuffer, ImgsBufferPtr, 
		GPU_Img_Data_Size, cudaMemcpyHostToDevice), 
		ErrMsg, "ImgsBuffer to GPU_ImgsBuffer : cudaMemcpy")) return(__LINE__);
	
	/* Initialize the variables in GPU */
	cudaMemset(GPU_info, 0, GPU_Info_Size);
			
	/* Initialize GPU_LMFit_Paras */
	GPU_LMFit_Paras.n = n;
	GPU_LMFit_Paras.m = m;
	GPU_LMFit_Paras.ImgDim = ImgDim;
	GPU_LMFit_Paras.JacMethod = JacMethod;
	GPU_LMFit_Paras.FitMethod = FitMethod;

	/* Start GPU Kernels */
	cudaMemcpy(GPU_WaitingIdx, &BLKsPer1DGrid, sizeof(int), cudaMemcpyHostToDevice);
	dim2DGaussianInitGrid1D.x = BLKsPer1DGrid;
	GPU_2DGau_Init_Kernel <<<dim2DGaussianInitGrid1D, dim2DGaussianInitBlk1D, GPU_2DGau_Init_SVecSize >>>
		(n, GPU_x, GPU_ImgsBuffer, ImgDim, init_s, GPU_B_Corr, NumOfImgs, GPU_WaitingIdx);
	if (CheckCudaError(cudaDeviceSynchronize(), ErrMsg, "cudaDeviceSynchronize after GPU_2DGau_Init_Kernel"))
		return(__LINE__);

	cudaMemcpy(GPU_WaitingIdx, &BLKsPer1DGrid, sizeof(int), cudaMemcpyHostToDevice);
	dimGPULMFitGrid1D.x = BLKsPer1DGrid;
#ifdef _RETURN_INFONUM_
	TwoDGauss_GPU_LMFit_Kernel <<<dimGPULMFitGrid1D, dimGPULMFitBlk1D, GPU_LMFit_SVecSize >>>(
		GPU_LMFit_Paras, GPU_x, GPU_ImgsBuffer, GPU_info, NULL, GPU_LMFit_SVecSize, NumOfImgs, GPU_WaitingIdx);
#else
	TwoDGauss_GPU_LMFit_Kernel << <dimGPULMFitGrid1D, dimGPULMFitBlk1D, GPU_LMFit_SVecSize >> >(
		GPU_LMFit_Paras, GPU_x, GPU_ImgsBuffer, NULL, GPU_info, GPU_LMFit_SVecSize, NumOfImgs, GPU_WaitingIdx);
#endif
	if (CheckCudaError(cudaDeviceSynchronize(), ErrMsg,
		"cudaDeviceSynchronize after TwoDGauss_GPU_LMFit_Kernel"))
		return(__LINE__);

	dimBCorrGrid1D.x = (unsigned int)ceil((NumOfImgs + zero) / dimBCorrBlk1D.x);
	GPU_B_Corr_Kernel <<<dimBCorrGrid1D, dimBCorrBlk1D >>>(n, NumOfImgs, GPU_x, GPU_B_Corr);
	if (CheckCudaError(cudaDeviceSynchronize(), ErrMsg, "cudaDeviceSynchronize after GPU_B_Corr_Kernel"))
		return(__LINE__);
	/* End GPU Kernels */

	// Clean up GPU_LMFit working memory. 
	void GPU_LMFit_Clear();

	/* CUDA results return */
	// GPU_x -> outx
	if(CheckCudaError(cudaMemcpy(outx, GPU_x, GPU_x_Size, cudaMemcpyDeviceToHost), 
		ErrMsg, "GPU_x to outx : cudaMemcpy")) return(__LINE__);
	// GPU_info -> CPU_x -> Info
	if(Info){
		if(CheckCudaError(cudaMemcpy(Info, GPU_info, GPU_Info_Size, cudaMemcpyDeviceToHost), 
			ErrMsg, "GPU_info to Info : cudaMemcpy")) return(__LINE__);
#ifdef _RETURN_INFONUM_
		for(ii=NumOfImgs-1; ii>=0; ii--) Info[ii] = (_real_)(((int *)Info)[ii]);
#endif
	}
	
	/* Free device memory */
	if(GPU_x)
		cudaFree(GPU_x); 
	if(GPU_B_Corr)
		cudaFree(GPU_B_Corr); 
	if(GPU_info)
		cudaFree(GPU_info); 
	if(GPU_ImgsBuffer)
		cudaFree(GPU_ImgsBuffer); 

	/* Normal return */
	return(0); 
}
