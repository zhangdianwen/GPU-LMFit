/*
	GPU_LMFit:
		A GPU-based parallel Levenberg¨CMarquardt least-squares minimization fitting library 
		in CUDA C/C++ 

		This software was developed based on the C version MPFit (see References).
		Parallel computation algorithms and translation to CUDA C/C++ were applied by Xiang 
		Zhu and Dianwen Zhang, at Image Technology Group, Beckman Institute for Advanced 
		Science & Technology, University of Illinois at Urbana-Champaign. Our contact 
		information can be found at http://itg.beckman.illinois.edu/.
 
		Warranty disclaimer and copyright notice can be found at the end of this document. 

	References:
		MPFit:
			Originally developed based on Minpack by Craig Markwardt at 1) Department of 
			Astronomy, University of Maryland and 2) CRESST and Astroparticle Physics 
			Laboratory, NASA/GSFC. For more details, please refer to 
			http://cow.physics.wisc.edu/~craigm/.
		Minpack:
			Originally developed in Fortran by Jorge More', Burt Garbow, and Ken Hillstrom 
			at Argonne National Laboratory. For more details, please refer to 
			http://www.netlib.org/minpack/.
*/

#ifndef _GPU_LMFIT_H_
#define _GPU_LMFIT_H_

#include "Floating_Data_Type.h"


/*
	The below definition of GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread is the minimal 
	size of the dynamic shared memory required by each block of threads of GPU_LMFit.
	
	2*blockDim.x*sizeof(_real_) bytes extern shared memory per block is for internal warp 
	reduction operations using two double-blockDim.x-size shared memory. 

	Note: GPU_LMFit always initializes the extern shared memory to zeros before use, so this 
	part of shared memory can still be used by other device programs (e.g., the fit function or
	the user defined Jacobian function) as a temporary buffer. 
*/
#define GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread 2*sizeof(_real_)


/* 
	Definition of GPU_LMFit termination and configuration criteria structure 

	NOTE: 1) This structure is the same as originally defined in MPFit. 
			2) Some of these criteria are originally defined in Minpack. 
			3) The user may set the value explicitly for each criterion, or if any criterion value 
			    passed to GPU_LMFit is zero or less than zero, then the its default value will be 
			    substituted in GPU_LMFit. 
*/
struct LM_CRITERIA_STRUCT {
	_real_ ftol;			// Termination tolerance on the Chi-square value (default: 1e-4f)
	_real_ xtol;			// Termination tolerance on the fitting parameters (default: 1e-4f) 
	_real_ gtol;			// Termination tolerance for orthogonality convergence (default: 1e-4f)  
	_real_ epsfcn;		// The finite derivative step size for finite-difference 
	                              // approximation to Jacobian (default: 1.19209e-07f) 
	_real_ stepfactor;	// Initial step bound (default: 20.0f) 
	int maxiter;		// Maximum number of iterations (default: 25)  
};  
typedef struct LM_CRITERIA_STRUCT LM_CRITERIA;


/* 
	Prototypes of external fit function and analytical Jacobian function:

	__device__ int GPU_FitFunc(const _real_ *x, _real_ *fvec, void *FitFuncPrivDataPtr);
	__device__ int GPU_JacFunc(const _real_ *x, _real_ *fjac, void *JacFuncPrivDataPtr);

	where x is the fitting parameters pointer, fvec and fjac are the pointers to the values of 
	the fit function and the user-defined Jacobian function, respectively. FitFuncPrivDataPtr 
	and JacFuncPrivDataPtr are two pointers to the user's data to be used in both functions 
	and passed through GPU_LMFit.

	Note: 
		1) Both functions must be device functions, and all parameters must also be device 
		    parameters.
		2) If the fit function returns -1, GPU_LMFit will stop immediately after a calling of 
		     this function.
		3) If the use-defined Jacobian function returns -1, GPU_LMFit will compute the finite-
		    difference approximation to Jacobian use the GPU-based parallel function fdjac2 
		    (see also Minpack or MPFit).
*/
typedef int(*Device_FitFunc)(const _real_ *, _real_ *, void *); 
typedef int(*Device_JacFunc)(const _real_ *, _real_ *, void *); 


/* 
	Main GPU_LMFit library routine  

	__device__ void GPU_LMFit(Device_FitFunc FitFuncPtr, void *FitFuncPrivDataPtr, 
								    Device_JacFunc JacFuncPtr, void *JacFuncPrivDataPtr, 
								    _real_ *x, int *UseLowerBounds, _real_ *LowerBounds, 
								    int *UseUpperBounds, _real_ *UpperBounds, 
								    int *InfoNum, _real_ *Chisq, 
								    void *SharedMem,  int SharedMemSize, 
								    int ExtTempSharedBufferSize);
	
	Instructions: 

		1) GPU_LMFit is a GPU device function, so it can only be called from a GPU kernel 
		     function (__global__ function) or a GPU device function (__device__ function). 
		2) One data fitting is accomplished in one GPU block, where multiple threads run 
		     together to perform parallel Levenberg¨CMarquardt least-minimization. 
		3) Requirements for the GPU block dimensions and sizes: 
			a. Only one-dimensional block (blockDim.x) is used, i.e., blockDim.y = blockDim.z 
			    = 1. Here blockDim.x, blockDim.y and blockDim.z are internal CUDA variables 
			    to GPU block sizes (number of threads) in x-, y- and z-dimension, respectively. 
			b. blockDim.x must be power of 2, or otherwise GPU_LMFit writes BLOCKDIMERR 
			    (defined below) to InfoNum[blockIdx.x], if InfoNum is not NULL. Here blockIdx.x 
			    is internal CUDA variable to GPU block index in x-dimension. 
			c. Limited by the used GPU device compute capability, blockDim.x has maximum 
			    number of threads per block, which can be known by calling the CUDA function
			    cudaGetDeviceProperties. For example, the maximum value of blockDim.x is 
			    1024 for CUDA compute capability 2.0.
			d. Each GPU multiprocessor has only a limited amount of shared memory and 
			    registers, which are shared by all active blocks. It's usually a case of setting block 
			    size for optimal performance of a GPU-based parallel algorithm. For more 
			    detailed information, the user can refer to our  example program, or NVIDIA 
			    Toolkit Documentation, for example, "CUDA C Programming Guide.pdf", and 
			    "CUDA Occupancy Calculator.xls"; 
		4) Requirements for the GPU grid dimensions and sizes: 
			a. Only one-dimensional grid (gridDim.x) is used, i.e., gridDim.y = gridDim.z = 1. 
			    Here gridDim.x, gridDim.y and gridDim.z are internal CUDA variables to GPU grid 
			    sizes (number of blocks) in x-, y- and z-dimension, respectively. 
			b. Limited by the used GPU device compute capability, gridDim.x has maximum 
			    number of threads per block, which can be known by calling the CUDA function
			    cudaGetDeviceProperties. For example, the maximum value of gridDim.x is 
			    65535 for CUDA compute capability 2.0.
			c. The maximum value of gridDim.x can generally be considered very large, and in 
			    many cases it seems to allow thousands (or even more) of blocks to solve a large 
			    number of data fittings in one call to the kernel function. However, when your CUDA 
				Device Driver Mode (check with deviceQuery.exe in CUDA samples) is Windows 
				Display Driver Model (WDDM), too many blocks make GPU intensive computation 
				operations, and can make the GPU unresponsive to the operating system or the user 
				and appear completely "frozen" or "hung" for a int time. Windows 7 and Vista can 
				detect these GPU hang situations and recover a responsive desktop dynamically by 
				restarting the GPU device, and as a result, the processing kernel function is unexpectedly 
				stopped. To avoid this, the user can use a smaller value of gridDim.x, or refer to the 
				Microsoft online article "Timeout Detection and Recovery of GPUs" to change some 
				registry keys located under HKLM\System\CurrentControlSet\Control\GraphicsDrivers. 
				For example, the registry key TdrDelay, it is a number of REG_DWORD data type, and is the 
			    number of seconds that the GPU is allowed to delay the preempt Timeout Detection 
				and Recovery request from the Windows system. The default value is 2 seconds for
				Windows Vista. The other method to circumvent this problem is to use a smaller value
				of gridDim.x and successively launch kernel functions multiple times. 
		5) Requirements for shared memory:
			a. For a kernel function, the shared memory per block has to be possible to include
			    1) the static shared variables in the kernel function, the size of which can be 
			    known by calling the CUDA function cudaFuncGetAttributes, and 2) the dynamic 
			    shared variables externally allocated and passed to the kernel function when the 
			    kernel function is called.
			    As mentioned in Instruction 3)-d, each GPU multiprocessor has only a limited 
			    amount of shared memory for all active blocks, so the total size of the shared 
			    memory used for one block should not exceed the average size of the shared 
			    memory for the maximum number of active blocks per multiprocessor. 
			b. GPU_LMFit requires at least 2*blockDim.x*sizeof(_real_) bytes extern shared 
			    memory per block for internal warp reduction operations, so after the 
			    determination of blockDim.x, the number of threads per block, a extern shared 
			    memory with the size of at least a size of 
			    GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread*blockDim.x bytes needs to be 
			    allocated externally and passed to GPU_LMFit through the user's kernel function 
			    or device function. 
			c. To optimize the extern shared memory size, the function 
			   GPU_LMFit_Num_Of_SMEM_Vecs needs to be used. For more details, please refer to 
			   the below description for the input parameter SharedMem.

	Function Parameters:

		FitFuncPtr (input):
			A pointer to fit function to be minimized (must be device function).
		FitFuncPrivDataPtr (input): 
			A pointer to any private user data which is to be passed directly to the fit function 
			without any modification by GPU_LMFit.
		JacFuncPtr (input):
			A pointer to the user defined Jacobian function (must be device function);
			If JacFuncPtr==NULL or if the user defined Jacobian function returns -1, GPU_LMFit 
			will use the GPU-based parallel function fdjac2 (see Minpack or MPFit) to calculate 
			the finite-difference approximation to Jacobian.	
		JacFuncPrivDataPtr (input):
			A pointer to any private user data which is to be passed directly to the user defined 
			Jacobian function without any modification by GPU_LMFit.
		x (input and output):   
			The pointer of all fitting paramters for all fits. Its size should be at least (gridDim.x)*
			(the number of fitting parameters of each fit)*sizeof(_real_). As an input parameter,
			every fitting parameter in x must be given an initial value before passed to 
			GPU_LMFit. As an output parameter, it returns the solution x from GPU_LMFit.
		 UseLowerBounds (input):
			The pointer of an integer vector with the same length as x to disable (by zero 
			elements) or enable (by non-zeros) the low bound constraints (LowerBounds) on 
			the corresponding element of x. If no low bound constraints are applied to all fitting 
			parameters, it should be assigned to NULL.
		 LowerBounds
			The pointer of a _real_ vector with the same length as x and it contains the low bound 
			constraints of the corresponding element of x. Use the above parameter 
			UseLowerBounds to enable or disable the lower bound constraint for a specific 
			fitting parameter in x. If UseLowerBounds = NULL, it will be ignored in GPU-LMFit.
		 UseUpperBounds
			The same as for UseLowerBounds, but it is for upper bound constraints. 
		 UpperBounds
			The same as for LowerBounds, but it is for upper bound constraints. 
		InfoNum (output):
			The pointer of success- or error-status codes (see definitions below) returned from 
			GPU_LMFit if it is not set to be NULL. Its size should be at least (gridDim.x)*
			sizeof(int).
		Chisq (output): 
			The pointer of the values of fit function calculated at the solution x if it is not set to 
			be NULL. Its size should be at least (gridDim.x)*sizeof(_real_).
		SharedMem (input): 
			The pointer to the extern shared memory. Depending on the total size of this buffer, 
			GPU_LMFit may separate this SharedMem buffer into two parts: 1) the first part is 
			in the size of at least GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread*blockDim.x 
			bytes and used as a temporary buffer to enable warp reduction operations. 
			GPU_LMFit always first initializes this part of memory to zeros before using it, so 
			this part can also be used by user's other device functions (e.g., the fit function) as a 
			temporary buffer in the same way as GPU_LMFit does. However, this part used by 
			the user's functions may have a size different from that used by GPU_LMFit, and in 
			this case the user's functions used shared memory size needs to be passed to 
			GPU_LMFit through the last parameter - ExtTempSharedBufferSize. The final size of
			the first part of the SharedMem buffer is then actually the maximum value in between 
			ExtTempSharedBufferSize and GPU_LMFit_Extern_Shared_Mem_Size_Per_Thread*
			blockDim.x. 2) If this SharedMem buffer can still have some spare space after the 
			first part, GPU_LMFit will use the second part for its work variables. In this case, 
			GPU_LMFit will store its n- and m-element work vectors as many as possible in 
			shared memory for higher access speed. The user can use the function 
			GPU_LMFit_Num_Of_SMEM_Vecs() to determine how many GPU_LMfit work vectors 
			can be in shared memory. 
		SharedMemSize (input): 
			The total size (in bytes) of the SharedMem buffer.
		ExtTempSharedBufferSize (input): 
			The size (in bytes) of the extern shared memory used by user's functions in the 
			buffer SharedMem. 
*/
extern __device__ void GPU_LMFit(Device_FitFunc, void *, Device_JacFunc, void *, 
	_real_ *,   int *, _real_ *, int *, _real_ *, int *, _real_ *, void *, int, int);

/* Accessorial functions */
/* 
	Prototype: 
		extern void GPU_LMFit_Num_Of_SMEM_Vecs(int m, int n, int SMEM_Size, int *NumOfSMEMmVec, 
																						int *NumOfSMEMnVec);
	Function: 
		To determine the numbers *NumOfSMEMmVec and *NumOfSMEMnVec, which 
		are, respectively, the numbers of m- and n-element vectors which are allocated 
		external shared memory. 

	Inputs:
		m and n must be the same as those in GPU_LMFit;
		SMEM_Size is the maximum extern shared memory per block, which should be 
			calculated by subtracting the size of the static shared memory (can be known 
			by calling the CUDA function cudaFuncGetAttributes) in a kernel function from 
			the maximum size of shared memory per block. 

	Outputs:
		*NumOfSMEMmVec and *NumOfSMEMnVec.
*/
void GPU_LMFit_Num_Of_SMEM_Vecs(int, int, int, int *, int *);

/* 
	Prototype: 
		extern int GPU_LMFit_Config(int m, int n, LM_CRITERIA LM_Criteria, int BLKsPer1DGrid, 
															int NumOfSMEMmVec, int NumOfSMEMnVec, char *ErrMsg);
	Function: 
		Setup the necessary parameters and global memory working arrays for running 
		GPU_LMFit(). It is required to be performed before calling the function GPU_LMFit(). 
	Inputs:
		m and n must be the same as those in GPU_LMFit;
		LM_Criteria is user-defined LM criteria. 
		BLKsPer1DGrid is 1D grid size = the number of CUDA blocks.
		NumOfSMEMmVec and NumOfSMEMnVec are supposed to get from the function 
			GPU_LMFit_Num_Of_SMEM_Vecs. 
	Outputs:
		Return value is the size (bytes) of GPU-LMFit working arrays in global memory. It can be 
		negative if errors occur in the function, and in this case, a string message containing the 
		error information is given to the parameter ErrMsg. 
*/
extern int GPU_LMFit_Config(int, int, LM_CRITERIA, int, int, int, char *);

/* 
	Prototype: 
		extern void GPU_LMFit_Clear();
	Function: 
		To release the GPU_LMFit working arrays in the GPU global memory.  It should be 
		performed after calling the function GPU_LMFit().
*/
extern void GPU_LMFit_Clear();


/* 
	The below definitions are potential success status codes - Originally defined in subroutine 
	lmder of Minpack and MPFit

	InfoNum = 1  Chi-square value converged to a solution.
	InfoNum = 2  Change in the fitting parameters was less than the specified tolerance.
	InfoNum = 3  Convergence in the conditions for both InfoNum = 1 and InfoNum = 2.
	InfoNum = 4  Convergence in orthogonality.
	InfoNum = 5  Number of iterations reached.
	InfoNum = 6  Change in the fit function values was too small. No further optimization is 
					necessary.
	InfoNum = 7  Change in the fitting parameters was too small. No further optimization is 
					necessary.
	InfoNum = 8  Change in the orthogonality was too small. The values of the fit function are 
					orthogonal to the columns of the Jacobian to machine precision, and thus no
					further optimization is necessary.
*/
#define LM_OK_CHI (1)            
#define LM_OK_PAR (2)           
#define LM_OK_BOTH (3)           
#define LM_OK_DIR (4)       
#define LM_MAXITER (5) 
#define LM_FTOL (6)     
#define LM_XTOL (7) 
#define LM_GTOL (8)  


/* Error code possibly returned in InfoNum from GPU_LMFit */
#define INTERNALERR (-10000)                     // Detect an internal error. 
#define BLOCKDIMERR (-20)                   // blockDim.x are not power of 2 or CUDA blcok is not 1D
#define SHAREDMEMSIZEERR (-30)       // External shared memory size is smaller than (2*blockDim.x*sizeof(_real_))
#define NOGLOBALMEMORY (-40)           // No global memory for the buffers requires in GPU-LMFit
#define NLARGERTHANM (-50)		         // Number of fitting parameters is larger than number of data points
#define NLARGERTHANBDIM (-60)          // Number of fitting parameters is larger than blockDim.x
#define INFINITEJAC (-70)                         // Found non-finite values in Jacobian matrix 
#define BOUNDSERR (-80)                         // Lower bound >= upper bound for one or more fitting parameter(s)
#define LBERR (-90)                                    // The initial values of some fitting parameters are smaller than lower bounds
#define UBERR (-100)                                 // The initial values of some fitting parameters are larger than upper bounds
#define STOPBYFITFUNC (-1000)              // If user-defined fitting function returns -1, then GPU-LMFit stops immediately. 
#endif //_GPU_LMFIT_H_


/********************************************************************************************                                
					
						WARRANTY DISCLAIMER AND COPYRIGHT NOTICE


GPU_LMFit: A GPU-based parallel Levenberg¨CMarquardt least-squares minimization fitting 
			   library in CUDA C/C++ 

This software was developed based on the C version MPFit (see below).

Parallel computation algorithms and translation to CUDA C/C++ were applied by Xiang Zhu 
and Dianwen Zhang, at Image Technology Group, Beckman Institute for Advanced Science & 
Technology, University of Illinois at Urbana-Champaign. Our contact information can be found 
at http://itg.beckman.illinois.edu/.

GPU_LMFit Copyright (C) 2012, 2013, 2014, 2015 Xiang Zhu and Dianwen Zhang.  All rights reserved.

THIS SOFTWARE IS PROVIDED AS IS WITHOUT ANY WARRANTY WHATSOEVER.
Permission to use, copy and modify is hereby granted, provided that this copyright and disclaimer 
are included unchanged. Rental, leasing, sale and any other form of distribution are not allowed 
without our permission.

LIMITATION OF LIABILITY. In no event will the copyright holder be liable for any indirect, incidental, 
consequential, special or punitive damages of any kind or nature, including but not limited to loss of 
profits or loss of data, for any reason whatsoever, whether such liability is asserted on the basis of 
contract, tort (including negligence or strict liability).

Source code derived from MPFit must have the following disclaimer text provided.

============

MPFIT: A MINPACK-1 Least Squares Fitting Library in C

Original public domain version by B. Garbow, K. Hillstrom, J. More'
(Argonne National Laboratory, MINPACK project, March 1980)
Copyright (1999) University of Chicago
(see below)

Tranlation to C Language by S. Moshier (moshier.net)
(no restrictions placed on distribution)

Enhancements and packaging by C. Markwardt
(comparable to IDL fitting routine MPFIT 
see http://cow.physics.wisc.edu/~craigm/idl/idl.html)
Copyright (C) 2003, 2004, 2006, 2007 Craig B. Markwardt

This software is provided as is without any warranty whatsoever.
Permission to use, copy, modify, and distribute modified or unmodified copies is granted, 
provided this copyright and disclaimer are included unchanged.

Source code derived from MINPACK must have the following disclaimer text provided.

============

Minpack Copyright Notice (1999) University of Chicago.  All rights reserved

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
    and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of 
    conditions and the following disclaimer in the documentation and/or other materials 
    provided with the distribution.

3. The end-user documentation included with the redistribution, if any, must include the 
    following acknowledgment:
    
    "This product includes software developed by the University of Chicago, as Operator of 
    Argonne National Laboratory.¡±
     
     Alternately, this acknowledgment may appear in the software itself, if and wherever such 
     third-party acknowledgments normally appear.

4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF 
    ANY KIND. THE COPYRIGHT HOLDER, THE UNITED STATES, THE UNITED STATES 
    DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, 
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF 
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, 
    (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, 
    COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE 
    OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT 
    WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-
    FREE OR THAT ANY ERRORS WILL BE CORRECTED.

5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT HOLDER, THE UNITED 
    STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES: BE LIABLE 
    FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF 
    ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS OR LOSS OF 
    DATA, FOR ANY REASON WHATSOEVER, WHETHER SUCH LIABILITY IS ASSERTED ON THE 
    BASIS OF CONTRACT, TORT (INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE, 
    EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE POSSIBILITY OF SUCH LOSS OR 
    DAMAGES.
    
*********************************************************************************************/