#ifndef _GPU_LMFIT_CU_
#define _GPU_LMFIT_CU_


#include <stdio.h> // To allow the use of the function printf in the function GetLibInfo.  
#include <string.h> // To allow the use of the function strlen in the function GetLibInfo.  
#include <math.h> // floor()

#include "cuda_runtime_api.h" // To use CUDA C/C++ 

#include "GPU_LMFit.cuh"


/* Definations for compiling */
#define __DEBUG_CHKOVERFLOW__

/* Definitions */
#define Max_Num_Of_n_Length_Vectors 8

/* Threads and Blocks indices */
#define tidx threadIdx.x
#define bidx blockIdx.x
#define bdim blockDim.x
#define gdim gridDim.x

/* Determine the data type of LMFIT */
#ifdef _SINGLE_FLOAT_ 
	/* Default Termination Creiteria */
	__constant__ LM_CRITERIA Default_LM_Criteria = {1e-4f, 1e-4f, 1e-5f, MACHINE_EPSILON, 20.0f, 25};
#else //_SINGLE_FLOAT_ 
	/* Default Termination Creiteria */
	__constant__ LM_CRITERIA Default_LM_Criteria = {1e-4, 1e-4, 1e-5, MACHINE_EPSILON, 20.0, 25};
#endif //_SINGLE_FLOAT_ 


/* Commonly used variables */
__shared__ _real_ *sp_buffer;

// Shared pointers and variables for GPU LMFit
// sv_ -> shared variables, sp_ -> shared pointers;
__shared__ int sv_LMPar_complete, sv_nfev, sv_iter, sv_InfoNum;
__shared__ _real_ sv_sum, sv_temp, sv_temp1, sv_temp2, sv_temp3, sv_alpha, 
						   sv_gnorm, sv_pnorm, sv_prered, sv_cosx, sv_sinx, sv_cotan, 
						   sv_tanx, sv_qtbpj, sv_actred, sv_delta, sv_dirder, sv_fnorm,
						   sv_fnorm1, sv_par, sv_ratio, sv_xnorm;
__shared__ _real_ *sp_wa1, *sp_wa2, *sp_wa3, *sp_qtf, *sp_x, *sp_x1, *sp_x2, *sp_diag;
__shared__ _real_ *sp_fvec,  *sp_wa4,  *sp_fjac;

// both sv_n and sv_m should be only used in __device__ functions, because they are registers in LMFit_Kernel.
__shared__ int sv_n, sv_m; // n - number of variables; m - number of data in each a fit. 

/* Basic LMFit precision control parameters */
__shared__ LM_CRITERIA sv_GPU_LM_Config; // shared variable, must be initialized externally.

/* External Functions */
__shared__ Device_FitFunc sp_GPU_FitFunctionPtr; 
__shared__ Device_JacFunc sp_GPU_JacFunctionPtr; 

/* Shared pointers and variable for fit function */
__shared__ char *sp_FitFuncPrivDataPtr;
__shared__ char *sp_JacFuncPrivDataPtr;

/*
GPU_LMFit_WA: 
The pointer of a device global memory buffer used by GPU_LMFit to temporarily 
store work variables. The size of this buffer needs to be determined as follows:

(gridDim.x)*(GPU_LMFit_Single_Block_Buffer_Size(n, m) -
NumOfnSVec*n*sizeof(_real_)-NumOfmSVec*m*sizeof(_real_));

with the help of the function GPU_LMFit_Single_Block_Buffer_Size() which is defined
in GPU_LMFit.cuh. NumOfnSVec and NumOfmSVec are, respectively, the numbers of 
n- and m-element vectors which have been allocated external shared memory. Both 
NumOfnSVec and NumOfmSVec can be determined using the function 
GPU_LMFit_Num_Of_SMEM_Vecs(), which prototype can be found in GPU_LMFit.cuh.			
*/
__constant__ _real_ *GPU_LMFit_WA_Ptr=NULL;

int GPU_LMFit_WA_Size=0;
_real_ *GPU_LMFit_WA=NULL; // _real_ global buffer

/* Maximal size of LMFIT_REAL_MEM needed in global memory*/
#define SINGLE_GPU_LMFIT_REAL_MEMSIZE(n, m) \
		(2*m*sizeof(_real_) +  \
		m*n*sizeof(_real_) + \
		8*n*sizeof(_real_))

/*		
NumOfnSVec: 
The numbers of n-element vectors which have been allocated external shared
memory. It must be obtained from the function GPU_LMFit_Num_Of_SMEM_Vecs() 
in GPU_LMFit.cuh.
NumOfmSVec:
The numbers of m-element vectors which have been allocated external shared
memory. It must be obtained from the function GPU_LMFit_Num_Of_SMEM_Vecs() 
in GPU_LMFit.cuh.
*/
__constant__ int NumOfmSVec=-1;
__constant__ int NumOfnSVec=-1;

/*
NumOfPoints = m:  The total number of the data points of the curve to be fitted.
*/
__constant__ int NumOfPoints=0;

/*
NumOfParameters = n: The total number of the fitting parameters of the fitting function.
*/
__constant__ int NumOfParameters=0;

/*
Users_LM_Criteria:
A LM_CRITERIA structure for all termination and configuration creiteria. For the 
instruction of the use of this parameter, please see the above discription for the 
LM_CRITERIA structure. 	
*/
__constant__ LM_CRITERIA Users_LM_Criteria={zero, zero, zero, zero, zero, 0};


/* Error handling */
bool ChkCudaError(cudaError errnum, char *ErrMsg, const char *InfoString)
{
	if (errnum != cudaSuccess){
		sprintf(ErrMsg, "%s : %s - '%s'.", __FUNCTION__, InfoString, cudaGetErrorString(errnum));
		return true;
	}
	else
		return false;
}


/* 
External call for the number of n- or m-length vectors of GPU_LMFit 
which can be in shared memory.
*/
void GPU_LMFit_Num_Of_SMEM_Vecs(int m, int n, int SMEM_Size, int *NumOfSMEMmVec, 
											int *NumOfSMEMnVec)
{
	// To find the number of n-element vectors which can be stored in the shared memory  
	*NumOfSMEMnVec = (int)floor((SMEM_Size+0.0f)/(n*sizeof(_real_)));
	*NumOfSMEMnVec = MIN(*NumOfSMEMnVec, Max_Num_Of_n_Length_Vectors);
	*NumOfSMEMnVec = MAX(*NumOfSMEMnVec, 0);
	// After all n-elements vectors can be stored in the shared memory, the below is to find out 
	// how many m-element vectors can still be in the share memory.
	*NumOfSMEMmVec = (int)floor((SMEM_Size-
		(*NumOfSMEMnVec)*n*sizeof(_real_)+0.0f)/(m*sizeof(_real_)));
	*NumOfSMEMmVec = MAX(*NumOfSMEMmVec, 0);
	if((*NumOfSMEMmVec)>=(2+n)) //fjac£¬fvec£¬wa4 may be in the shared memory
		*NumOfSMEMmVec=2+n; 
	else if((*NumOfSMEMmVec)<(2+n) && (*NumOfSMEMmVec)>=2) 
		*NumOfSMEMmVec=2; // fvec and wa4 may be in the shared memory
	else if((*NumOfSMEMmVec)<2 && (*NumOfSMEMmVec)>=1) 
		*NumOfSMEMmVec=1; // wa4 may be in the shared memory
	else
		*NumOfSMEMmVec=0;
}


/* GPU_LMFit_Config() should be called before GPU-LMFit() to settle down all common parameters in the GPU memory */
int GPU_LMFit_Config(int m, int n, LM_CRITERIA LM_Criteria, int BLKsPer1DGrid, 
	int NumOfSMEMmVec, int NumOfSMEMnVec, char *ErrMsg)
{
	GPU_LMFit_WA_Size = BLKsPer1DGrid*(SINGLE_GPU_LMFIT_REAL_MEMSIZE(n, m) -
		NumOfSMEMnVec*n*sizeof(_real_)-NumOfSMEMmVec*m*sizeof(_real_));

	if(GPU_LMFit_WA_Size){
		GPU_LMFit_Clear();
		if(ChkCudaError(cudaMalloc((void **)&GPU_LMFit_WA, GPU_LMFit_WA_Size), 
			ErrMsg, "GPU_LMFit_WA : cudaMalloc")) return(-__LINE__);
	}
	else 	GPU_LMFit_WA = NULL;

	if(ChkCudaError(cudaMemcpyToSymbol(GPU_LMFit_WA_Ptr, &GPU_LMFit_WA, sizeof(_real_ *),
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"GPU_LMFit_WA to GPU_LMFit_WA_Ptr : cudaMemcpyToSymbol")) return(-__LINE__);
	if(ChkCudaError(cudaMemcpyToSymbol(NumOfPoints, &m, sizeof(int), 
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"m to NumOfPoints : cudaMemcpyToSymbol")) return(-__LINE__);
	if(ChkCudaError(cudaMemcpyToSymbol(NumOfParameters, &n, sizeof(int), 
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"n to NumOfParameters : cudaMemcpyToSymbol")) return(-__LINE__);
	if(ChkCudaError(cudaMemcpyToSymbol(Users_LM_Criteria, &LM_Criteria, sizeof(LM_CRITERIA), 
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"n to NumOfParameters : cudaMemcpyToSymbol")) return(-__LINE__);
	if(ChkCudaError(cudaMemcpyToSymbol(NumOfmSVec, &NumOfSMEMmVec, sizeof(int), 
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"NumOfSMEMmVec to NumOfmSVec : cudaMemcpyToSymbol")) return(-__LINE__);
	if(ChkCudaError(cudaMemcpyToSymbol(NumOfnSVec, &NumOfSMEMnVec, sizeof(int), 
		0, cudaMemcpyHostToDevice), ErrMsg, 
		"NumOfSMEMnVec to NumOfnSVec : cudaMemcpyToSymbol")) return(-__LINE__);

	return(GPU_LMFit_WA_Size);
}


/* GPU_LMFit_Config() should be called after GPU-LMFit() to free all working memory in GPU */
void GPU_LMFit_Clear()
{
	if(GPU_LMFit_WA) {
		cudaFree(GPU_LMFit_WA);
		GPU_LMFit_WA = NULL;
		cudaMemcpyToSymbol(GPU_LMFit_WA_Ptr, &GPU_LMFit_WA, sizeof(_real_ *), 
			0, cudaMemcpyHostToDevice);
	}
}


/* ************************************************************************************************************
                                                                                       function enorm 
Note: 
	Euclidean Norm computation for int vector with warp redution;
      This function is to calculate the norm of the first Compute_Length elements in each vector Vec_Data in a block.
**************************************************************************************************************/
__device__ void GPU_ENorm(const _real_ *Vec_Data, const int Compute_Length, _real_ *Vec_Norm)
{	
	int i0;

	/* Initialize shared buffer sp_buffer */
	sp_buffer[tidx] = zero;
	sp_buffer[tidx+bdim] = zero;

	/* Start simple Euclidean Norm computation */
	for(i0=0; i0<Compute_Length; i0+=(2*bdim)) {
		if((i0+tidx) < Compute_Length) 
			sp_buffer[tidx] += Vec_Data[i0+tidx]*Vec_Data[i0+tidx];
		if((i0+tidx+bdim) < Compute_Length)
			sp_buffer[bdim+tidx] += Vec_Data[i0+bdim+tidx]*Vec_Data[i0+bdim+tidx];
	}
	__syncthreads(); 
	for (i0=bdim; i0>=1; i0>>=1){ 	//Warp Reduction 
		if (tidx < i0)
			sp_buffer[tidx] += sp_buffer[tidx + i0];
		if (i0>=warpSize) 
			__syncthreads(); 
	} 
	// Return the result
	if(tidx == 0) *Vec_Norm = cuSQRT(sp_buffer[0]);
}


/* ************************************************************************************************************
                                                                                       function FDJac2 
**************************************************************************************************************/
__device__ int GPU_FDJac2(_real_ *x,  _real_ *wa,  _real_ *fjac)
{
#define h sv_temp1

	__shared__ _real_ eps;
	__shared__ int i1;

	int i0;

	if(tidx==0){
		sv_temp = MAX(sv_GPU_LM_Config.epsfcn, MACHINE_EPSILON); 
		eps = cuSQRT(sv_temp);
	}
	
	/* Initialize the Jacobian derivative matrix */
	for (i0=0; i0<sv_m*sv_n; i0+=bdim)
		if((i0+tidx)<sv_m*sv_n) fjac[i0+tidx] = zero;

	/* Any parameters requiring numerical derivatives */
	if(tidx==0) i1=0;
	__syncthreads();

	while(i1<sv_n) {  /* Loop thru free parms */

		if(tidx==0){
			sv_temp = x[i1];
			h = eps * cuFABS(sv_temp);
			if (h == zero) h = eps;
			x[i1] = sv_temp + h;
		}
		__syncthreads();
	
		/* Call fit function */
		if((*sp_GPU_FitFunctionPtr)(x, wa, sp_FitFuncPrivDataPtr) != -1)
			__syncthreads();
		else
			return(-1); // stop by user-defined fitting function

		if(tidx==0) {
			sv_nfev +=1;
			x[i1] = sv_temp;
		}
	
		/* COMPUTE THE TWO-SIDED DERIVATIVE */
		for (i0=0; i0<sv_m; i0+=bdim)
			if((i0+tidx)<sv_m) fjac[i1*sv_m+(i0+tidx)] = wa[i0+tidx];

		/* Evaluate at x - h */
		if(tidx==0) x[i1] = sv_temp - h;
		__syncthreads();
		/* Call fit function */
		if((*sp_GPU_FitFunctionPtr)(x, wa, sp_FitFuncPrivDataPtr) != -1)
			__syncthreads();
		else
			return(-1); // stop by user-defined fitting function

		if(tidx==0){
			sv_nfev +=1;
			x[i1] = sv_temp;
		}

		/* Now compute derivative as (f(x+h) - f(x-h))/(2h) */
		for (i0=0; i0<sv_m; i0+=bdim)
			if((i0+tidx)<sv_m) fjac[i1*sv_m+(i0+tidx)] = 
				(fjac[i1*sv_m+(i0+tidx)] - wa[i0+tidx])/(two*h); 
		__syncthreads(); 

		if(tidx==0) i1+=1;
		__syncthreads();
	}
	
	// Normal return
	return(0);

#undef h
}


/* ************************************************************************************************************
                                                                                       function QRFac 
**************************************************************************************************************/
__device__ void GPU_QRFac(_real_ *a)
{
	/* 
		Version Nov-26-2012
	*/

#define ajnorm sv_temp1
#define rdiag sp_wa1
#define acnorm sp_wa2
#define wa sp_wa3

	__shared__ int i, k, ii, jj;
		
	int i0;

	/* compute the initial column norms and initialize several arrays. */
	for (i0=0; i0<sv_n; i0++) {
		GPU_ENorm(a+i0*sv_m, sv_m, &acnorm[i0]);
		__syncthreads();
	}
	if(tidx<sv_n) {
		rdiag[tidx] = acnorm[tidx];
		wa[tidx] = acnorm[tidx];
	}
	__syncthreads();

	/* reduce a to r with householder transformations. */
	if(tidx==0) i=0;
	__syncthreads();

	while (i<(sv_n<sv_m?sv_n:sv_m-1)) { // Start i-for-loop

		if(tidx==0) ii = i+sv_m*i;
		__syncthreads(); // Wait until ii is updated

		/* compute the householder transformation to reduce the
		*	 i1-th column of a to a multiple of the i1-th unit vector. */

		GPU_ENorm(a+ii, sv_m-i, &ajnorm);
		__syncthreads(); 

		if(ajnorm == zero) goto L100; 

		/* MPQRFac_Part2 */
		if(tidx==0 && a[ii]<zero) ajnorm = -ajnorm;
		__syncthreads(); // Wait until N0 is updated
		for(i0=0; i0<sv_m; i0+=bdim){
			if((i0+tidx)<(sv_m-i))
				a[ii+(i0+tidx)] /= ajnorm;
		}
		if(tidx==0)
			a[ii] += one;

		if(tidx==0) k=i+1;
		__syncthreads();
		while (k<sv_n){ // start k-for-loop

			if(tidx==0)	jj = i + sv_m*k;
			__syncthreads(); // Wait until the shared pointers and variables are initialized.

			sp_buffer[tidx] = zero;
			sp_buffer[tidx+bdim] = zero;
			__syncthreads(); // Wait for all threads to complete the above SVec initialization. 

			for(i0=0; i0<(sv_m-i); i0+=(2*bdim)) {
				if((i0+tidx)<(sv_m-i))
					sp_buffer[tidx] += a[i0+ii+tidx]*a[i0+jj+tidx];
				if((i0+tidx+bdim)<(sv_m-i))
					sp_buffer[bdim+tidx] += a[i0+ii+bdim+tidx]*a[i0+jj+bdim+tidx];
			}
			__syncthreads(); 
			for (i0=bdim; i0>=1; i0>>=1){ 	//Warp Reduction 
				if (tidx < i0)
					sp_buffer[tidx] += sp_buffer[tidx + i0];
				if (i0>=warpSize) 
					__syncthreads(); 
			} // sp_buffer[0] are the real value of sum in CPU LM_QRFac

			if (tidx==0) {
				sv_sum = sp_buffer[0];
				sv_temp = sv_sum/a[ii];
			}
			__syncthreads();

			for(i0=0; i0<sv_m; i0+=bdim){
				if((i0+tidx)<(sv_m-i))
					a[jj+(i0+tidx)] = a[jj+(i0+tidx)] - sv_temp*a[ii+(i0+tidx)];
			}
			__syncthreads(); // Wait for updating all a

			if(rdiag[k] != zero) {
				__syncthreads(); // To prevent rdiag[k] from being changed prior to some threads. 
				if(tidx == 0){
					sv_temp = a[i+sv_m*k]/rdiag[k];
					sv_temp = MAX(zero, (one-sv_temp*sv_temp));  
					rdiag[k] *= cuSQRT(sv_temp);
					sv_temp = rdiag[k]/wa[k];
				}
				__syncthreads(); // Wait for all above parameters updated.
				if ((p05*sv_temp*sv_temp) <= MACHINE_EPSILON){
					GPU_ENorm(a+i+1+sv_m*k, sv_m-i-1, &rdiag[k]);
					__syncthreads();  
					if(tidx==0) wa[k] = rdiag[k];
				}
			}
			__syncthreads(); 

			if(tidx==0) k+=1;
			__syncthreads();
		} // end of k-for-loop

L100:
		/* MPQRFac_Part5 */
		if(tidx==0) rdiag[i] = -ajnorm;
		__syncthreads(); 

		if(tidx==0) i+=1;
		__syncthreads();
	} // end of i-for-loop

	/* Start the correction made on Jun 20, 2014 for m<=n */
	for(i0=i; i0<sv_n; i0+=bdim){
		if((i0+tidx)<sv_n)
			rdiag[i0+tidx] = a[sv_m*(i0+tidx)+(sv_m-1)];
	}
	__syncthreads(); // Wait for updating all a
	/* End the correction made on Jun 20, 2014 for m<=n */

#undef ajnorm
#undef rdiag
#undef acnorm
#undef wa
}


/* ************************************************************************************************************
                                                                                       function QRSolv 
**************************************************************************************************************/
__device__ void GPU_QRSolv(_real_ *r,  _real_ *diag, const _real_ *qtb,  _real_ *x,  _real_ *sdiag,  _real_ *wa)
{
	__shared__ int i0, i1, i2, i3, i4, nsing;

	for (i1=0; i1<sv_n; i1++) {
		for (i0=i1; i0<sv_n; i0++)
			r[i1*(sv_m+1)+i0-i1] = r[i1*(sv_m+1)+(i0-i1)*sv_m];
		x[i1] = r[i1*(sv_m+1)];
		wa[i1] = qtb[i1];
	}

	/*
	*     eliminate the diagonal matrix d using a givens rotation.
	*/
	for (i1=0; i1<sv_n; i1++) {
		/*
		*	 prepare the row of d to be eliminated,  locating the
		*	 diagonal element using p from the qr factorization.
		*/
		if (diag[i1] == zero) {
			sdiag[i1] = r[i1+sv_m*i1];
			r[i1+sv_m*i1] = x[i1];
			continue;
		}
		for (i0=i1; i0<sv_n; i0++)
			sdiag[i0] = zero;
		sdiag[i1] = diag[i1];
		/*
		*	 the transformations to eliminate the row of d
		*	 modify only a single element of (Q transpose)*b
		*	 beyond the first n,  which is initially zero.
		*/
		sv_qtbpj = zero;
		for (i2=i1; i2<sv_n; i2++) {
			/*
			*	    determine a givens rotation which eliminates the
			*	    appropriate element in the current row of d.
			*/
			if (sdiag[i2] == zero) continue;
			if (cuFABS(r[i2+sv_m*i2]) < cuFABS(sdiag[i2])) {
				sv_cotan = r[i2+sv_m*i2]/sdiag[i2];
				sv_sinx = p5/cuSQRT(p25+p25*sv_cotan*sv_cotan);
				sv_cosx = sv_sinx*sv_cotan;
			}
			else {
				sv_tanx = sdiag[i2]/r[i2+sv_m*i2];
				sv_cosx = p5/cuSQRT(p25+p25*sv_tanx*sv_tanx);
				sv_sinx = sv_cosx*sv_tanx;
			}

			/*
			*	    compute the modified diagonal element of R and
			*	    the modified element of ((Q transpose)*b, 0).
			*/
			r[i2+sv_m*i2] = sv_cosx*r[i2+sv_m*i2] + sv_sinx*sdiag[i2];
			sv_temp = sv_cosx*wa[i2] + sv_sinx*sv_qtbpj;
			sv_qtbpj = -sv_sinx*wa[i2] + sv_cosx*sv_qtbpj;
			wa[i2] = sv_temp;
			/*
			*	    accumulate the tranformation in the row of s.
			*/
			if (sv_n > i2+1)
			{
				i3 = i2+sv_m*i2+1; // i3 is i7 in parallel CPU code
				for (i0=i2+1; i0<sv_n; i0++)
				{
					sv_temp = sv_cosx*r[i3] + sv_sinx*sdiag[i0];
					sdiag[i0] = -sv_sinx*r[i3] + sv_cosx*sdiag[i0];
					r[i3] = sv_temp;
					i3 += 1; 
				}
			}
		}
		/*
		*	 store the diagonal element of s and restore
		*	 the corresponding diagonal element of R.
		*/
		sdiag[i1] = r[i1+sv_m*i1];
		r[i1+sv_m*i1] = x[i1];
	}

	/*
	*     solve the triangular system for z. if the system is
	*     singular,  then obtain a least squares solution.
	*/
	nsing = sv_n;
	for (i1=0; i1<sv_n; i1++) {
		if ((sdiag[i1] == zero) && (nsing == sv_n))
			nsing = i1;
		if (nsing < sv_n)
			wa[i1] = zero;
	}
	if (nsing >= 1){
		for (i2=0; i2<nsing; i2++) {
			i1 = nsing-i2-1;
			sv_sum = zero;
			i4 = i1 + 1;
			if (nsing > i4)
			{
				i3 = i4 + sv_m * i1;
				for (i0=i4; i0<nsing; i0++)
				{
					sv_sum += r[i3]*wa[i0];
					i3 += 1;
				}
			}
			wa[i1] = (wa[i1] - sv_sum)/sdiag[i1];
		}
	}
	/*
	*     permute the components of z back to components of x.
	*/
	for (i1=0; i1<sv_n; i1++) 
		x[i1] = wa[i1];

}


/* ************************************************************************************************************
                                                                                       function LMPar 
**************************************************************************************************************/
__device__ void GPU_LMPar(_real_ *r, const _real_ *diag, const _real_ *qtb, 
	_real_ *x, _real_ *sdiag, _real_ *wa1, _real_ *wa2)
{
	/*
		Nov -18 - 2012
	*/

#define dxnorm sv_temp1
#define fp sv_temp2
#define parl sv_temp3
#define parc sv_alpha
#define paru sv_pnorm
#define LMPar_complete sv_LMPar_complete

	__shared__ int iter, nsing, i0, i1, i2, i3, i4;
	
	LMPar_complete = 0;

	/*
	*     compute and store in x the gauss-newton direction. if the
	*     jacobian is rank-deficient,  obtain a least squares solution.
	*/
	nsing = sv_n;
	for (i0=0; i0<sv_n; i0++) {
		i4 = i0*(sv_m+1);
		wa1[i0] = qtb[i0];
		if ((r[i4] == zero) && (nsing == sv_n))
			nsing = i0;
		if (nsing < sv_n)
			wa1[i0] = zero;
	}

	if (nsing >= 1){
		for (i2=0; i2<nsing; i2++) { // nsing is different for each thread, so it is impossible for parallel computing. 
			i1 = nsing - i2 - 1;
			sv_temp = wa1[i1];
			sv_temp /= r[i1+sv_m*i1];
			wa1[i1] = sv_temp;
			i4 = i1 - 1;
			if (i4 >= 0){
				i3 = sv_m * i1;
				for (i0=0; i0<=i4; i0++){
					wa1[i0] -= r[i3+i0]*sv_temp;
					//wa1[i0] = fma(r[bidx*ldr*n+i3+i0], -sv_temp, wa1[i0]);
				}
			}
		}
	}

	for (i0=0; i0<sv_n; i0++) 
		x[i0] = wa1[i0];

	iter = 0;
	for (i1=0; i1<sv_n; i1++)
		wa2[i1] = diag[i1]*x[i1];
	//start for dxnorm[fn] = LM_enorm(n, &wa2[fn*ldr]);
	dxnorm = zero;
	for (i0=0; i0<sv_n; i0++) dxnorm += (wa2[i0]*wa2[i0]);
	dxnorm = cuSQRT(dxnorm);
	//end for dxnorm[fn] = LM_enorm(n, &wa2[fn*ldr]);
	fp = dxnorm - sv_delta;
	if (fp <= p1*sv_delta) {
		if (iter == 0)
			sv_par = zero;
		LMPar_complete = 1; 
	}

	/*
	*     if the jacobian is not rank deficient,  the newton
	*     step provides a lower bound,  parl,  for the zero of
	*     the function. otherwise set this bound to zero.
	*/
	if(LMPar_complete) return;

	if (nsing >= sv_n) {
		for (i1=0; i1<sv_n; i1++)
			wa1[i1] = diag[i1]*(wa2[i1]/dxnorm);
		for (i1=0; i1<sv_n; i1++){
			i4 = i1*sv_m;
			i2 = i1 - 1;
			sv_sum = zero;
			if (i2 >= 0)
				for (i0=0; i0<=i2; i0++)
					sv_sum += r[i4+i0]*wa1[i0];
			wa1[i1] = (wa1[i1] - sv_sum)/r[i1+sv_m*i1];
		}
		//The below is for the CPU code: temp[fn] = LM_enorm(n, &wa1[fn*n]);
		sv_temp = zero;
		for (i0=0; i0<sv_n; i0++) sv_temp += (wa1[i0]*wa1[i0]);
		sv_temp = cuSQRT(sv_temp);
		parl = ((fp/sv_delta)/sv_temp)/sv_temp;
	}
	else
		parl = zero;

	for (i1=0; i1<sv_n; i1++) {
		i4 = i1*sv_m;
		sv_sum = zero;
		for (i0=0; i0<=i1; i0++)
			sv_sum += r[i4+i0]*qtb[i0];
		wa1[i1] = sv_sum/diag[i1];
	}
	//The below is for the CPU code: gnorm[fn] = LM_enorm(n, &wa1[fn*n]);
	sv_gnorm = zero;
	for (i0=0; i0<sv_n; i0++) sv_gnorm += (wa1[i0]*wa1[i0]);
	sv_gnorm = cuSQRT(sv_gnorm);
	paru = sv_gnorm/sv_delta;
	if (paru == zero)
		paru = MACHINE_FLOAT_MIN/MIN(sv_delta, p1); 
	/*
	*     if the input par lies outside of the interval (parl, paru), 
	*     set par to the closer endpoint.
	*/
	sv_par = MAX(sv_par, parl); 
	sv_par = MIN(sv_par, paru); 
	if (sv_par == zero)
		sv_par = sv_gnorm/dxnorm;

	/*
	*     beginning of the while-loop in the LM_LMPar function.
	*/
	while(!LMPar_complete){

		iter += 1;
		//evaluate the function at the current value of par.
		if (sv_par == zero)
			sv_par = MAX(MACHINE_FLOAT_MIN, (p001*paru)); 
		sv_temp = cuSQRT(sv_par);
		for (i1=0; i1<sv_n; i1++)
			wa1[i1] = sv_temp*diag[i1];

		/* Call LM_QRSolv */
		// if ratio > p0001, InfoNum!=0 and LMPar_complete!=0, do NOT call the below function.
		GPU_QRSolv(r, wa1, qtb, x, sdiag, wa2);

		for (i1=0; i1<sv_n; i1++)
			wa2[i1] = diag[i1]*x[i1];
		//The below is for the CPU code: dxnorm[fn] = LM_enorm(n, &wa2[fn*ldr]);
		dxnorm = zero;
		for (i0=0; i0<sv_n; i0++) dxnorm += (wa2[i0]*wa2[i0]);
		dxnorm = cuSQRT(dxnorm);
		sv_temp = fp;
		fp = dxnorm - sv_delta;

		/*
		*	 if the function is small enough,  accept the current value
		*	 of par. also test for the exceptional cases where parl
		*	 is zero or the number of iterations has reached 10.
		*/
		if ((cuFABS(fp) > p1*sv_delta)	&& ((parl != zero) || (fp > sv_temp) || (sv_temp >= zero)) && (iter < 10)){
			/*
			*	 compute the newton correction.
			*/
			for (i1=0; i1<sv_n; i1++) 
				wa1[i1] = diag[i1]*(wa2[i1]/dxnorm);
			i4 = 0;
			for (i1=0; i1<sv_n; i1++) {
				wa1[i1] = wa1[i1]/sdiag[i1];
				sv_temp = wa1[i1];
				i2 = i1 + 1;
				if (i2 < sv_n)
				{
					i3 = i2 + i4;
					for (i0=i2; i0<sv_n; i0++)
					{
						wa1[i0] -= r[i3]*sv_temp;
						i3 += 1; 
					}
				}
				i4 += sv_m; 
			}
			//The below is for the CPU code: temp[fn] = LM_enorm(n, &wa1[fn*n]);
			sv_temp = zero;
			for (i0=0; i0<sv_n; i0++) sv_temp += (wa1[i0]*wa1[i0]);
			sv_temp = cuSQRT(sv_temp);
			parc = ((fp/sv_delta)/sv_temp)/sv_temp;
			// depending on the sign of the function,  update parl or paru[fn].
			if (fp > zero)
				parl = MAX(parl, sv_par); 
			if (fp < zero)
				paru = MIN(paru, sv_par); 
			// compute an improved estimate for par. 
			sv_par = MAX(parl, (sv_par + parc)); 
			// end of an iteration. 
		}
		else
		{
			LMPar_complete = 1;
			if (iter == 0)
				sv_par = zero;
		}

	} // End of the while-loop in the LM_LMPar function.

#undef dxnorm 
#undef fp
#undef parl 
#undef parc 
#undef paru
#undef LMPar_complete

}


/* ************************************************************************************************************
                                                                                       function LMFit 
**************************************************************************************************************/
__device__ void GPU_LMFit(Device_FitFunc FitFuncPtr, void *FitFuncPrivDataPtr, Device_JacFunc JacFuncPtr, 
	void *JacFuncPrivDataPtr, _real_ *x, int *UseLowerBounds, _real_ *LowerBounds, int *UseUpperBounds, 
	_real_ *UpperBounds, int *InfoNum, _real_ *Chisq, void *SharedMem, int SharedMemSize, 
	int ExtTempSharedBufferSize)
{
	/* Shared pointers to bound constraints of fitting parameter */
	__shared__ int *sp_UseLowerBounds, *sp_UseUpperBounds;   // Are bounds used?
	__shared__ _real_ *sp_LowerBounds, *sp_UpperBounds; // Bound constraints of fitting parameters
	__shared__ int sv_lpegged, sv_upegged;
	__shared__ _real_ sv_alpha;

	/* Iterative numbers */
	__shared__ int i3, i4;
	int i0, i1;

	/* Check validation of bounds */
	if((UseLowerBounds && !LowerBounds) || (UseUpperBounds && !UpperBounds)) {
		if(tidx==0) InfoNum[bidx] = BOUNDSERR;
		return;
	}

	if(tidx==0){
		/* LMFIT Pointers and variables */
		sv_n = NumOfParameters; 
		sv_m = NumOfPoints;
		sv_InfoNum = 0;
		if(sv_n>bdim){
			sv_InfoNum = NLARGERTHANBDIM;
			goto ChkPoint; 
		}
		if(sv_n>sv_m){
			sv_InfoNum = NLARGERTHANM;
			goto ChkPoint; 
		}
		if(bdim!=nearest_pow2(bdim) || blockDim.y!=1 || blockDim.z!=1) { 
			// bdim must be power of 2 or CUDA blcok is not 1D.
			sv_InfoNum = BLOCKDIMERR;
			goto ChkPoint; 
		}
		if(SharedMemSize<(MAX(2*bdim*sizeof(_real_), ExtTempSharedBufferSize)+
			NumOfnSVec*sv_n*sizeof(_real_)+
			NumOfmSVec*sv_m*sizeof(_real_))){
			// external shared memory size is at least (2*bdim*sizeof(_real_)).
			sv_InfoNum = SHAREDMEMSIZEERR;
			goto ChkPoint; 
		}
		if(NumOfmSVec<(2+sv_n) && GPU_LMFit_WA_Ptr==NULL){
			sv_InfoNum = NOGLOBALMEMORY;
			goto ChkPoint; 
		}
	}

ChkPoint:	
	__syncthreads(); 
	if(sv_InfoNum) {
		if(InfoNum)
			*InfoNum = sv_InfoNum;
		return; 
	}

	if(tidx==0){ // Array Pointers' initialization		
		/* Arrays must be in shared memory */
		sp_buffer = (_real_ *)SharedMem;

		/* Determine how many GPU_LMFit vector variables can be in shared memory SharedMem
		     GPU_LMFit shared memory configuration
			---------------------------------------------------------------------------------------------------------------------------------------------------------
			|  temporary buffer:                                                                |  Fixed buffer for some GPU_LMFit                               |
			|  1. always require initialization before use.                       |    variables if the SharedMemSize is large                   |
			|  2. the size is MAX(2*bdim*sizeof(_real_),                          |    enough, but it is also possible that                            |
			|      ExtTempSharedBufferSize);  2*bdim*                           |     this part does not exist if the SharedMemSize        |
			|      sizeof(_real_) is for GPU_LMFit warp reduction           |     is just enough for the front buffer only.                   |
			|      and ExtTempSharedBufferSize is for external             |                                                                                           |
			|      functions.                                                                          |                                                                                           |
			----------------------------------------------------------------------------------------------------------------------------------------------------------	   
		*/

		/* Arrays are possible to be in shared memory */
		// NumOfnSVec determines how much n-length vectors can be in shared memory.
		if(NumOfnSVec>=1)
			sp_x1 = (_real_ *)(&(((char *)sp_buffer)[MAX(2*bdim*sizeof(_real_), ExtTempSharedBufferSize)])); 
		if(NumOfnSVec>=2) sp_x2 = &sp_x1[sv_n];
		if(NumOfnSVec>=3) sp_qtf = &sp_x2[sv_n];
		if(NumOfnSVec>=4) sp_diag = &sp_qtf[sv_n];
		if(NumOfnSVec>=5) sp_wa1 = &sp_diag[sv_n]; 
		if(NumOfnSVec>=6) sp_wa2 = &sp_wa1[sv_n]; 
		if(NumOfnSVec>=7) sp_wa3 = &sp_wa2[sv_n]; 
		if(NumOfnSVec>=8) sp_x = &sp_wa3[sv_n];
		// NumOfmSVec determines how much m-length vectors can be in shared memory,
		// but it is always considered after all above n-length vectors can be in shared memory.
		if(NumOfnSVec>=8){ // Otherwise, sp_x pointer used below has not been initialized.
			if(NumOfmSVec>=1)  // wa4 to shared memory
				sp_wa4 = (_real_ *)&sp_x[sv_n]; 
			if(NumOfmSVec>=2) // fvec and wa4 to shared memory
				sp_fvec = &sp_wa4[sv_m]; 
			if(NumOfmSVec>=(2+sv_n)) //fjac, fvec and wa4 to shared memory
				sp_fjac = &sp_fvec[sv_m]; // fjac is n m-length vectors.
		}

		/* Arrays have to be in global memory 
		     Note: 
					1. All arrays must be stored in the opposite order of those above in shared 
						memory.  This way ensures the fjac pointer must have been initialized in global 
						memory before any other array pointers.
					2. All the same name variables for different fittings (i.e., different bidx LMFit) are 
						saved continuously in the global memory. i.e., for example, all fjac blocks for 
						all different bidx LMFit are adjoining in the global memory, and they are followed by
						all fvec blocks for all different bidx LMFit, and so on.
		*/
		if(NumOfmSVec<(2+sv_n)) 
			sp_fjac = GPU_LMFit_WA_Ptr; // The pointer to the working array in the global memory
		if(NumOfnSVec<8 || NumOfmSVec<2)  
			sp_fvec = &sp_fjac[gdim*sv_m*sv_n]; // in global memory, fvec is immediately after fjac;
		if(NumOfnSVec<8 || NumOfmSVec<1)
			sp_wa4 = &sp_fvec[gdim*sv_m]; // in global memory, wa4 is immediately after fvec;
		if(NumOfnSVec<8) sp_x = &sp_wa4[gdim*sv_m];
		if(NumOfnSVec<7) sp_wa3 = &sp_x[gdim*sv_n];
		if(NumOfnSVec<6) sp_wa2 = &sp_wa3[gdim*sv_n];
		if(NumOfnSVec<5) sp_wa1 = &sp_wa2[gdim*sv_n];
		if(NumOfnSVec<4) sp_diag = &sp_wa1[gdim*sv_n];
		if(NumOfnSVec<3) sp_qtf = &sp_diag[gdim*sv_n];
		if(NumOfnSVec<2) sp_x2 = &sp_qtf[gdim*sv_n];
		if(NumOfnSVec<1) sp_x1 = &sp_x2[gdim*sv_n];
		/* 
		Now the inital pointer of all fjac, fvec, ... and x1 blocks are found and saved to sp_fjac,
		sp_fvec, ... and sp_x1. The below code is to specify the pointers to fjac, fvec, ... and x1 for each
		LMFit block (one LMFit GPU block completes one fitting.). 
		*/
		if(NumOfnSVec<8 || NumOfmSVec<(2+sv_n)) 
			sp_fjac = &sp_fjac[bidx*sv_m*sv_n];
		if(NumOfnSVec<8 || NumOfmSVec<2)  
			sp_fvec = &sp_fvec[bidx*sv_m];
		if(NumOfnSVec<8 || NumOfmSVec<1)
			sp_wa4 = &sp_wa4[bidx*sv_m];
		if(NumOfnSVec<8) sp_x = &sp_x[bidx*sv_n];
		if(NumOfnSVec<7) sp_wa3 = &sp_wa3[bidx*sv_n]; 
		if(NumOfnSVec<6) sp_wa2 = &sp_wa2[bidx*sv_n]; 
		if(NumOfnSVec<5) sp_wa1 = &sp_wa1[bidx*sv_n]; 
		if(NumOfnSVec<4) sp_diag = &sp_diag[bidx*sv_n];
		if(NumOfnSVec<3) sp_qtf = &sp_qtf[bidx*sv_n];
		if(NumOfnSVec<2) sp_x2 = &sp_x2[bidx*sv_n];
		if(NumOfnSVec<1) sp_x1 = &sp_x1[bidx*sv_n]; 
	} // Array Pointers' initialization

	if(tidx==1){ // Initialize the parameters other than those initialized by tidx=0
		/*Initialize external function pointers */
		sp_GPU_FitFunctionPtr = FitFuncPtr;
		sp_GPU_JacFunctionPtr = JacFuncPtr;

		/* FitFunction private data pointer */
		sp_FitFuncPrivDataPtr = (char *)FitFuncPrivDataPtr;
		sp_JacFuncPrivDataPtr = (char *)JacFuncPrivDataPtr;

		/* Bound constraints on fitting parameters - all in global memory */
		if(UseLowerBounds && LowerBounds) {
			sp_UseLowerBounds = &UseLowerBounds[bidx*sv_n];
			sp_LowerBounds = &LowerBounds[bidx*sv_n];
		}
		else{
			sp_UseLowerBounds = NULL;
			sp_LowerBounds = NULL;
		}
		if(UseUpperBounds && UpperBounds) {
			sp_UseUpperBounds = &UseUpperBounds[bidx*sv_n];   
			sp_UpperBounds = &UpperBounds[bidx*sv_n]; 
		}
		else{
			sp_UseUpperBounds = NULL;   
			sp_UpperBounds = NULL; 
		}
		
		/* GPU_LMFit termination and configuration creiteria */
		if(Users_LM_Criteria.epsfcn>0) 
			sv_GPU_LM_Config.epsfcn = Users_LM_Criteria.epsfcn;
		else
			sv_GPU_LM_Config.epsfcn = Default_LM_Criteria.epsfcn;
		if(Users_LM_Criteria.ftol>0) 
			sv_GPU_LM_Config.ftol = Users_LM_Criteria.ftol;
		else
			sv_GPU_LM_Config.ftol = Default_LM_Criteria.ftol;
		if(Users_LM_Criteria.gtol>0) 
			sv_GPU_LM_Config.gtol = Users_LM_Criteria.gtol;
		else
			sv_GPU_LM_Config.gtol = Default_LM_Criteria.gtol;
		if(Users_LM_Criteria.maxiter>0) 
			sv_GPU_LM_Config.maxiter = Users_LM_Criteria.maxiter;
		else
			sv_GPU_LM_Config.maxiter = Default_LM_Criteria.maxiter;
		if(Users_LM_Criteria.stepfactor>0) 
			sv_GPU_LM_Config.stepfactor = Users_LM_Criteria.stepfactor;
		else
			sv_GPU_LM_Config.stepfactor = Default_LM_Criteria.stepfactor;
		if(Users_LM_Criteria.xtol>0) 
			sv_GPU_LM_Config.xtol = Users_LM_Criteria.xtol;
		else
			sv_GPU_LM_Config.xtol = Default_LM_Criteria.xtol;

		/* Initialize Levelberg-Marquardt parameter and iteration counter */
		sv_LMPar_complete = 0;

		sv_sum = zero;
		sv_temp = zero;
		sv_temp1 = zero;
		sv_temp2 = zero;
		sv_temp3 = zero;
		sv_alpha = zero;
		sv_gnorm = zero;
		sv_prered = zero;
		sv_cosx = zero;
		sv_sinx = zero;
		sv_cotan = zero;
		sv_tanx = zero;
		sv_qtbpj = zero;
		sv_actred = zero;
		sv_dirder = zero;
		sv_ratio = zero;

		sv_par = zero;
		sv_iter = 1;
		sv_nfev = 0;
		sv_fnorm = -one;
		sv_fnorm1 = -one;
		sv_xnorm = -one;
		sv_delta = zero;
		sv_InfoNum = 0;
		sv_pnorm = 0;
	}
	__syncthreads(); // Wait until all above parameters are initialized. 

	/* Initialize n-length vectors */
	if(tidx<sv_n){
		sp_x[tidx] = x[tidx];
		sp_x1[tidx] = sp_x[tidx];
		sp_x2[tidx] = sp_x[tidx];
		sp_qtf[tidx] = zero;
		sp_diag[tidx] = zero;
		sp_wa1[tidx] = zero;
		sp_wa2[tidx] = zero;
		sp_wa3[tidx] = zero;
	}
	__syncthreads(); // Wait for x being initialized.
	
	/* Check validation of bounds */
	if(sp_UseLowerBounds && sp_UseUpperBounds){ // Both bounds are applied to x. 		
		if(tidx<sv_n){
			if(sp_UseLowerBounds[tidx] && sp_UseUpperBounds[tidx] && (sp_UpperBounds[tidx]<=sp_LowerBounds[tidx]))
				sv_InfoNum = BOUNDSERR;
		}
		__syncthreads(); 
		if(sv_InfoNum) {
			if(InfoNum)
				*InfoNum = sv_InfoNum;
			return; 
		}
	}

	/* Check validation of initial x */
	if(sp_UseLowerBounds){ 
		if(tidx<sv_n){
			if(sp_UseLowerBounds[tidx] && (sp_x[tidx]<sp_LowerBounds[tidx]))
				sv_InfoNum = LBERR;
		}
		__syncthreads(); 
		if(sv_InfoNum) {
			if(InfoNum)
				*InfoNum = sv_InfoNum;
			return; 
		}
	}

	if(sp_UseUpperBounds){ 
		if(tidx<sv_n){
			if(sp_UseUpperBounds[tidx] && (sp_x[tidx]>sp_UpperBounds[tidx]))
				sv_InfoNum = UBERR;
		}
		__syncthreads(); 
		if(sv_InfoNum) {
			if(InfoNum)
				*InfoNum = sv_InfoNum;
			return; 
		}
	}

	// Fit function
	if((*sp_GPU_FitFunctionPtr)(sp_x, sp_fvec, sp_FitFuncPrivDataPtr) != -1){
		if(tidx==0) sv_nfev +=1;
		__syncthreads();
	}
	else{
		if(tidx==0) sv_InfoNum = STOPBYFITFUNC;
		goto COMPLETE;
	}

	// Norm of fvec to fnorm;
	GPU_ENorm(sp_fvec, sv_m, &sv_fnorm);
	__syncthreads();

	/* Start outer while-loop */
	while(sv_InfoNum==0){

		if(tidx<sv_n)
			sp_x2[tidx] = sp_x1[tidx];
		__syncthreads();

		 /* Calculate the jacobian matrix */
		if(sp_GPU_JacFunctionPtr){
			if((*sp_GPU_JacFunctionPtr)(sp_x2, sp_fjac, sp_JacFuncPrivDataPtr)==-1)
				GPU_FDJac2(sp_x2,  sp_wa4,  sp_fjac);
		}
		else
			GPU_FDJac2(sp_x2,  sp_wa4,  sp_fjac);
		__syncthreads();

		/* Determine if any of the parameters are pegged at the limits */
		if (sp_UseLowerBounds || sp_UseUpperBounds) {	
			for (i1=0; i1<sv_n; i1++) {
				if(tidx==0){
					if(sp_UseLowerBounds) sv_lpegged = (sp_UseLowerBounds[i1] && (sp_x1[i1] == sp_LowerBounds[i1]));
					else sv_lpegged = 0;
					if(sp_UseUpperBounds) sv_upegged = (sp_UseUpperBounds[i1] && (sp_x1[i1] == sp_UpperBounds[i1]));
					else sv_upegged = 0;
				}
				__syncthreads();
				if (sv_lpegged || sv_upegged) {
					sp_buffer[tidx] = zero;
					sp_buffer[tidx+bdim] = zero;
					for(i0=0; i0<sv_m; i0+=(2*bdim)){
						if((i0+tidx)<sv_m) sp_buffer[tidx] += sp_fvec[i0+tidx]*sp_fjac[i1*sv_m+i0+tidx];
						if((i0+tidx+bdim)<sv_m) 
							sp_buffer[bdim+tidx] += sp_fvec[i0+bdim+tidx]*sp_fjac[i1*sv_m+i0+bdim+tidx];
					}
					__syncthreads();
					for (i0=bdim; i0>=1; i0>>=1){ //Warp Reduction 
						if (tidx<i0)
							sp_buffer[tidx] += sp_buffer[tidx + i0];
						if(i0>=warpSize)
							__syncthreads(); 
					} 
					// sp_buffer[0] is the sum now. 
				}
				__syncthreads();
				if (sv_lpegged && (sp_buffer[0] > 0)) {
					for(i0=0; i0<sv_m; i0+=bdim)
						if((i0+tidx)<sv_m) sp_fjac[i1*sv_m+i0+tidx] = 0;
				}
				if (sv_upegged && (sp_buffer[0] < 0)) {
					for(i0=0; i0<sv_m; i0+=bdim)
						if((i0+tidx)<sv_m) sp_fjac[i1*sv_m+i0+tidx] = 0;
				}
			}		
			__syncthreads();
		} 
		
		/* Compute the QR factorization of the jacobian */
		GPU_QRFac(sp_fjac);
		__syncthreads();

		/*
		*	 on the first iteration and if mode is 1,  scale according
		*	 to the norms of the columns of the initial jacobian.
		*/
		if(tidx<sv_n && sv_iter == 1){
			if (sp_wa2[tidx] == zero) 
				sp_diag[tidx] = one;
			else
				sp_diag[tidx] = sp_wa2[tidx];
			sp_wa3[tidx] = sp_diag[tidx]*sp_x1[tidx];
		}
		__syncthreads(); 

		/*
		*	 on the first iteration,  calculate the norm of the scaled x
		*	 and initialize the step bound delta.
		*/
		if (tidx==0 && sv_iter==1){
			// start for xnorm[fn] = LM_enorm(n,  &wa3[fn*n]);	
			sv_xnorm = zero;
			for(i0=0; i0<sv_n; i0++) sv_xnorm += (sp_wa3[i0]*sp_wa3[i0]);
			sv_xnorm = cuSQRT(sv_xnorm);
			// end for xnorm[fn] = LM_enorm(n,  &wa3[fn*n]);	
			sv_delta = sv_GPU_LM_Config.stepfactor*sv_xnorm;
			if (sv_delta == zero) sv_delta = sv_GPU_LM_Config.stepfactor;
		}

		/*
		*	 form (Q transpose)*fvec and store the first n components in qtf.
		*/
		for (i0=0; i0<sv_m; i0+=bdim)
			if((i0+tidx)<sv_m)
				sp_wa4[i0+tidx] = sp_fvec[i0+tidx];

		if(tidx==0) i3=0;
		__syncthreads(); 

		while(i3<sv_n){

			if(tidx==0){
				i4 = i3*(sv_m+1);
				sv_temp3 = sp_fjac[i4];
			}
			__syncthreads(); 

			if (sv_temp3 != zero) {
				sp_buffer[tidx] = zero;
				sp_buffer[tidx+bdim] = zero;
				__syncthreads(); // Wait for all threads to complete the above SVec initialization. 

				for(i0=0; i0<(sv_m-i3); i0+=(2*bdim)) {
					if((i0+tidx) < (sv_m-i3))
						sp_buffer[tidx] += sp_fjac[i0+i4+tidx]*sp_wa4[i0+i3+tidx];
					if((i0+tidx+bdim) < (sv_m-i3))
						sp_buffer[bdim+tidx] += sp_fjac[i0+i4+bdim+tidx]*sp_wa4[i0+i3+bdim+tidx];
				}
				__syncthreads(); // Wait untill all sp_buffer are initialized. 
				for (i0=bdim; i0>=1; i0>>=1){ 	//Warp Reduction 
					if (tidx < i0)
						sp_buffer[tidx] += sp_buffer[tidx + i0];
					if (i0>=warpSize) 
						__syncthreads(); 
				} // sp_buffer[0] are the real value of sum in CPU 
				sv_sum = sp_buffer[0];
				if(tidx==0) sv_temp = -sv_sum/sv_temp3; //temp[fn] = -sum[fn] / temp3[fn] in CPU code

				__syncthreads(); // Wait until sv_temp is updated. 
				for(i0=0;  i0<(sv_m-i3); i0+=(2*bdim)) {
					if((i0+tidx) < (sv_m-i3))
						sp_wa4[ i3+i0+tidx] += (sp_fjac[ i4+i0+tidx]*sv_temp);
					if((i0+tidx+bdim) < (sv_m-i3))
						sp_wa4[i3+i0+bdim+tidx] += (sp_fjac[i4+i0+bdim+tidx]*sv_temp);
				}
			}

			__syncthreads(); // Wait until wa4 is updated.
			if(tidx==0){
				sp_fjac[i4] = sp_wa1[i3];
				sp_qtf[i3] = sp_wa4[i3];
			}
			__syncthreads(); 

			if(tidx==0) i3+=1;
			__syncthreads(); 
		}

		// From this point on,  only the square matrix,  consisting of the  triangle of R,  is needed.

#if defined(__DEBUG_CHKOVERFLOW__)
		/*
			Check for overflow.  This should be a cheap test here since FJAC has been reduced to a 
			(small) square matrix, and the test is O(N^2). 
		*/
		for (i0=0; i0<sv_n*sv_m; i0+=bdim) {
			if((i0+tidx)<(sv_n*sv_m)){
				if (!isfinite(sp_fjac[i0+tidx])) sv_InfoNum = INFINITEJAC;
			}
		}
		__syncthreads(); 
		if(sv_InfoNum == INFINITEJAC) goto COMPLETE;
#endif //__DEBUG_CHKOVERFLOW__

		/*
		*	 compute the norm of the scaled gradient.
		*/
		if(tidx==0 && sv_fnorm != zero) {
			sv_gnorm = zero; 
			for (i3=0; i3<sv_n; i3++ ) {
				i4 = i3*sv_m;
				if (sp_wa2[i3] != zero) {
					sv_sum = zero;
					for(i0=0; i0<=i3; i0++)
						sv_sum += sp_fjac[i3*sv_m + i0]*(sp_qtf[i0]/sv_fnorm);
					sv_gnorm = MAX(sv_gnorm, cuFABS(sv_sum/sp_wa2[i3])); 
				}
			}
		}
		__syncthreads();
		
		/*
		*	 test for convergence of the gradient norm.
		*/
		// if gnorm is changed above, all threads must meet a __syncthreads(), so __syncthreads() is NOT needed here. 
		if (sv_gnorm <= sv_GPU_LM_Config.gtol) {
			if(tidx==0) sv_InfoNum = LM_OK_DIR;
			goto COMPLETE;
		}
		
		if(tidx<sv_n)
			sp_diag[tidx] = MAX(sp_diag[tidx], sp_wa2[tidx]);
		if(tidx==0) sv_ratio = zero;
		__syncthreads();

		/* Start inner while-loop */
		while(sv_ratio < p0001 && sv_InfoNum == 0) {

			if(tidx==0) {
				GPU_LMPar(sp_fjac, sp_diag, sp_qtf, sp_wa1, sp_wa2, sp_wa3, sp_wa4);
				sv_alpha = one;
			}
			__syncthreads();

			if(tidx<sv_n) sp_wa1[tidx] = -sp_wa1[tidx];
			__syncthreads();

			if ((!sp_UseLowerBounds) && (!sp_UseUpperBounds)) { 	
				if(tidx<sv_n) sp_wa2[tidx] = sp_x1[tidx] + sp_wa1[tidx]; 
			} 
			else {
				if(tidx==0){
					for (i1=0; i1<sv_n; i1++) {
						if(sp_UseLowerBounds) sv_lpegged = (sp_UseLowerBounds[i1] && (sp_x1[i1] <= sp_LowerBounds[i1]));
						else sv_lpegged = 0;
						if(sp_UseUpperBounds) sv_upegged = (sp_UseUpperBounds[i1] && (sp_x1[i1] >= sp_UpperBounds[i1]));
						else sv_upegged = 0;

						if (sv_lpegged && (sp_wa1[i1] < 0)) sp_wa1[i1] = 0;
						if (sv_upegged && (sp_wa1[i1] > 0)) sp_wa1[i1] = 0;

						// Upgrade on Apr 28, 2014 with "sp_UseLowerBounds &&" and "sp_UseUpperBounds &&"
						if (sp_UseLowerBounds && (cuFABS(sp_wa1[i1]) > MACHINE_EPSILON) && sp_UseLowerBounds[i1] && 
							((sp_x1[i1] + sp_wa1[i1]) < sp_LowerBounds[i1]))
							sv_alpha = MIN(sv_alpha, (sp_LowerBounds[i1]-sp_x1[i1])/sp_wa1[i1]);
						if (sp_UseUpperBounds && (fabs(sp_wa1[i1]) > MACHINE_EPSILON) && sp_UseUpperBounds[i1] && 
							((sp_x1[i1] + sp_wa1[i1]) > sp_UpperBounds[i1])) 
							sv_alpha = MIN(sv_alpha, (sp_UpperBounds[i1]-sp_x1[i1])/sp_wa1[i1]);
					}
				}
				__syncthreads();
				/* Scale the resulting vector, advance to the next position */
				if (tidx<sv_n) {
					sp_wa1[tidx] = sp_wa1[tidx] * sv_alpha;
					sp_wa2[tidx] = sp_x1[tidx] + sp_wa1[tidx];
					/* 
					* Adjust the output values.  If the step put us exactly
					* on a boundary, make sure it is exact.
					*/
					// Upgrade on Apr 28, 2014 with "sp_UseLowerBounds &&" and "sp_UseUpperBounds &&"
					if (sp_UseUpperBounds && sp_UseUpperBounds[tidx] && 
						(sp_wa2[tidx] >= (sp_UpperBounds[tidx]*(1-((sp_UpperBounds[tidx] >= 0) ? (+1) : (-1))*MACHINE_EPSILON) - 
						((sp_UpperBounds[tidx] == 0)?(MACHINE_EPSILON):0)))) sp_wa2[tidx] = sp_UpperBounds[tidx];
					if (sp_UseLowerBounds && sp_UseLowerBounds[tidx] && 
						(sp_wa2[tidx] <= (sp_LowerBounds[tidx]*(1+((sp_LowerBounds[tidx] >= 0) ? (+1) : (-1))*MACHINE_EPSILON) + 
						((sp_LowerBounds[tidx] == 0)?(MACHINE_EPSILON):0)))) sp_wa2[tidx] = sp_LowerBounds[tidx];
				}
				__syncthreads();
			}
			
			if(tidx<sv_n) sp_wa3[tidx] = sp_diag[tidx]*sp_wa1[tidx];
			__syncthreads();

			if(tidx==0){
				//pnorm[fn] = LM_enorm(sv_n, &wa3[fn*sv_n]);
				sv_pnorm = zero;
				for(i0=0; i0<sv_n; i0++) sv_pnorm += sp_wa3[i0]*sp_wa3[i0];
				sv_pnorm = cuSQRT(sv_pnorm);
				if (sv_iter == 1) 
					sv_delta = MIN(sv_delta, sv_pnorm); 
			}

			/* evaluate the function at x + p and calculate its norm. */
			if(tidx<sv_n)
				sp_x2[tidx] = sp_wa2[tidx];
			__syncthreads();

			/* call fit function */
			if((*sp_GPU_FitFunctionPtr)(sp_x2, sp_wa4, sp_FitFuncPrivDataPtr)!= -1){
				if(tidx==0) sv_nfev +=1;
				__syncthreads();
			}
			else{
				if(tidx==0) sv_InfoNum = STOPBYFITFUNC;
				goto COMPLETE;
			}

			//fnorm1[fn] = LM_enorm(sv_m, &wa4[fn*sv_m]);
			GPU_ENorm(sp_wa4, sv_m, &sv_fnorm1);
			__syncthreads();

			if(tidx==0){  // start tidx==0 case

				sv_actred = -one;
				if ((p1*sv_fnorm1) < sv_fnorm) {
					sv_temp = sv_fnorm1/sv_fnorm;
					sv_actred = one - sv_temp*sv_temp;
				}

				for (i1=0; i1<sv_n; i1++ ) {
					/* compute the scaled predicted reduction and the scaled directional derivative. */
					sp_wa3[i1] = zero;
					sv_temp = sp_wa1[i1];
					for (i0=0; i0<=i1; i0++ ) {
						sp_wa3[i0] += sp_fjac[i1*sv_m+i0]*sv_temp;
					}
				}
				sv_temp1 = zero;
				for(i0=0; i0<sv_n; i0++) sv_temp1 += sp_wa3[i0]*sp_wa3[i0];
				sv_temp1 = cuSQRT(sv_temp1);
				sv_temp1 = sv_temp1*sv_alpha/sv_fnorm;
				sv_temp2 = (cuSQRT(sv_par)*sv_pnorm)/sv_fnorm;
				sv_prered = sv_temp1*sv_temp1 + (sv_temp2*sv_temp2)/p5;
				sv_dirder = -(sv_temp1*sv_temp1 + sv_temp2*sv_temp2);

				/* compute the ratio of the actual to the predicted reduction. */
				sv_ratio = zero;
				if (sv_prered != zero) 
					sv_ratio = sv_actred/sv_prered;

				/* update the step bound. */
				if (sv_ratio <= p25) {
					if (sv_actred >= zero) 
						sv_temp = p5; 
					else 
						sv_temp = p5*sv_dirder/(sv_dirder + p5*sv_actred);

					if (((p1*sv_fnorm1) >= sv_fnorm) || (sv_temp < p1) ) 
						sv_temp = p1;

					sv_delta = sv_temp*MIN(sv_delta, sv_pnorm/p1);
					sv_par = sv_par/sv_temp;
				} 
				else {
					if ((sv_par == zero) || (sv_ratio >= p75) ) {
						sv_delta = sv_pnorm/p5;
						sv_par = p5*sv_par;
					}
				}

			} // end of tidx==0 case
			__syncthreads();

			/* test for successful iteration. */
			if (sv_ratio >= p0001) {
				/* successful iteration. update x,  fvec,  and their norms. */
				if(tidx<sv_n){
					sp_x1[tidx] = sp_wa2[tidx];
					sp_wa2[tidx] = sp_diag[tidx]*sp_x1[tidx];
				}
				__syncthreads();

				for(i0=0; i0<sv_m; i0+=bdim)
					if((i0+tidx)<sv_m) sp_fvec[i0+tidx] = sp_wa4[i0+tidx];

				//xnorm[fn] = LM_enorm(sv_n, &wa2[fn*sv_n]);
				if(tidx==0){
					sv_xnorm = zero;
					for(i0=0; i0<sv_n; i0++) sv_xnorm += sp_wa2[i0]*sp_wa2[i0];
					sv_xnorm = cuSQRT(sv_xnorm);
					sv_fnorm = sv_fnorm1;
					sv_iter += 1;
				}
			}
			__syncthreads();

			if(tidx==0){
				/*
				*	    tests for convergence.
				*/
				if ((cuFABS(sv_actred) <= sv_GPU_LM_Config.ftol) &&
					(sv_prered <= sv_GPU_LM_Config.ftol) && 
					(p5*sv_ratio <= one) ) sv_InfoNum = LM_OK_CHI;
				if ((sv_InfoNum==LM_OK_CHI) && (sv_delta<=sv_GPU_LM_Config.xtol*sv_xnorm)) 
					sv_InfoNum = LM_OK_BOTH;
				else{
					if (sv_delta<=sv_GPU_LM_Config.xtol*sv_xnorm) sv_InfoNum = LM_OK_PAR;
				}					
			}
			__syncthreads();

			if(sv_InfoNum!=0) goto COMPLETE;

			if(tidx==0){
				/*
				*	    tests for termination and stringent tolerances.
				*/
				if (sv_InfoNum == 0){
					if (sv_iter >= sv_GPU_LM_Config.maxiter) 
						/* Too many iterations */
						sv_InfoNum = LM_MAXITER;
					if ((cuFABS(sv_actred) <= MACHINE_EPSILON) && (sv_prered <= MACHINE_EPSILON) 
						&& (p5*sv_ratio <= one) ) 
						sv_InfoNum = LM_FTOL;
					if (sv_delta <= MACHINE_EPSILON*sv_xnorm) 
						sv_InfoNum = LM_XTOL;
					if (sv_gnorm <= MACHINE_EPSILON) 
						sv_InfoNum = LM_GTOL;	
				}
			}

			__syncthreads();
		} // inner while-loop

		__syncthreads();
	} // outer while-loop

COMPLETE:
	__syncthreads(); // Maybe not necessary
	if(Chisq){
		(*sp_GPU_FitFunctionPtr)(sp_x1, sp_fvec, sp_FitFuncPrivDataPtr);
		__syncthreads();
		GPU_ENorm(sp_fvec, sv_m, &sv_fnorm);
		__syncthreads();
		*Chisq = sv_fnorm;
	}
	if(tidx<sv_n)
		x[tidx] = sp_x1[tidx];
	if(tidx==0) {
		if(InfoNum)
			*InfoNum = sv_InfoNum;
	}

}


#endif //_GPU_LMFIT_CU_
