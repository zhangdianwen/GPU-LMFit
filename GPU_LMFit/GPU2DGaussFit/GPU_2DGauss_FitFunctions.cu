/* 
	Two GPU device functions:
	1. Fitting function using either MLE or uWLS;
	2. Analytic Jacobian function for uWLS 
*/

#include "GPU_2DGauss_FitFunctions.cuh"

/* GPU 2D Gaussian Fit function */
__device__ int GPU_FitFunction(const _real_ *x, _real_ *fvec, void *FitFuncPrivDataPtr)
{
#define B x[0]
#define A x[1]
#define x0 x[2]
#define y0 x[3]
#define s x[4]
#define TwoD_Gaussian GPU_f_Cs->sp_buffer

	// Shared variable 
	__shared__ GPU_FUNC_CONSTS *GPU_f_Cs;

	// Iteration counters
	int i0, j0;

	// Get basic parameters from the third argument of this function
	if(tidx==0) GPU_f_Cs = (GPU_FUNC_CONSTS *)FitFuncPrivDataPtr;
	__syncthreads();

	switch(GPU_f_Cs->sv_FitMethod)
	{
	case 0: // Ted A Laurence & Brett A Chromy's MLE
		for(j0=0; j0<GPU_f_Cs->sv_ImgDim; j0++){
			for(i0=0; i0<GPU_f_Cs->sv_ImgDim; i0+=bdim){
				if((i0+tidx)<GPU_f_Cs->sv_ImgDim){
					TwoD_Gaussian[tidx] = B + cuFABS(A)*(cuEXP(-p5*(((i0+tidx)-x0)*((i0+tidx)-x0)+
						(j0-y0)*(j0-y0))/s/s));
					if(GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)]!=0 && TwoD_Gaussian[tidx]>0){
						fvec[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)] = cuSQRT(cuFABS(TwoD_Gaussian[tidx] -  
							GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)] -
							GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)]*logf(TwoD_Gaussian[tidx]/
							GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)])));
					}
					else{
						fvec[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)] = cuSQRT(cuFABS(TwoD_Gaussian[tidx]-
							GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)]));
					}
				}
			}
		}	
		break;
	case 1: // uWLS
		for(j0=0; j0<GPU_f_Cs->sv_ImgDim; j0++){
				for(i0=0; i0<GPU_f_Cs->sv_ImgDim; i0+=bdim){
					if((i0+tidx)<GPU_f_Cs->sv_ImgDim){
						TwoD_Gaussian[tidx] = B + cuFABS(A)*(cuEXP(-p5*(((i0+tidx)-x0)*((i0+tidx)-x0)+
							(j0-y0)*(j0-y0))/s/s));
						fvec[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)] = TwoD_Gaussian[tidx] - 
							GPU_f_Cs->sp_CurrData[j0*GPU_f_Cs->sv_ImgDim+(i0+tidx)];
					}
				}
			}	
		break;
	default:
		return (0);
	}
	return (0);

#undef B
#undef A
#undef x0
#undef y0
#undef s
#undef TwoD_Gaussian
}


/* Analytical Jacobian */
__device__ int GPU_AnalyticalJacobian(const _real_ *x, _real_ *FJac, void *JacFuncPrivDataPtr)
{
#define A x[1]
#define x0 x[2]
#define y0 x[3]
#define s x[4]
#define CommonItem GPU_f_Cs->sp_buffer
#define CommonItem1 ((i0+tidx)-x0)
#define CommonItem2 (j0-y0)

	// Shared variable
	__shared__ GPU_FUNC_CONSTS *GPU_f_Cs;

	// Iteration counters
	int i0, j0;

	// Get basic parameters from the third argument of this function
	if(tidx==0) GPU_f_Cs = (GPU_FUNC_CONSTS *)JacFuncPrivDataPtr;
	__syncthreads();

	if(!GPU_f_Cs->sv_JacMethod) return(-1); // Return from here for numerical Jacobian.
		
	switch(GPU_f_Cs->sv_FitMethod)
	{
	case 0: // Ted A Laurence & Brett A Chromy's MLE
		return (-1);
	case 1: // uWLS
		for(j0=0; j0<GPU_f_Cs->sv_ImgDim; j0++){
			for(i0=0; i0<GPU_f_Cs->sv_ImgDim; i0+=bdim){
				if((i0+tidx)<GPU_f_Cs->sv_ImgDim){	
					CommonItem[tidx] = cuEXP(-p5/(s*s)*CommonItem1*CommonItem1
						-p5/(s*s)*CommonItem2*CommonItem2);
					FJac[0*GPU_f_Cs->sv_m + (j0*GPU_f_Cs->sv_ImgDim + (i0+tidx))] = one;
					FJac[1*GPU_f_Cs->sv_m + (j0*GPU_f_Cs->sv_ImgDim + (i0+tidx))] = CommonItem[tidx];
					FJac[2*GPU_f_Cs->sv_m + (j0*GPU_f_Cs->sv_ImgDim + (i0+tidx))] = 
						A*CommonItem[tidx]*CommonItem1/(s*s);
					FJac[3*GPU_f_Cs->sv_m + (j0*GPU_f_Cs->sv_ImgDim + (i0+tidx))] = 
						A*CommonItem[tidx]*CommonItem2/(s*s);
					FJac[4*GPU_f_Cs->sv_m + (j0*GPU_f_Cs->sv_ImgDim + (i0+tidx))] = 
						A*CommonItem[tidx]*(CommonItem1*CommonItem1+
						CommonItem2*CommonItem2)/(s*s*s);
				}
			}
		}
		break;
	default:
		return (-1);
	}

return(0);

#undef A
#undef x0
#undef y0
#undef s
#undef CommonItem
#undef CommonItem1
#undef CommonItem2
}