#ifndef _GPU_2DGAUSS_SOLVER_H_
#define _GPU_2DGAUSS_SOLVER_H_

// GPU_LMFit library
#include "GPU_LMFit.cuh"

/* Kernel constant inputs struct definition */
struct TWODGAUSS_GPU_LMFIT_INS_STRUCT {
	int n;								// Number of fitting parameters.
	int m;								// Number of data points.
	int ImgDim;						// Image dimension size (assume square images).
	int JacMethod;					// 0 - Analytical Jacobian; 1 - Numerical Jacobian.	
	int FitMethod;					// 0 - maximum likelihood estimator, or 1 - unweighted least squares. 
};
typedef struct TWODGAUSS_GPU_LMFIT_INS_STRUCT TWODGAUSS_GPU_LMFIT_INS;


/* External function prototypes */
extern int GPU_LMFIT_Solver(int, int, int, int, int, int, _real_, int, _real_ *,  int *, int *, _real_ *, _real_ *, char *);


#endif //_GPU_2DGAUSS_SOLVER_H_
