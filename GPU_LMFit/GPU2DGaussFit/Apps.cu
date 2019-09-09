/* Main program */

#include "Apps.cuh"
#include "mex.h"
#include "GPU_2DGauss_Solver.cuh" // To use GPU_LMFIT_Solver

/* Macro for Matlab floating data precision */
#ifdef _SINGLE_FLOAT_ 
	#define mxFLOAT mxSINGLE_CLASS
#else
	#define mxFLOAT mxDOUBLE_CLASS
#endif

/* The Matlab mex interface function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input and basic parameters */
	int n = 5;					// Number of fitting parameters
	int m;						// Number of data points 
	int NumOfImgs;			// Number of images
	int ImgDim;				// Image dimension size (assume square images)
	int FitMethod;			// 0 - maximum likelihood estimator, or 1 - unweighted least squares. 
	int JacMethod;			// 0 - Analytical Jacobian; 1 - Numerical Jacobian
	_real_ init_s = one;			// Initial value of Gaussian waist width s, default = 1.0;

	// Images data buffer
	_real_ *ImgsBuffer;			// The pointer to images data buffer
	unsigned int ImgDataSize;	// Images data buffer size
	
	// GPU CUDA
	int GPU_device_ID;		// GPU device ID
	int GPU_Block_Size;		// User-defined maximum blockDim.x (blockDim.y = blockDim.z = 1)
	int GPU_Grid_Size;		// User-defined maximum gridDim.x (gridDim.y = gridDim.z = 1)

	/* Output */
	_real_ *outx;				// Fitted parameters
	char UsageMsg[TempStringBufferSize] = {};		// A string for the information to display the function usage.
	char ErrorMsg[TempStringBufferSize+1] = {};	// A string for the information of errors from the function 
	_real_ *Info = NULL;		// To return either Chi squares or infomation code from GPU-LMFit
	
	/* Prepare the usage string */
	sprintf(UsageMsg,  
		"\n\t%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n"
		"%s\n%s\n%s\n%s\n%s\n%s\n\n",
		"Useage: x = GPU2DGaussFit(ImgsData, ImgDimension, Init_s, FitMethod, JacMethod, ...",
		"                     GPU_device_ID, GPU_Block_Size, GPU_Grid_Size).",
		"        or [x infonum] = GPU2DGaussFit(ImgsData, ImgDimension, Init_s, FitMethod, ...", 
		"                     JacMethod, GPU_device_ID, GPU_Block_Size, GPU_Grid_Size).",
		"        ImgsData is 1D square image data array.",
		"              If ImgsData is originally 2D, it need to be converted to 1D.",
		"        ImgDimension is the dimension size of the images. It must be a scalar.",
		"        Init_s is the initial value of the Gaussian waist width. (default = 1.0 pixel)",
		"        FitMethod is to select fitting methods (estimators): ",
		"              0 - maximum likelihood estimator (MLE);",
		"              1 - unweighted least squares (uWLS);",
		"        JacMethod = 0 is to use numerical Jacobian matrix, or otherwise analytical", 
		"              Jacobian matrix. (Note: no analytical Jacobian function is available for MLE).",
		"        GPU_device_ID is the device number of GPU to be used", 
		"             (it is zero if only one GPU is available);",
		"        GPU_Block_Size and GPU_Grid_Size are for user to preset the maximum block ",
		"              size and the maximum grid size of CUDA, respectively;",
		"        x is fitted parameters, infonum is the status of each fit.");

	/* Check for proper number of arguments */
	if(nrhs!=8) {
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Eight inputs required!");
	}
	if(nlhs!=1 && nlhs!=2) {
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Incorrect output parameters!");
	}

	/* Check the formats for the input arguments */
#ifdef _SINGLE_FLOAT_
	if(!mxIsSingle(prhs[0])) 
	{
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("ImgsData must be a single precision floating data type vector");
	}
#else
	if(mxIsSingle(prhs[0])) 
	{
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("ImgsData must be a double precision floating data type vector");
	}
#endif
	if(  mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]) ||
		mxIsComplex(prhs[2]) || mxIsComplex(prhs[3]) || mxIsComplex(prhs[4]) ||
		mxIsComplex(prhs[5]) || mxIsComplex(prhs[6]) || mxIsComplex(prhs[7]) ||
		mxGetNumberOfElements(prhs[1])!=1 || mxGetNumberOfElements(prhs[3])!=1 ||
		mxGetNumberOfElements(prhs[4])!=1 || mxGetNumberOfElements(prhs[5])!=1 ||
		mxGetNumberOfElements(prhs[6])!=1 || mxGetNumberOfElements(prhs[7])!=1 ) 
	{
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Input parameters are wrong!");
	}

	/* Get the intial value of fitting parameter s */
	if(mxGetNumberOfElements(prhs[2])==1) {
		if ((mxGetPr(prhs[2])[0]!=0))
			init_s = (_real_)(mxGetPr(prhs[2])[0]);
		else
			init_s = one;
	}
	else {
		mexPrintf("%s", UsageMsg);
		mexErrMsgTxt("Inital value of Gaussian waist width is wrong!");
	}
	
	/* Check the format of input image data */
	if(mxGetM(prhs[0])!=1)
		mexErrMsgTxt("ImgsData must be a row vector.");

	/* Get the image dimension size */
	ImgDim = (int)mxGetScalar(prhs[1]);

	/* m - number of pixels */
	m = ImgDim*ImgDim;
	
	/* Get the size of the input image data */
	ImgDataSize = (unsigned int)mxGetN(prhs[0]);

	/* Calculate the number of images*/
	NumOfImgs = (int)(ImgDataSize/ImgDim/ImgDim);

	/* Check the input image data validation */
	if ((NumOfImgs-(_real_)ImgDataSize/ImgDim/ImgDim) == 0.0) {
		
		/* Get and check the fitting method */
		FitMethod = (int)mxGetScalar(prhs[3]);
		if (FitMethod<0 || FitMethod>1) {
			mexPrintf("%s", UsageMsg);
			mexErrMsgTxt("Wrong Fit methods.");
		}

		/* Get Jacobian method */
		JacMethod = (int)mxGetScalar(prhs[4]);
		if(JacMethod!=0) JacMethod = 1;

		/* Create a pointer to the image data */
		ImgsBuffer = (_real_ *)mxGetPr(prhs[0]);

		/* Get the user-defined maxiumum CUDA block and grid size*/
		GPU_Block_Size = (int)mxGetScalar(prhs[6]);
		GPU_Grid_Size = (int)mxGetScalar(prhs[7]);

		/* Prepare the output matrices */
		// Fitted x
		plhs[0] = mxCreateNumericMatrix(1, n*NumOfImgs, mxFLOAT, mxREAL);
		outx = (_real_ *) mxGetData(plhs[0]);
		if(outx == NULL)
			mexErrMsgTxt("Fail to allocate memory for output x!");
		// Info
		if(nlhs == 2) {
			plhs[1] = mxCreateNumericMatrix(1, NumOfImgs,  mxFLOAT, mxREAL);
			Info = (_real_ *) mxGetData(plhs[1]);
			if(Info == NULL)
				mexErrMsgTxt("Fail to allocate memory for output Info!");
		}
		
		/* Get the user specified GPU device ID */
		GPU_device_ID = (int)mxGetScalar(prhs[5]);

		if(GPU_device_ID == -1){
			/* 
			One may add MPFit routine here for CPU version fitting: MPFit is originally developed 
			based on Minpack by Craig Markwardt. For more details, please refer to 
			http://cow.physics.wisc.edu/~craigm/.
			*/
			mexErrMsgTxt("GPU computation only! No CPU version fitting routine!");
		}
		else{
			if(GPU_LMFIT_Solver(n, m, NumOfImgs, ImgDim, JacMethod, FitMethod, init_s, 
				GPU_device_ID, ImgsBuffer, &GPU_Block_Size, &GPU_Grid_Size, outx, Info, ErrorMsg))
				mexErrMsgTxt(ErrorMsg);
		}
	}
	else
		mexErrMsgTxt("Image Data Size, Dimension and Number of Images do NOT match!");
}
