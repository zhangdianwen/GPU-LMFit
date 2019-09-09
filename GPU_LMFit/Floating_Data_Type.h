#ifndef _FLOATING_DATA_TYPE_H_
#define _FLOATING_DATA_TYPE_H_


//#define _SINGLE_FLOAT_ // define _SINGLE_FLOAT_ to use single precision floating data type
#ifdef _SINGLE_FLOAT_ 
	#define _real_ float

	#define cuFABS fabsf
	#define cuSQRT sqrtf
	#define cuRSQRT rsqrtf
	#define cuEXP expf
	#define cuPOW powf
	#define cuCEIL ceilf
	#define cuLOG logf
	#define cuCOPYSIGN copysignf

	#define FABS fabsf
	#define SQRT sqrtf
	#define EXP expf
	#define POW powf
	#define LOG logf
	#define COS cosf
	#define SIN sinf

	#define LOG2(x) (logf(x+0.0f)/logf(2.0f))
	#define nearest_pow2(x) (cuPOW(2.0f, floorf(LOG2(x+0.0f))))

	/* Constants */
	#define MACHINE_FLOAT_MIN   1.175494351e-38f
	#define MACHINE_FLOAT_MAX   3.402823466e+38f
	#define MACHINE_EPSILON 1.192092896e-07f
	#define zero 0.0f
	#define one 1.0f
	#define two 2.0f
	#define three 3.0f
	#define six 6.0f
	#define p1 0.1f
	#define p001 0.001f
	#define p0001 1.0e-4f
	#define p5 0.5f
	#define p05 0.05f
	#define p25 0.25f
	#define p75 0.75f
	#define E10 1e10f
#else //_SINGLE_FLOAT_ 
	#define _real_ double

	#define cuFABS fabs
	#define cuSQRT sqrt
	#define cuRSQRT rsqrt
	#define cuEXP exp
	#define cuPOW pow
	#define cuCEIL ceil
	#define cuLOG log
	#define cuCOPYSIGN copysign

	#define FABS fabs
	#define SQRT sqrt
	#define EXP exp
	#define POW pow
	#define LOG log
	#define COS cos
	#define SIN sin

	#define LOG2(x) log(x+0.0)/log(2.0)
	#define nearest_pow2(x) (cuPOW(2.0, floor(LOG2(x+0.0))))

	/* Constants */
	#define MACHINE_FLOAT_MIN 2.2250738585072014e-308
	#define MACHINE_FLOAT_MAX 1.7976931348623158e+308
	#define MACHINE_EPSILON 2.2204460492503131e-016 // Old value 1.19209e-16 was too small for certain machine or GPU device. 
	#define zero 0.0
	#define one 1.0
	#define two 2.0
	#define three 3.0
	#define six 6.0
	#define p1 0.1
	#define p001 0.001
	#define p0001 1.0e-4
	#define p5 0.5
	#define p05 0.05
	#define p25 0.25
	#define p75 0.75
	#define E10 1e10
#endif //_SINGLE_FLOAT_ 

/* Utility macros */
#ifndef _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_ // To avoid the definition confliction in cutil_inline.h
	#define MIN(a, b) ((a)<(b)?(a):(b))
	#define MAX(a, b) ((a)>(b)?(a):(b))
#endif
#define SIGN(a) (((a)>zero)?one:-one) // for CPU functions only. It gives Inf with CUDA 6.5 for Tesla K20. 


#else
#endif
