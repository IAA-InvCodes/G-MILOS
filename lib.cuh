#include "definesCuda.cuh"
#include "defines.h"

/**
 * 
 */

 __device__ void covarm(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM);

__device__ void covarmf(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM);

/**
 * 
 */
__device__ int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result);





/**
 * 
 */

__device__ REAL fchisqr(const REAL * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree);


/**
 * 
 */
__device__ void multmatrixIDLValue(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value);
/**
 * 
 */
__device__ void totalParcialMatrixf(const REAL * __restrict__ A, int f,int c,int p,REAL * result);
/**
 * 
 */
__device__ void totalParcialf(const REAL * __restrict__ A, int f,int c,PRECISION * result);



/**
 * 
 */
__device__ int multmatrix_transpose(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value);



__global__ void d_multmatrix_transpose(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,REAL value);

