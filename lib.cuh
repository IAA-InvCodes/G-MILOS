#include "definesCuda.cuh"
#include "defines.h"

/**
 * 
 */

__device__ int covarm(const REAL * __restrict__ w,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM);
__device__ void covarm2(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM);
__device__ void covarmf(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM);
__device__ void covarmf3(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM);
/**
 * 
 */
__device__ int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result);

__device__ void multmatrixf(float *a,int naf,int nac, float *b,int nbf,int nbc,float *result);


__device__ int multmatrixShare(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);
/**
*/
__device__ int multmatrixCUBLAS(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);
/**
 * 
 */

__device__ REAL fchisqr(const REAL * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree);
//__device__ REAL fchisqr2(const float4 * __restrict__ spectra,const int  nspectro,const float4 * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree);
__device__ REAL fchisqr2(const float * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree);
__device__ REAL fchisqr3(const float4 * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree);
__global__ void d_fchisqr(const REAL * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL sig, const REAL nfree, REAL * TOT);

/**
 * 
 */
__device__ void multmatrixIDLValue(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value);
__device__ int multmatrixIDLValueSigma(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,int *fil,int *col, const REAL sigma);
/**
 * 
 */
__device__ void totalParcialMatrixf(const REAL * __restrict__ A, int f,int c,int p,REAL * result);
/**
 * 
 */
__device__ void totalParcialf(const REAL * __restrict__ A, int f,int c,PRECISION * result);
__device__ void totalParcialff(const REAL * __restrict__ A, int f,int c,float *  result);

__device__ void multmatrix_transposeCUBLAS(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value);
/**
 * 
 */
__device__ int multmatrix_transpose(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value);
__device__ void multmatrix_transpose2(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value);

__device__ int multmatrix_transpose_sigma(const REAL * __restrict__ a,int naf,int nac, const REAL * __restrict__ b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL weigth, const REAL sigma);

//__global__ void d_multmatrix_transpose(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,REAL weigth, const REAL sigma);
__global__ void d_multmatrix_transpose(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,REAL value);

__global__ void d_multmatrixIDLValueSigma(REAL *a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL *result,const REAL sigma);
__global__ void d_totalParcialMatrixf(REAL * A, int f,int c,int p,REAL *result);