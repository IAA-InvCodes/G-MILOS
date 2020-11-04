#include "defines.h"
#include "definesCuda.cuh"

#define SIZE_KERNEL 50
__global__ void d_direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__global__ void d_direct_convolution(REAL * __restrict__ x, const double * __restrict__ h, int nh);
__global__ void d_direct_convolution_ic(REAL * __restrict__ x, const double * __restrict__ h, int nh, REAL Ic);
__global__ void d_convCircular(REAL * __restrict__ x, const double * __restrict__ h, const int size, REAL * __restrict__ result);
__device__ void direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__device__ void direct_convolution(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__device__ void direct_convolution_ic(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar,REAL Ic);
__device__ void direct_convolution_ic2(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar,REAL Ic);
__device__ void convCircular(const REAL * __restrict__ x, const double * __restrict__ h, int size, REAL * __restrict__ result,REAL * __restrict__ resultConv);
__device__ void direct_convolution_ic3(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,REAL Ic);
__device__ void direct_convolution_ic4(REAL *  x, int nx, const double *  h, int nh,PRECISION  *  dirConvPar,REAL Ic);
__device__ void direct_convolution2(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__device__ void direct_convolution3(REAL *  x, int nx, const double *  h, int nh,PRECISION  *  dirConvPar);