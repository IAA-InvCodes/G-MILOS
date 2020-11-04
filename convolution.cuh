#include "defines.h"
#include "definesCuda.cuh"


__global__ void d_direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__global__ void d_direct_convolution(REAL * __restrict__ x, const double * __restrict__ h, int nh);
__global__ void d_direct_convolution_ic(REAL * __restrict__ x, const double * __restrict__ h, int nh, REAL Ic);
__global__ void d_convCircular(REAL * __restrict__ x, const double * __restrict__ h, const int size, REAL * __restrict__ result);
__device__ void direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__device__ void direct_convolution(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar);
__device__ void direct_convolution_ic(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar,REAL Ic);
__device__ void convCircular(const REAL * __restrict__ x, const double * __restrict__ h, int size, REAL * __restrict__ result,REAL * __restrict__ resultConv);
