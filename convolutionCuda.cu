#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "definesCuda.cuh"
#include "defines.h"
#include "convolutionCuda.cuh"



////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex scale
__device__ __host__ cufftDoubleComplex ComplexScale(cufftDoubleComplex  a, float s)
{
    cufftDoubleComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
__device__ __host__ cufftDoubleComplex ComplexMul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    cufftDoubleComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
/**
* @param a: Array with operand a of multiplication 
* @param b: array with operand b of multpilication 
* @param c: array with result of multiplication 
* @param size: length of param arrays 
* @param scale: scale factor for multiplication 
*/
__global__ void ComplexPointwiseMulAndScale(const cufftDoubleComplex *a, const cufftDoubleComplex *b, cufftDoubleComplex * c, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        c[i] = ComplexMul(a[i], ComplexScale(b[i], scale));
    }
}

/**
* 
**/
__global__ void Scale_PSF_FFT(cufftDoubleComplex *a, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = ComplexScale(a[i], scale);
    }
}