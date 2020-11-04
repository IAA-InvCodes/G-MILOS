#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>


__device__ __host__ cufftDoubleComplex ComplexScale(cufftDoubleComplex a, float s);
__device__ __host__ cufftDoubleComplex ComplexMul(cufftDoubleComplex a, cufftDoubleComplex b);
__global__ void ComplexPointwiseMulAndScale(const cufftDoubleComplex *a, const cufftDoubleComplex *b, cufftDoubleComplex * c, int size, float scale);
__global__ void Scale_PSF_FFT(cufftDoubleComplex *a, int size, float scale);
