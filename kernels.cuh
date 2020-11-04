#include "defines.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cufft.h>
#include <cuComplex.h>


__global__ void kernel_synthesis(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda,REAL *spectra,REAL  ah,PRECISION * slight, REAL * spectra_mc, int  filter, int  useFFT, cufftDoubleComplex * psfFunction);