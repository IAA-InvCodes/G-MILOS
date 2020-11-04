#include "defines.h"
#include "definesCuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cufft.h>
#include <cuComplex.h>
#include "kernels.cuh"
#include "milosUtils.cuh"


__global__ void kernel_synthesis(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,PRECISION *lambda,int nlambda,REAL *spectra,REAL  ah,PRECISION * slight, REAL * spectra_mc, int  filter, int  useFFT, cufftDoubleComplex * psfFunction){

	
	ProfilesMemory * pM = (ProfilesMemory *) malloc(sizeof(ProfilesMemory));
	InitProfilesMemoryFromDevice(nlambda,pM,cuantic);
	CUFFT_Memory * mCUFFT = (CUFFT_Memory *) malloc(sizeof(CUFFT_Memory));
	//createMemoryFFTFromDevice(mCUFFT,nlambda,useFFT);
	//mCUFFT->fftw_G_PSF = psfFunction;

	mil_sinrf(cuantic, initModel, wlines, lambda, nlambda, spectra, ah,slight, spectra_mc, filter, pM, useFFT,  mCUFFT);

	FreeProfilesMemoryFromDevice(pM,cuantic);
	free(pM);
	free(mCUFFT);
}