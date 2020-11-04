#include "definesCuda.cuh"
#include <complex.h>
#include <math.h>

// constant variables used in file milosUtils.cu 

/**
 * 
 */
__device__ void AplicaDelta(const Init_Model * model, PRECISION * delta, Init_Model *modelout);
/**
*
*/
__device__ void AplicaDeltaf(const Init_Model * model, float * delta, Init_Model *modelout);
/**
 * 
 */
__device__ int check(Init_Model *model);
/**
 * 
 */
__device__ void FijaACeroDerivadasNoNecesarias(REAL * __restrict__ d_spectra, const int  nlambda);

/**
 * 
 */
__device__ int mil_svd(PRECISION * h, PRECISION *beta, PRECISION *delta);


/*
*
*
* Cálculo de las estimaciones clásicas.
*
*
* lambda_0 :  centro de la línea
* lambda :    vector de muestras
* nlambda :   numero de muesras
* spectro :   vector [I,Q,U,V]
* initModel:  Modelo de atmosfera a ser modificado
*
*/
__host__ __device__ void estimacionesClasicas(const PRECISION  lambda_0, const PRECISION * lambda, const int  nlambda, const float *  spectro, Init_Model *initModel, const int forInitialUse, const Cuantic *  cuantic);

/*
 *
 * nwlineas :   numero de lineas espectrales
 * wlines :		lineas spectrales
 * lambda :		wavelength axis in angstrom
			longitud nlambda
 * spectra : IQUV por filas, longitud ny=nlambda
 */

__global__ void lm_mils(const float * __restrict__ spectro,
				Init_Model * vInitModel, float * vChisqrf,
				const REAL *  slight, int * vIter,float * spectra, const int * __restrict__ displsSpectro, const int * __restrict__  sendCountPixels, const int * __restrict__ displsPixels, const int N_RTE_PARALLEL, const int numberStream);


/**
 * 	@param nlamda Number of nlambdas to register.
 * 
 * */
__device__ void InitProfilesMemoryFromDevice(int numl, ProfilesMemory * pM, const Cuantic  cuantic);


/**
 * @param pM --> pointer to memory reservation of profiles memory 
 * @param cuantic --> pointer to array with cuantic numbers. 
 * 
 * */
__device__ void FreeProfilesMemoryFromDevice(ProfilesMemory * pM,const Cuantic   cuantic);
