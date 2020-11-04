#include "defines.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cufft.h>
#include <cuComplex.h>

#ifndef DEFINES_CUH_
#define DEFINES_CUH_


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


// STRUCT WITH INTERMEDIATE MEMORY FOR EACH PIXEL CALCULATION 


struct PROFILES_MEMORY{
	REAL * gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;
	REAL *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;
	REAL sin_gm,azi_2,sinis,cosis,cosis_2,cosi,sina,cosa,sinda,cosda,sindi,cosdi,sinis_cosa,sinis_sina;
	REAL *fi_p,*fi_b,*fi_r,*shi_p,*shi_b,*shi_r;
	REAL *etain,*etaqn,*etaun,*etavn,*rhoqn,*rhoun,*rhovn;
	REAL *etai,*etaq,*etau,*etav,*rhoq,*rhou,*rhov;
	REAL *parcial1,*parcial2,*parcial3;	
	REAL *nubB,*nupB,*nurB;
	REAL *uuGlobalInicial;
	REAL *HGlobalInicial;
	REAL *FGlobalInicial;
	REAL *dtaux, *etai_gp3, *ext1, *ext2, *ext3, *ext4;
	REAL *dgp1, *dgp2, *dgp3, *dgp4, *dgp5, *dgp6, *d_dt;
	REAL *d_ei, *d_eq, *d_eu, *d_ev, *d_rq, *d_ru, *d_rv;
	REAL *dfi, *dshi;
	REAL *u, * dtiaux;
	REAL * dH_u, *dF_u, *auxCte;
	REAL *spectra, *d_spectra, *spectra_mac,*spectra_slight, * d_spectra_backup;
	REAL *opa;
	cuDoubleComplex  *z,* zden, * zdiv;
	//cuFloatComplex   *z,* zden, * zdiv;
	PRECISION *GMAC,*GMAC_DERIV,*dirConvPar;
	REAL * resultConv;
	REAL * AP, * BT;
	int uuGlobal,HGlobal,FGlobal;
	PRECISION * term;
	float *v,*w;
};

typedef struct PROFILES_MEMORY ProfilesMemory;





struct CUFFT_MEMORY{
	cufftDoubleComplex * inSpectraFwPSF, *inSpectraBwPSF, *outSpectraFwPSF, *outSpectraBwPSF;
	cufftDoubleComplex * inSpectraFwMAC, *inSpectraBwMAC, *outSpectraFwMAC, *outSpectraBwMAC;

	cufftDoubleComplex * inFilterMAC, * inFilterMAC_DERIV, * outFilterMAC, * outFilterMAC_DERIV;

	cufftDoubleComplex * fftw_G_PSF, * fftw_G_MAC_PSF, * fftw_G_MAC_DERIV_PSF;
	cufftDoubleComplex * inPSF_MAC, * inMulMacPSF, * inPSF_MAC_DERIV, *inMulMacPSFDeriv, *outConvFilters, * outConvFiltersDeriv;

	cufftHandle plan1D;
};

typedef struct CUFFT_MEMORY CUFFT_Memory;


__device__ void fgauss(const PRECISION  MC, const int  neje, const PRECISION  landa, const int deriv,ProfilesMemory * pM);


//__device__  int me_der(const Cuantic * cuantic,Init_Model *initModel,const PRECISION * wlines,const int nlambda,REAL *d_spectra,REAL *spectra, REAL * spectra_slight, REAL ah,const REAL * slight,int filter,ProfilesMemory * pM, const int * fix);
__device__  int me_der(const Cuantic * cuantic,Init_Model *initModel,const PRECISION * wlines,const int nlambda,REAL *d_spectra,REAL *spectra, REAL * spectra_slight, REAL ah,const REAL * slight,int filter,ProfilesMemory * pM, const int * fix,REAL cosi, REAL sinis,REAL sina, REAL cosa, REAL sinda, REAL cosda, REAL sindi, REAL cosdi,REAL cosis_2, int * uuGlobal, int * FGlobal,int * HGlobal);
//__device__ void mil_sinrf(const Cuantic cuantic,Init_Model *initModel,const PRECISION * wlines, const int nlambda,REAL *spectra,REAL  ah, const REAL * slight, REAL * spectra_mc, REAL * spectra_slight, int  filter, ProfilesMemory * pM);
__device__ void mil_sinrf(const Cuantic cuantic,Init_Model *initModel,const PRECISION * wlines, const int nlambda,REAL *spectra,REAL  ah, const REAL * slight, REAL * spectra_mc, REAL * spectra_slight, int  filter, ProfilesMemory * pM,REAL * cosi, REAL * sinis,REAL * sina, REAL * cosa, REAL * sinda, REAL * cosda, REAL * sindi, REAL * cosdi,REAL *cosis_2,int * uuGlobal, int * FGlobal,int * HGlobal);
__device__ int fvoigt(PRECISION * damp, REAL *vv, int  nvv, REAL *h, REAL *f,ProfilesMemory * pM);


__device__ const PRECISION a_fvoigt[] = {122.607931777104326, 214.382388694706425, 181.928533092181549,
									93.155580458138441, 30.180142196210589, 5.912626209773153,
									0.564189583562615};

__device__	const PRECISION b_fvoigt[] = {122.60793177387535, 352.730625110963558, 457.334478783897737,
									348.703917719495792, 170.354001821091472, 53.992906912940207,
									10.479857114260399, 1.};

__device__	const PRECISION	cte_static_A_0=-122.607931777104326;
__device__	const PRECISION	cte_static_A_1=-214.382388694706425;
__device__	const PRECISION	cte_static_A_2=-181.928533092181549;
__device__	const PRECISION	cte_static_A_3=-93.155580458138441;
__device__	const PRECISION	cte_static_A_4=-30.180142196210589;
__device__	const PRECISION	cte_static_A_5=-5.912626209773153;
__device__	const PRECISION	cte_static_A_6=-0.564189583562615;

__device__	const PRECISION	cte_static_B_0=122.60793177387535;
__device__	const PRECISION	cte_static_B_1=352.730625110963558;
__device__	const PRECISION	cte_static_B_2=457.334478783897737;
__device__	const PRECISION	cte_static_B_3=348.703917719495792;
__device__	const PRECISION	cte_static_B_4=170.354001821091472;
__device__	const PRECISION	cte_static_B_5=53.992906912940207;
__device__	const PRECISION	cte_static_B_6=10.479857114260399;

#define TAMANIO_SVD 10
__device__ void svdcordic(PRECISION *a, int m, int n, PRECISION w[TAMANIO_SVD], PRECISION v[TAMANIO_SVD * TAMANIO_SVD],int max_iter);
__device__ void svdcordicf(float *a, int m, int n, float w[TAMANIO_SVD], float v[TAMANIO_SVD * TAMANIO_SVD],int max_iter);
__global__ void svdcordic_kernel(float *a, int m, int n, float w[TAMANIO_SVD], float v[TAMANIO_SVD * TAMANIO_SVD],int max_iter);

#define NORMALIZATION_SVD 0 //1 for using normalization matrixes ONLY  in the SVD_CORDIC

#define NUM_ITER_SVD_CORDIC 18 //9,18,27,36  --> 18 parece ok!
/*#define LIMITE_INFERIOR_PRECISION_SVD pow(2.0,-39)
#define LIMITE_INFERIOR_PRECISION_TRIG pow(2.0,-39)
#define LIMITE_INFERIOR_PRECISION_SINCOS pow(2.0,-39)*/


#define MAX_LAMBDA 50
#define BLOCK_SIZE 32

#endif /*DEFINES_CUH_*/
