#include "definesCuda.cuh"
#include "defines.h"
#include "lib.cuh"
#include <string.h>
#include "convolution.cuh"
#include "milosUtils.cuh"
#include "readConfig.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "convolutionCuda.cuh"

extern __constant__ PRECISION d_lambda_const [MAX_LAMBDA];
//extern __constant__ PRECISION d_lambda_const_wcl  [MAX_LAMBDA];
extern __constant__ PRECISION d_wlines_const [2];
extern __constant__ PRECISION d_psfFunction_const  [MAX_LAMBDA];
extern __constant__ cuDoubleComplex d_zdenV[7];
extern __constant__ cuDoubleComplex d_zdivV[7];

__device__  void funcionComponentFor_sinrf(REAL *u,const int  n_pi,int  numl,const REAL * __restrict__ wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION  A,PRECISION  MF,ProfilesMemory * pM);
__device__  void funcionComponentFor_sinrf2(const int  n_pi,int  numl,const REAL *  wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION * A,PRECISION * MF,ProfilesMemory * pM,PRECISION * dopp, REAL * ulos,int * uuGlobal, int * FGlobal,int * HGlobal);
__device__ int fvoigt(PRECISION  damp, REAL *vv, int  nvv, REAL *h, REAL *f,ProfilesMemory * pM);
__global__ void d_fvoigt(const PRECISION  damp,const REAL * vv, int  nvv, REAL *h, REAL *f);
__global__ void d_fvoigt2(const PRECISION  damp, const REAL *  u, REAL *fi_x, REAL *shi_x, REAL * uu, REAL * F, REAL * H, REAL r_nuxB,  REAL wex,int firstZero);
__global__ void d_fvoigt3(PRECISION  damp, REAL *fi_x, REAL *shi_x, REAL * uu, REAL * F, REAL * H, REAL r_nuxB, const REAL wex,int firstZero,PRECISION  dopp, REAL  ulos);
__global__ void mil_sinrf_kernel(PRECISION E0_2,PRECISION S1, PRECISION S0, PRECISION ah,int nlambda,REAL *spectra,REAL *spectra_mc, ProfilesMemory * pM, REAL cosis_2,REAL sinis_cosa,REAL sinis_sina, REAL cosi, REAL sinis);
__global__ void mil_sinrf_set_memory_zero(ProfilesMemory * pM);

/*
	E00	int eta0; // 0
	MF	int B;    
	VL	PRECISION vlos;
	LD	PRECISION dopp;
	A	PRECISION aa;
	GM	int gm; //5
	AZI	int az;
	B0	PRECISION S0;
	B1	PRECISION S1;
	MC	PRECISION mac; //9
		PRECISION alfa;		
*/

__constant__ REAL CC = PI / 180.0;
__constant__ REAL CC_2 = (PI / 180.0) * 2;

__device__ void mil_sinrf(const Cuantic cuantic, Init_Model *initModel, const PRECISION * wlines, const int nlambda, REAL *spectra, REAL  ah, const REAL * slight, REAL * spectra_mc, REAL * spectra_slight, int  filter, ProfilesMemory * pM, REAL * cosi,REAL * sinis, REAL * sina, REAL * cosa, REAL * sinda, REAL * cosda, REAL * sindi, REAL * cosdi, REAL * cosis_2,int * uuGlobal, int * FGlobal,int * HGlobal)
{
	PRECISION GM,AZI;
	int	 i;
	//int j;
	PRECISION E0;	
	REAL ulos;
	REAL  parcial;

	//Definicion de ctes.
	//a radianes	

	AZI=initModel->az*CC;
	GM=initModel->gm*CC;

	/*REAL sin_gm=SIN(GM);
	pM->cosi=COS(GM);*/
	REAL sin_gm;
	//__sincosf(GM,&sin_gm,&pM->cosi);
	__sincosf(GM,&sin_gm,cosi);
	//__sincosf(initModel->gm*CC,&sin_gm,cosi);


	//pM->sinis=sin_gm*sin_gm;
	*sinis=sin_gm*sin_gm;
	//pM->cosis=pM->cosi*pM->cosi;
	//pM->cosis=(*cosi)*(*cosi);
	//pM->cosis_2=__fdividef((1+pM->cosis),2);
	REAL cosis=(*cosi)*(*cosi);
	//pM->cosis_2=__fdividef((1+(cosis)),2);
	*cosis_2 =__fdividef((1+(cosis)),2);
	//pM->azi_2=2*AZI;
	//REAL azi_2=2*AZI;
	//__sincosf(azi_2,&pM->sina,&pM->cosa);
	__sincosf(2*AZI,sina,cosa);
	//__sincosf(2*(initModel->az*CC),sina,cosa);
	/*pM->sina=SIN(pM->azi_2);
	pM->cosa=COS(pM->azi_2);*/
	

	/*pM->sinda=pM->cosa*CC_2;
	pM->cosda=-pM->sina*CC_2;*/

	/*pM->sinda=(*cosa)*CC_2;
	pM->cosda=-(*sina)*CC_2;*/
	*sinda=(*cosa)*CC_2;
	*cosda=-(*sina)*CC_2;

	//pM->sindi=pM->cosi*sin_gm*CC_2;
	/*pM->sindi=(*cosi)*sin_gm*CC_2;
	pM->cosdi=-sin_gm*CC;*/
	*sindi=(*cosi)*sin_gm*CC_2;
	*cosdi=-sin_gm*CC;
	
	/*pM->sinis_cosa=pM->sinis*pM->cosa;
	pM->sinis_sina=pM->sinis*pM->sina;*/

	/*pM->sinis_cosa=*sinis*pM->cosa;
	pM->sinis_sina=*sinis*pM->sina;	*/
	/*pM->sinis_cosa=*sinis*(*cosa);
	pM->sinis_sina=*sinis*(*sina);		*/
	REAL sinis_cosa=*sinis*(*cosa);
	REAL sinis_sina=*sinis*(*sina);
		
	E0=initModel->eta0*cuantic.FO; //y sino se definio Fo que debe de pasar 0 o 1 ...??
	//frecuency shift for v line of sight
	ulos=__fdividef((initModel->vlos*wlines[1]),(VLIGHT*initModel->dopp));

	//printf("\n ULOS : %f ",ulos);
	//printf("\n Dooples velocity: \n");
	//doppler velocity
	/*#pragma unroll	    
	for(i=0;i<nlambda;i=i+2){
		//pM->u[i]=((lambda[i]-wlines[1])/initModel->dopp)-ulos;
		pM->u[i]=((d_lambda_const[i]-wlines[1])/initModel->dopp)-ulos;
		pM->u[i+1]=((d_lambda_const[i+1]-wlines[1])/initModel->dopp)-ulos;
	}*/

	/*memset(pM->fi_p , 0, (nlambda)*sizeof(REAL));
	memset(pM->fi_b , 0, (nlambda)*sizeof(REAL));
	memset(pM->fi_r , 0, (nlambda)*sizeof(REAL));
	memset(pM->shi_p , 0, (nlambda)*sizeof(REAL));
	memset(pM->shi_b , 0, (nlambda)*sizeof(REAL));
	memset(pM->shi_r , 0, (nlambda)*sizeof(REAL));*/

	/*#pragma unroll
	for(i=0;i<nlambda;i++){
		pM->fi_p[i]=0;
		pM->fi_b[i]=0;
		pM->fi_r[i]=0;
	}
	
	for(i=0;i<nlambda;i++){
		pM->shi_p[i]=0;
		pM->shi_b[i]=0;
		pM->shi_r[i]=0;
	}*/

	
	// ******* GENERAL MULTIPLET CASE ********
	
	parcial=(((wlines[1]*wlines[1]))/initModel->dopp)*(CTE4_6_13);
	//printf("\n PARCIAL : %f\n",parcial);
	//caso multiplete						
	for(i=0;i<cuantic.N_SIG;i++){
		pM->nubB[i]=parcial*cuantic.NUB[i]; // Spliting	
	}

	//printf(" \n nupb \n");
	for(i=0;i<cuantic.N_PI;i++){
		pM->nupB[i]=parcial*cuantic.NUP[i]; // Spliting			    
	}						

	//printf(" \n nurb \n");
	for(i=0;i<cuantic.N_SIG;i++){
		pM->nurB[i]=-pM->nubB[(int)cuantic.N_SIG-(i+1)]; // Spliting
	}						

	/*pM->uuGlobal=0;
	pM->FGlobal=0;
	pM->HGlobal=0;*/

	*uuGlobal=0;
	*FGlobal=0;
	*HGlobal=0;
	
	//central component
											
	//funcionComponentFor_sinrf(pM->u,cuantic.N_PI,nlambda,cuantic.WEP,pM->nupB,pM->fi_p,pM->shi_p,initModel->aa,initModel->B,pM);
	funcionComponentFor_sinrf2(cuantic.N_PI,nlambda,cuantic.WEP,pM->nupB,pM->fi_p,pM->shi_p,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);
	
	//blue component
	//funcionComponentFor_sinrf(pM->u,cuantic.N_SIG,nlambda,cuantic.WEB,pM->nubB,pM->fi_b,pM->shi_b,initModel->aa,initModel->B,pM);
	funcionComponentFor_sinrf2(cuantic.N_SIG,nlambda,cuantic.WEB,pM->nubB,pM->fi_b,pM->shi_b,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);

	//red component
	//funcionComponentFor_sinrf(pM->u,cuantic.N_SIG,nlambda,cuantic.WER,pM->nurB,pM->fi_r,pM->shi_r,initModel->aa,initModel->B,pM);
	funcionComponentFor_sinrf2(cuantic.N_SIG,nlambda,cuantic.WER,pM->nurB,pM->fi_r,pM->shi_r,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);
	
	/*pM->uuGlobal=0;
	pM->FGlobal=0;
	pM->HGlobal=0;*/

	*uuGlobal=0;
	*FGlobal=0;
	*HGlobal=0;

	//*****
	mil_sinrf_kernel<<<1,nlambda>>>(__fdividef(E0,2.0),initModel->S1, initModel->S0,ah, nlambda,spectra,spectra_mc, pM,*cosis_2,sinis_cosa,sinis_sina,*cosi,*sinis);
	//dispersion profiles				
	cudaDeviceSynchronize();
	//cudaStreamSynchronize(streamId);



	int macApplied = 0;
    if(initModel->mac > 0.0001 && spectra_mc!=NULL){

		macApplied = 1;

    	fgauss(initModel->mac,nlambda,wlines[1],0,pM);  // gauss kernel is stored in global array GMAC  		 
		//convolucion de I
		if(filter){			
			direct_convolution_double(pM->GMAC, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
			/*d_direct_convolution_double<<<1,nlambda>>>(pM->GMAC, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
			cudaDeviceSynchronize();*/
		}
		// FOR USE CIRCULAR CONVOLUTION 

		for (i = 0; i < NPARMS; i++){
			//convCircular(spectra + nlambda * i, pM->GMAC, nlambda,spectra + nlambda * i,pM);	
			d_convCircular<<<1,nlambda,nlambda*sizeof(REAL)+nlambda*sizeof(double)>>>(spectra + nlambda * i, pM->GMAC, nlambda,spectra + nlambda * i);	
			cudaDeviceSynchronize();
			//cudaStreamSynchronize(streamId);
		}

    }//end if(MC > 0.0001)
	

	/*int kk;
	printf("\nSPECTRA ANTES DE LA CONVOLUCION \n");
	for (kk = 0; kk < nlambda; kk++)
	{
		printf("%f\t%e\t%e\t%e\t%e\n", (d_lambda_const[kk]-wlines[1])*1000, spectra[kk], spectra[kk + nlambda], spectra[kk + nlambda * 2], spectra[kk + nlambda * 3]);
	}
	printf("\n\n");*/
	
	if(!macApplied && filter){
		REAL Ic;
		if(spectra[0]>spectra[nlambda - 1])
			Ic = spectra[0];
		else				
			Ic = spectra[nlambda - 1];

		/*#pragma unroll
		for (i = 0; i < nlambda; i++)
			spectra[i] = Ic - spectra[i];*/

		
		direct_convolution_ic(spectra, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic); 
		//direct_convolution_ic4(spectra, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic); 
		//direct_convolution_ic2(spectra, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic); 
		/*d_direct_convolution<<<1,nlambda>>>(spectra, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
		cudaDeviceSynchronize();*/
		//d_direct_convolution_ic<<<1,nlambda,nlambda*sizeof(double)>>>(spectra, d_psfFunction_const, nlambda,Ic); 
		//cudaDeviceSynchronize();
		/*#pragma unroll
		for (i = 0; i < nlambda; i++)
			spectra[i] = Ic - spectra[i];*/

		//convolucion QUV
		//#pragma unroll
		for (i = 1; i < NPARMS; i++){
			direct_convolution(spectra + nlambda * i, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
			//direct_convolution3(spectra + nlambda * i, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
			//direct_convolution2(spectra + nlambda * i, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
			//d_direct_convolution<<<1,nlambda,nlambda*sizeof(double)>>>(spectra + nlambda * i, d_psfFunction_const, nlambda); 
			//cudaDeviceSynchronize();
			//convolve(spectra + (*nlambda) * i, (*nlambda), G, (*nlambda),spectra + (*nlambda) * i,1); 
		}
		//cudaDeviceSynchronize();
		//direct_convolution_ic3(spectra, nlambda, d_psfFunction_const, nlambda,Ic); 
	}	


	/*printf("\nSPECTRA DESPUES DE LA CONVOLUCION \n");
	for (kk = 0; kk < nlambda; kk++)
	{
		printf("%f\t%e\t%e\t%e\t%e\n", (d_lambda_const[kk]-wlines[1])*1000, spectra[kk], spectra[kk + nlambda], spectra[kk + nlambda * 2], spectra[kk + nlambda * 3]);
	}
	printf("\n\n");*/

	if(slight!=NULL){  //ADDING THE STRAY-LIGHT PROFILE

		for(i=0;i<nlambda*NPARMS;i++){
			spectra_slight[i] = spectra[i];
			spectra[i] = spectra[i]*initModel->alfa+slight[i]*(1.0-initModel->alfa);
		}

	}

}

__device__  void funcionComponentFor_sinrf(REAL *u,const int  n_pi,int  numl,const REAL * __restrict__ wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION  A,PRECISION  MF,ProfilesMemory * pM)
{
	REAL *uu,*F,*H;
	int i,j;

	//component
	
	for(i=0;i<n_pi;i++){

		uu=pM->uuGlobalInicial + (pM->uuGlobal*numl);
		F=pM->FGlobalInicial + (pM->HGlobal*numl);
		H=pM->HGlobalInicial + (pM->FGlobal*numl);
		REAL r_nuxB = nuxB[i]*MF;
		//REAL r_wex = wex[i];
		#pragma unroll
		for(j=0;j<numl;j++){
			//uu[j]=u[j]-nuxB[i]* MF;
			uu[j]=u[j]-r_nuxB;
			//uu[j+1]=u[j+1]-r_nuxB;
		}
		
		//fvoigt(A,uu,numl,H,F,pM);
		//d_fvoigt<<<1,numl,numl*sizeof(REAL)>>>(A,uu,numl,H,F);
		d_fvoigt<<<1,numl>>>(A,uu,numl,H,F);
		cudaDeviceSynchronize();
		
		#pragma unroll
		for(j=0;j<numl;j++){
			fi_x[j]=fi_x[j]+wex[i]*H[j];
			//fi_x[j+1]=fi_x[j+1]+wex[i]*H[j+1];
		}

		#pragma unroll
		for(j=0;j<numl;j++){
			shi_x[j]=(shi_x[j]+(wex[i]*F[j]*2));
			//shi_x[j+2]=(shi_x[j+2]+(wex[i]*F[j+2]*2));
		}

		pM->uuGlobal=pM->uuGlobal + 1;
		pM->HGlobal=pM->HGlobal+ 1;
		pM->FGlobal=pM->FGlobal+ 1;

	}//end for 
	
}


__device__  void funcionComponentFor_sinrf2(const int  n_pi,int  numl,const REAL * wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION  *A,PRECISION  *MF,ProfilesMemory * pM, PRECISION * dopp, REAL * ulos,int * uuGlobal, int * FGlobal,int * HGlobal)
{
	REAL *uu,*F,*H;
	int i;
	//int j;

	//component
	
	for(i=0;i<n_pi;i++){

		/*uu=pM->uuGlobalInicial + (pM->uuGlobal*numl);
		F=pM->FGlobalInicial + (pM->HGlobal*numl);
		H=pM->HGlobalInicial + (pM->FGlobal*numl);*/
		uu=pM->uuGlobalInicial + ((*uuGlobal)*numl);
		F=pM->FGlobalInicial + ((*HGlobal)*numl);
		H=pM->HGlobalInicial + ((*FGlobal)*numl);		
		REAL r_nuxB = nuxB[i]* (*MF);

		//d_fvoigt2<<<1,numl>>>(A,u,fi_x,shi_x,uu,F,H,r_nuxB,wex[i],i);
		d_fvoigt3<<<1,numl>>>(*A,fi_x,shi_x,uu,F,H,r_nuxB,wex[i],i,*dopp,*ulos);
		cudaDeviceSynchronize();

		/*printf("\n FIX_X: ");
		for(j=0;j<numl;j++){
			printf("%f\t",fi_x[j]);
		}
		printf("\n");

		printf("\n SHI_X: ");
		for(j=0;j<numl;j++){
			printf("%f\t",shi_x[j]);
		}
		printf("\n");	*/	
		/*pM->uuGlobal=pM->uuGlobal + 1;
		pM->HGlobal=pM->HGlobal+ 1;
		pM->FGlobal=pM->FGlobal+ 1;*/
		*uuGlobal= (*uuGlobal) + 1;
		*HGlobal= (*HGlobal) + 1;
		*FGlobal= (*FGlobal) + 1;		

	}//end for 
	
}


__device__ int fvoigt(PRECISION  damp, REAL *vv, int  nvv, REAL *h, REAL *f,ProfilesMemory * pM)
{

	int i, j;

	/*PRECISION a[] = {122.607931777104326, 214.382388694706425, 181.928533092181549,
									93.155580458138441, 30.180142196210589, 5.912626209773153,
									0.564189583562615};

	PRECISION b[] = {122.60793177387535, 352.730625110963558, 457.334478783897737,
									348.703917719495792, 170.354001821091472, 53.992906912940207,
									10.479857114260399, 1.};*/

	#pragma unroll
	for (i = 0; i < nvv ; i++)
	{		
		pM->z[i] =  cuCsub(make_cuDoubleComplex(damp,0),cuCmul( make_cuDoubleComplex(FABS(vv[i]), 0),make_cuDoubleComplex(0,1)));
		//pM->z[i] =  cuCsubf(make_cuFloatComplex(damp,0),cuCmulf( make_cuFloatComplex(FABS(vv[i]), 0),make_cuFloatComplex(0,1)));
	}

	#pragma unroll
	for (i = 0; i < nvv; i++)
	{
		pM->zden[i] = make_cuDoubleComplex(a_fvoigt[6], 0);
		//pM->zden[i] = make_cuFloatComplex(a_fvoigt[6], 0);
	}
	
	#pragma unroll
	for (j = 5; j >= 0; j--)
	{
		#pragma unroll
		for (i = 0; i < nvv; i++)
		{
			pM->zden[i] = cuCadd(cuCmul(pM->zden[i], pM->z[i]), make_cuDoubleComplex(a_fvoigt[j],0));
			//pM->zden[i] = cuCaddf(cuCmulf(pM->zden[i], pM->z[i]), make_cuFloatComplex(a_fvoigt[j],0));
		}
	}
	
	#pragma unroll
	for (i = 0; i < nvv; i++)
	{
		pM->zdiv[i] = cuCadd(pM->z[i], make_cuDoubleComplex(b_fvoigt[6],0));
		//pM->zdiv[i] = cuCaddf(pM->z[i], make_cuFloatComplex(b_fvoigt[6],0));
	}

	#pragma unroll
	for (j = 5; j >= 0; j--)
	{
		#pragma unroll
		for (i = 0; i < nvv; i++)
		{
			pM->zdiv[i] = cuCadd(cuCmul(pM->zdiv[i] , pM->z[i]), make_cuDoubleComplex(b_fvoigt[j],0));
			//pM->zdiv[i] = cuCaddf(cuCmulf(pM->zdiv[i] , pM->z[i]), make_cuFloatComplex(b_fvoigt[j],0));
		}
	}

	#pragma unroll
	for (i = 0; i < nvv; i++)
	{
		pM->z[i] = cuCdiv(pM->zden[i], pM->zdiv[i]);
		//pM->z[i] = cuCdivf(pM->zden[i], pM->zdiv[i]);
	}

	#pragma unroll
	for (i = 0; i < nvv; i++)
	{
		h[i] = cuCreal(pM->z[i]);
		//h[i] = cuCrealf(pM->z[i]);
	}

	#pragma unroll
	for (i = 0; i < nvv; i++)
	{
		f[i] = vv[i] >= 0 ? (double)cuCimag(pM->z[i]) * 0.5 : (double)cuCimag(pM->z[i]) * -0.5;
		//f[i] = vv[i] >= 0 ? (double)cuCimagf(pM->z[i]) * 0.5 : (double)cuCimagf(pM->z[i]) * -0.5;
	}

	return 1;
}


__global__ void d_fvoigt(const PRECISION  damp, const REAL *  vv, int  nvv, REAL *h, REAL *f)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/*extern __shared__ REAL vv_aux [];
	vv_aux[i]=vv[i];
	__syncthreads();*/

	//int j;
	REAL vv_aux = vv[i];
	//cuDoubleComplex z = cuCsub(make_cuDoubleComplex(damp,0),cuCmul( make_cuDoubleComplex(FABS(vv_aux[i]), 0),make_cuDoubleComplex(0,1)));
	cuDoubleComplex z = cuCsub(make_cuDoubleComplex(damp,0),cuCmul( make_cuDoubleComplex(FABS(vv_aux), 0),make_cuDoubleComplex(0,1)));
	/*cuDoubleComplex zden = make_cuDoubleComplex(a_fvoigt[6], 0);
	cuDoubleComplex zden2 = make_cuDoubleComplex(a_fvoigt[5], 0);
	cuDoubleComplex zden3 = make_cuDoubleComplex(a_fvoigt[4], 0);
	cuDoubleComplex zden4 = make_cuDoubleComplex(a_fvoigt[3], 0);
	cuDoubleComplex zden5 = make_cuDoubleComplex(a_fvoigt[2], 0);
	cuDoubleComplex zden6 = make_cuDoubleComplex(a_fvoigt[1], 0);
	cuDoubleComplex zden7 = make_cuDoubleComplex(a_fvoigt[0], 0);
	zden = cuCadd(cuCmul(zden, z), zden2);
	zden = cuCadd(cuCmul(zden, z), zden3);
	zden = cuCadd(cuCmul(zden, z), zden4);
	zden = cuCadd(cuCmul(zden, z), zden5);
	zden = cuCadd(cuCmul(zden, z), zden6);
	zden = cuCadd(cuCmul(zden, z), zden7);*/
	cuDoubleComplex zden = cuCadd(cuCmul(d_zdenV[0], z), d_zdenV[1]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[2]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[3]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[4]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[5]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[6]);

	cuDoubleComplex zdiv = cuCadd(z, d_zdivV[0]);
	/*cuDoubleComplex zdiv2 = make_cuDoubleComplex(b_fvoigt[5],0);
	cuDoubleComplex zdiv3 = make_cuDoubleComplex(b_fvoigt[4],0);
	cuDoubleComplex zdiv4 = make_cuDoubleComplex(b_fvoigt[3],0);
	cuDoubleComplex zdiv5 = make_cuDoubleComplex(b_fvoigt[2],0);
	cuDoubleComplex zdiv6 = make_cuDoubleComplex(b_fvoigt[1],0);
	cuDoubleComplex zdiv7 = make_cuDoubleComplex(b_fvoigt[0],0);*/
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[1]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[2]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[3]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[4]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[5]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[6]);

	/*cuDoubleComplex zden = make_cuDoubleComplex(a_fvoigt[6], 0);
	
	#pragma unroll
	for (j = 5; j >= 0; j--)
	{
		zden = cuCadd(cuCmul(zden, z), make_cuDoubleComplex(a_fvoigt[j],0));	
	}
	
	cuDoubleComplex zdiv = cuCadd(z, make_cuDoubleComplex(b_fvoigt[6],0));

	#pragma unroll
	for (j = 5; j >= 0; j--)
	{
		zdiv = cuCadd(cuCmul(zdiv , z), make_cuDoubleComplex(b_fvoigt[j],0));
	}*/
	z = cuCdiv(zden, zdiv);

	h[i] = cuCreal(z);
	//f[i] = vv_aux[i] >= 0 ? (double)cuCimag(z) * 0.5 : (double)cuCimag(z) * -0.5;
	f[i] = vv_aux >= 0 ? (double)cuCimag(z) * 0.5 : (double)cuCimag(z) * -0.5;
	
}


__global__ void d_fvoigt2(const PRECISION  damp, const REAL *  u, REAL *fi_x, REAL *shi_x, REAL * uu, REAL * F, REAL * H, REAL r_nuxB,  REAL wex,int firstZero)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//int j;
	REAL vv_aux = u[i]- r_nuxB;
	//vv_aux = vv_aux - r_nuxB;
	//REAL vv_aux = uu[i];
	uu[i]= vv_aux;
	//cuDoubleComplex z = cuCsub(make_cuDoubleComplex(damp,0),cuCmul( make_cuDoubleComplex(FABS(vv_aux), 0),make_cuDoubleComplex(0,1)));
	cuDoubleComplex z = cuCsub(make_cuDoubleComplex(damp,0),cuCmul( make_cuDoubleComplex(vv_aux, 0),make_cuDoubleComplex(0,1)));

	cuDoubleComplex zden = cuCadd(cuCmul(d_zdenV[0], z), d_zdenV[1]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[2]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[3]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[4]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[5]);
	zden = cuCadd(cuCmul(zden, z), d_zdenV[6]);

	cuDoubleComplex zdiv = cuCadd(z, d_zdivV[0]);

	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[1]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[2]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[3]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[4]);
	zdiv = cuCadd(cuCmul(zdiv , z), d_zdivV[5]);
	
	cuDoubleComplex zdivAux = cuCadd(cuCmul(zdiv , z), d_zdivV[6]);
	

	//zdiv = cuCadd(make_cuDoubleComplex((zdiv.x*z.x)-(zdiv.y*z.y),(zdiv.x*z.y)+(zdiv.y*z.x)),d_zdivV[6]);
	/*PRECISION auxZdivX = zdiv.x;
	zdiv.x = ((zdiv.x*z.x)-(zdiv.y*z.y))+ d_zdivV[6].x;
	zdiv.y = ((auxZdivX*z.y)+(zdiv.y*z.x))+ d_zdivV[6].y;*/
	/*prod = make_cuDoubleComplex ((cuCreal(x) * cuCreal(y)) -
	(cuCimag(x) * cuCimag(y)),
	(cuCreal(x) * cuCimag(y)) +
	(cuCimag(x) * cuCreal(y)));*/


	z = cuCdiv(zden, zdivAux);

	REAL r_h = cuCreal(z);
	//REAL r_f = vv_aux >= 0 ? (double)cuCimag(z) * 0.5 : (double)cuCimag(z) * -0.5;
	REAL r_f = (double)cuCimag(z) * 0.5;
	H[i] = r_h;
	F[i] = r_f;
	if(firstZero==0){
		fi_x[i]=wex*r_h;
		shi_x[i]=wex*r_f*2;
	}
	else{
		fi_x[i]=fi_x[i]+wex*r_h;
		shi_x[i]=(shi_x[i]+(wex*r_f*2));
	}
}


__global__ void d_fvoigt3(PRECISION  damp,REAL *fi_x, REAL *shi_x, REAL * uu,REAL * F,REAL * H, REAL r_nuxB, const REAL  wex,int firstZero,PRECISION  dopp, REAL  ulos)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	
	//REAL vv_aux = u[i]-r_nuxB;
	//uu[i]= vv_aux;

	//REAL vv_aux = ( __fdividef((d_lambda_const[i]-d_wlines_const[1]),(dopp))-(ulos))-r_nuxB;
	//REAL vv_aux = ( ((d_lambda_const[i]-d_wlines_const[1])/(dopp))-(ulos))-r_nuxB;
	//REAL vv_aux = (((d_lambda_const_wcl[i])/(dopp))-(ulos))-r_nuxB;
	REAL sub_aux = (d_lambda_const[i]-d_wlines_const[1]);
	PRECISION znumr_p_0_0=damp*cte_static_A_6;
	REAL vv_aux= __fdividef(sub_aux,(dopp));
	vv_aux = vv_aux - ulos;
	vv_aux = vv_aux -r_nuxB;
	uu[i]= vv_aux;
	PRECISION zi_p_0 = vv_aux;
	//PRECISION zi_p_0 = FABS(vv_aux);
	PRECISION znumi_p_0_0=zi_p_0*cte_static_A_6;

	PRECISION znumr_p_1_0=znumr_p_0_0+cte_static_A_5;
	//PRECISION znumr_p_1_0=znumr_p_0_0+-5.912626209773153;
	PRECISION znumi_p_1_0_a=znumi_p_0_0*damp;
	PRECISION znumi_p_1_0_b=znumr_p_1_0*zi_p_0;
	PRECISION znumi_p_1_0=znumi_p_1_0_a+znumi_p_1_0_b;

	PRECISION znumr_p_2_0_a=znumr_p_1_0*damp;
	PRECISION znumr_p_2_0_b=znumi_p_0_0*zi_p_0;
	PRECISION znumr_p_2_0=znumr_p_2_0_a-znumr_p_2_0_b;

	PRECISION znumr_p_3_0=znumr_p_2_0+cte_static_A_4;

	PRECISION znumi_p_2_0_a=znumi_p_1_0*damp;
	PRECISION znumi_p_2_0_b=znumr_p_3_0*zi_p_0;
	PRECISION znumi_p_2_0=znumi_p_2_0_a+znumi_p_2_0_b;	

	PRECISION znumr_p_4_0_a=znumr_p_3_0*damp;
	PRECISION znumr_p_4_0_b=znumi_p_1_0*zi_p_0;
	PRECISION znumr_p_4_0=znumr_p_4_0_a-znumr_p_4_0_b;

	PRECISION znumr_p_5_0=znumr_p_4_0+cte_static_A_3;

	PRECISION znumi_p_4_0_a=znumi_p_2_0*damp;
	PRECISION znumi_p_4_0_b=znumr_p_5_0*zi_p_0;
	PRECISION znumi_p_4_0=znumi_p_4_0_a+znumi_p_4_0_b;


	PRECISION znumr_p_6_0_a=znumr_p_5_0*damp;
	PRECISION znumr_p_6_0_b=znumi_p_2_0*zi_p_0;
	PRECISION znumr_p_6_0=znumr_p_6_0_a-znumr_p_6_0_b;

	PRECISION znumr_p_7_0=znumr_p_6_0+cte_static_A_2;

	PRECISION znumi_p_5_0_a=znumi_p_4_0*damp;
	PRECISION znumi_p_5_0_b=znumr_p_7_0*zi_p_0;
	PRECISION znumi_p_5_0=znumi_p_5_0_a+znumi_p_5_0_b;

	PRECISION znumr_p_8_0_a=znumr_p_7_0*damp;
	PRECISION znumr_p_8_0_b=znumi_p_4_0*zi_p_0;
	PRECISION znumr_p_8_0=znumr_p_8_0_a-znumr_p_8_0_b;

	PRECISION znumr_p_9_0=znumr_p_8_0+cte_static_A_1;

	PRECISION znumi_p_6_0_a=znumi_p_5_0*damp;
	PRECISION znumi_p_6_0_b=znumr_p_9_0*zi_p_0;
	PRECISION znumi_p_6_0=znumi_p_6_0_a+znumi_p_6_0_b;

	PRECISION znumr_p_10_0_a=znumr_p_9_0*damp;
	PRECISION znumr_p_10_0_b=znumi_p_5_0*zi_p_0;
	PRECISION znumr_p_10_0=znumr_p_10_0_a-znumr_p_10_0_b;

	PRECISION znumr_p_11_0=znumr_p_10_0+cte_static_A_0;

	//#######################################################
	//#######################################################
	//----------> Den

	PRECISION zdenr_p_1_0=damp+cte_static_B_6;

	PRECISION zdeni_p_1_0_a=zi_p_0*damp;
	PRECISION zdeni_p_1_0_b=zdenr_p_1_0*zi_p_0;
	PRECISION zdeni_p_1_0=zdeni_p_1_0_a+zdeni_p_1_0_b;

	PRECISION zdenr_p_2_0_a=zdenr_p_1_0*damp;
	PRECISION zdenr_p_2_0_b=zi_p_0*zi_p_0;
	PRECISION zdenr_p_2_0=zdenr_p_2_0_a-zdenr_p_2_0_b;

	PRECISION zdenr_p_3_0=zdenr_p_2_0+cte_static_B_5;

	PRECISION zdeni_p_2_0_a=zdeni_p_1_0*damp;
	PRECISION zdeni_p_2_0_b=zdenr_p_3_0*zi_p_0;
	PRECISION zdeni_p_2_0=zdeni_p_2_0_a+zdeni_p_2_0_b;

	PRECISION zdenr_p_4_0_a=zdenr_p_3_0*damp;
	PRECISION zdenr_p_4_0_b=zdeni_p_1_0*zi_p_0;
	PRECISION zdenr_p_4_0=zdenr_p_4_0_a-zdenr_p_4_0_b;

	PRECISION zdenr_p_5_0=zdenr_p_4_0+cte_static_B_4;
	
	PRECISION zdeni_p_4_0_a=zdeni_p_2_0*damp;
	PRECISION zdeni_p_4_0_b=zdenr_p_5_0*zi_p_0;
	PRECISION zdeni_p_4_0=zdeni_p_4_0_a+zdeni_p_4_0_b;

	PRECISION zdenr_p_6_0_a=zdenr_p_5_0*damp;
	PRECISION zdenr_p_6_0_b=zdeni_p_2_0*zi_p_0;
	PRECISION zdenr_p_6_0=zdenr_p_6_0_a-zdenr_p_6_0_b;

	PRECISION zdenr_p_7_0=zdenr_p_6_0+cte_static_B_3;

	PRECISION zdeni_p_5_0_a=zdeni_p_4_0*damp;
	PRECISION zdeni_p_5_0_b=zdenr_p_7_0*zi_p_0;
	PRECISION zdeni_p_5_0=zdeni_p_5_0_a+zdeni_p_5_0_b;

	PRECISION zdenr_p_8_0_a=zdenr_p_7_0*damp;
	PRECISION zdenr_p_8_0_b=zdeni_p_4_0*zi_p_0;
	PRECISION zdenr_p_8_0=zdenr_p_8_0_a-zdenr_p_8_0_b;

	PRECISION zdenr_p_9_0=zdenr_p_8_0+cte_static_B_2;

	PRECISION zdeni_p_6_0_a=zdeni_p_5_0*damp;
	PRECISION zdeni_p_6_0_b=zdenr_p_9_0*zi_p_0;
	PRECISION zdeni_p_6_0=zdeni_p_6_0_a+zdeni_p_6_0_b;

	PRECISION zdenr_p_10_0_a=zdenr_p_9_0*damp;
	PRECISION zdenr_p_10_0_b=zdeni_p_5_0*zi_p_0;
	PRECISION zdenr_p_10_0=zdenr_p_10_0_a-zdenr_p_10_0_b;

	PRECISION zdenr_p_11_0=zdenr_p_10_0+cte_static_B_1;

	PRECISION zdeni_p_7_0_a=zdeni_p_6_0*damp;
	PRECISION zdeni_p_7_0_b=zdenr_p_11_0*zi_p_0;
	PRECISION zdeni_p_7_0=zdeni_p_7_0_a+zdeni_p_7_0_b;

	PRECISION zdenr_p_12_0_a=zdenr_p_11_0*damp;
	PRECISION zdenr_p_12_0_b=zdeni_p_6_0*zi_p_0;
	PRECISION zdenr_p_12_0=zdenr_p_12_0_a-zdenr_p_12_0_b;

	PRECISION zdenr_p_13_0=zdenr_p_12_0+cte_static_B_0;

	//########################################

	PRECISION aux1_p_0=zdenr_p_13_0*zdenr_p_13_0;
	PRECISION aux2_p_0=zdeni_p_7_0*zdeni_p_7_0;
	//PRECISION aux3_p_0=aux1_p_0+aux2_p_0;
	REAL aux3_p_0=aux1_p_0+aux2_p_0;

	PRECISION maux1_p_0=znumr_p_11_0*zdenr_p_13_0;
	PRECISION maux2_p_0=znumi_p_6_0*zdeni_p_7_0;
	//PRECISION maux3_p_0=maux1_p_0+maux2_p_0;
	REAL maux3_p_0=maux1_p_0+maux2_p_0;
	PRECISION miaux1_p_0=znumi_p_6_0*zdenr_p_13_0;
	PRECISION miaux2_p_0=znumr_p_11_0*zdeni_p_7_0;
	//PRECISION miaux3_p_0=miaux1_p_0-miaux2_p_0;
	REAL miaux3_p_0=miaux1_p_0-miaux2_p_0;
	//PRECISION H_p_0_aux=maux3_p_0/aux3_p_0;
	REAL H_p_0_aux=__fdividef(maux3_p_0,aux3_p_0);
	REAL H_p_0=__fdividef(H_p_0_aux,(-1));
	//PRECISION faux_p_0=miaux3_p_0/aux3_p_0;
	REAL faux_p_0=__fdividef(miaux3_p_0,aux3_p_0);
	REAL F_p_0=faux_p_0*(0.5);
	
	//PRECISION F_p_0= (vv_aux >= 0 ? faux_p_0 * 0.5 : faux_p_0 * -0.5);	

	H[i] = H_p_0;
	F[i] = F_p_0;
	if(firstZero==0){
		fi_x[i]=wex*H_p_0;
		shi_x[i]=wex*F_p_0*2;
	}
	else{
		fi_x[i]=fi_x[i]+wex*H_p_0;
		shi_x[i]=(shi_x[i]+(wex*F_p_0*2));
	}
}

__global__ void mil_sinrf_kernel(PRECISION E0_2,PRECISION S1, PRECISION S0,PRECISION ah,int nlambda,REAL *spectra,REAL *spectra_mc, ProfilesMemory * pM, REAL cosis_2,REAL sinis_cosa,REAL sinis_sina, REAL cosi, REAL sinis){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	/*extern __shared__ REAL fi_p[];
	REAL * fi_b = (REAL *)&fi_p[nlambda];
	REAL * fi_r = (REAL *)&fi_b[nlambda];
	REAL * shi_p = (REAL *)&fi_r[nlambda];
	REAL * shi_b = (REAL *)&shi_p[nlambda];
	REAL * shi_r = (REAL *)&shi_b[nlambda];*/
	//REAL * etai = (REAL *)&shi_r[nlambda];

	/*fi_p[i] = pM->fi_p[i];
	fi_b[i] = pM->fi_b[i];
	fi_r[i] = pM->fi_r[i];
	shi_p[i] = pM->shi_p[i];
	shi_b[i] = pM->shi_b[i];
	shi_r[i] = pM->shi_r[i];*/
	//etai[i] = pM->etai[i];

	//__syncthreads();
	REAL r_fi_p = pM->fi_p[i];
	REAL r_fi_b = pM->fi_b[i];
	REAL r_fi_r = pM->fi_r[i];
	REAL r_shi_p = pM->shi_p[i];
	REAL r_shi_b = pM->shi_b[i];
	REAL r_shi_r = pM->shi_r[i];	


	REAL r_parcial1 = r_fi_b+r_fi_r;
	REAL r_parcial2 = (E0_2)*( r_fi_p-__fdividef(r_parcial1,2) );
	REAL r_parcial3 = (E0_2)*(r_shi_p- __fdividef((r_shi_b+r_shi_r),2));

	//printf("\n PARCIAL 1 %f PARCIAL 2 %f PARCIAL 3 %f hebra %i",r_parcial1,r_parcial2,r_parcial3,i);
	//pM->cosis_2,pM->sinis_cosa,pM->sinis_sina,pM->cosi

	REAL r_etain = ((E0_2)*(r_fi_p*sinis+(r_parcial1)*cosis_2));	
	REAL r_etaqn = (r_parcial2*sinis_cosa);
	REAL r_etaun = (r_parcial2*sinis_sina);
	REAL r_etavn = (r_fi_r-r_fi_b)*(E0_2*cosi);
	REAL r_rhoqn = (r_parcial3*sinis_cosa);
	REAL r_rhoun = (r_parcial3*sinis_sina);
	REAL r_rhovn = (r_shi_r-r_shi_b)*(E0_2*cosi);
	
	//REAL r_etai =  etai[i]+r_etain;
	REAL r_etai =  1.0+r_etain;
	/*REAL r_etaq = pM->etaq[i]+r_etaqn;
	REAL r_etau = pM->etau[i]+r_etaun;
	REAL r_etav = pM->etav[i]+r_etavn;
	REAL r_rhoq = pM->rhoq[i]+r_rhoqn;
	REAL r_rhou = pM->rhou[i]+r_rhoun;
	REAL r_rhov = pM->rhov[i]+r_rhovn;*/
	
	/*REAL r_etaq = r_etaqn;
	REAL r_etau = r_etaun;
	REAL r_etav = r_etavn;
	REAL r_rhoq = r_rhoqn;
	REAL r_rhou = r_rhoun;
	REAL r_rhov = r_rhovn;*/

	REAL r_etai_2 = r_etai*r_etai;
	
	REAL auxq,auxu,auxv;
	auxq=r_rhoqn*r_rhoqn;
	auxu=r_rhoun*r_rhoun;
	auxv=r_rhovn*r_rhovn;
	REAL r_gp1,r_gp3;
	//r_gp1=r_etai_2-r_etaq*r_etaq-r_etau*r_etau-r_etav*r_etav+auxq+auxu+auxv;
	r_gp1=r_etai_2-r_etaqn*r_etaqn-r_etaun*r_etaun-r_etavn*r_etavn+auxq+auxu+auxv;
	
	r_gp3=r_etai_2+auxq+auxu+auxv;
	//REAL r_gp2 = r_etaq*r_rhoq+r_etau*r_rhou+r_etav*r_rhov;
	REAL r_gp2 = r_etaqn*r_rhoqn+r_etaun*r_rhoun+r_etavn*r_rhovn;
	REAL r_dt = r_etai_2*r_gp1-r_gp2*r_gp2;
	REAL r_dti = __fdividef(1.0,r_dt);
	
	REAL r_gp4,r_gp5,r_gp6;
	//r_gp4 = r_etai_2*r_etaq+r_etai*(r_etav*r_rhou-r_etau*r_rhov);
	//r_gp5 = r_etai_2*r_etau+r_etai*(r_etaq*r_rhov-r_etav*r_rhoq);	
	//r_gp6 = r_etai_2*r_etav+r_etai*(r_etau*r_rhoq-r_etaq*r_rhou);

	r_gp4 = r_etai_2*r_etaqn+r_etai*(r_etavn*r_rhoun-r_etaun*r_rhovn);
	r_gp5 = r_etai_2*r_etaun+r_etai*(r_etaqn*r_rhovn-r_etavn*r_rhoqn);	
	r_gp6 = r_etai_2*r_etavn+r_etai*(r_etaun*r_rhoqn-r_etaqn*r_rhoun);

	REAL r_gp4_gp2_rhoq,r_gp5_gp2_rhou, r_gp6_gp2_rhov;
	r_gp4_gp2_rhoq = r_gp4+r_rhoqn*r_gp2;
	r_gp5_gp2_rhou = r_gp5+r_rhoun*r_gp2;
	r_gp6_gp2_rhov = r_gp6+r_rhovn*r_gp2;

	
	REAL r_dtiaux = r_dti*( -((S1)*ah) );
	
	//espectro
	
	REAL spectra0 = S0-r_dtiaux*r_etai*r_gp3;
	REAL spectra1 = (r_dtiaux*(r_gp4_gp2_rhoq));
	REAL spectra2 = (r_dtiaux*(r_gp5_gp2_rhou));
	REAL spectra3 = (r_dtiaux*(r_gp6_gp2_rhov));


	pM->parcial1[i]=r_parcial1;
	pM->parcial2[i]=r_parcial2;
	pM->parcial3[i]= r_parcial3;
	//pM->etai[i]= r_etai;

	pM->etaq[i]= r_etaqn;
	//pM->etaqn[i]= r_etaqn;

	pM->etau[i]= r_etaun;
	//pM->etaun[i]= r_etaun;

	pM->etav[i]= r_etavn;
	//pM->etavn[i]= r_etavn;

	pM->rhoq[i]= r_rhoqn;
	//pM->rhoqn[i]= r_rhoqn;

	pM->rhou[i]= r_rhoun;
	//pM->rhoun[i] = r_rhoun;

	pM->rhov[i]= r_rhovn;
	//pM->rhovn[i]= r_rhovn;	

	//	pM->etai_2[i]=r_etai_2;
	pM->gp1[i]=r_gp1;
	pM->gp2[i]= r_gp2;
	pM->gp3[i]=r_gp3;
	pM->gp4[i] = r_gp4;
	pM->gp5[i] = r_gp5;	
	pM->gp6[i] = r_gp6;	
	pM->dt[i]=r_dt;
	//pM->dti[i]=r_dti;
	//pM->dtiaux[i] = r_dtiaux;
	pM->etain[i]= r_etain;
	
	pM->gp4_gp2_rhoq[i] = r_gp4_gp2_rhoq;
	pM->gp5_gp2_rhou[i] = r_gp5_gp2_rhou;
	pM->gp6_gp2_rhov[i] = r_gp6_gp2_rhov;

	spectra[i] = spectra0;
	spectra[i+nlambda] = spectra1;
	spectra[i+nlambda*2] = spectra2;
	spectra[i+nlambda*3] = spectra3;

	if(spectra_mc!=NULL){
		spectra_mc[i] = spectra0;
		spectra_mc[i+nlambda  ] = spectra1;
		spectra_mc[i+nlambda*2]=  spectra2;
		spectra_mc[i+nlambda*3] = spectra3;
		/*spectra_mc[i*NPARMS] = spectra0;
		spectra_mc[(i*NPARMS)+1] = spectra1;
		spectra_mc[(i*NPARMS)+2] = spectra2;
		spectra_mc[(i*NPARMS)+3] = spectra3;*/
	}

}



__global__ void mil_sinrf_set_memory_zero(ProfilesMemory * pM){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	pM->etaq[i] = 0;
	pM->etau[i] = 0;
	pM->etav[i] = 0;
	pM->rhoq[i] = 0;
	pM->rhou[i] = 0;
	pM->rhov[i] = 0;

	pM->fi_p[i] = 0;
	pM->fi_b[i] = 0;
	pM->fi_r[i] = 0;
	pM->shi_p[i] = 0;
	pM->shi_b[i] = 0;
	pM->shi_r[i] = 0;

}