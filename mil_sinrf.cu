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

extern __constant__ PRECISION d_lambda_const [MAX_LAMBDA];
//extern __constant__ PRECISION d_lambda_const_wcl  [MAX_LAMBDA];
extern __constant__ PRECISION d_wlines_const [2];
extern __constant__ PRECISION d_psfFunction_const  [MAX_LAMBDA];
extern __constant__ cuDoubleComplex d_zdenV[7];
extern __constant__ cuDoubleComplex d_zdivV[7];

__device__  void funcionComponentFor_sinrf(const int  n_pi,int  numl,const REAL *  wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION * A,PRECISION * MF,ProfilesMemory * pM,PRECISION * dopp, REAL * ulos,int * uuGlobal, int * FGlobal,int * HGlobal);
__global__ void d_fvoigt(PRECISION  damp, REAL *fi_x, REAL *shi_x, REAL * uu, REAL * F, REAL * H, REAL r_nuxB, const REAL wex,int firstZero,PRECISION  dopp, REAL  ulos);
__global__ void mil_sinrf_kernel(PRECISION E0_2,PRECISION S1, PRECISION S0, PRECISION ah,int nlambda,REAL *spectra,REAL *spectra_mc, ProfilesMemory * pM, REAL cosis_2,REAL sinis_cosa,REAL sinis_sina, REAL cosi, REAL sinis);

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
	PRECISION E0;	
	REAL ulos;
	REAL  parcial;

	//Definicion de ctes.
	//a radianes	

	AZI=initModel->az*CC;
	GM=initModel->gm*CC;


	REAL sin_gm;
	__sincosf(GM,&sin_gm,cosi);
	*sinis=sin_gm*sin_gm;
	REAL cosis=(*cosi)*(*cosi);
	*cosis_2 =__fdividef((1+(cosis)),2);
	__sincosf(2*AZI,sina,cosa);
	
	*sinda=(*cosa)*CC_2;
	*cosda=-(*sina)*CC_2;

	*sindi=(*cosi)*sin_gm*CC_2;
	*cosdi=-sin_gm*CC;
	
	REAL sinis_cosa=*sinis*(*cosa);
	REAL sinis_sina=*sinis*(*sina);
		
	E0=initModel->eta0*cuantic.FO; //y sino se definio Fo que debe de pasar 0 o 1 ...??
	//frecuency shift for v line of sight
	ulos=__fdividef((initModel->vlos*wlines[1]),(VLIGHT*initModel->dopp));

	// ******* GENERAL MULTIPLET CASE ********
	
	parcial=(((wlines[1]*wlines[1]))/initModel->dopp)*(CTE4_6_13);
	//caso multiplete						
	for(i=0;i<cuantic.N_SIG;i++){
		pM->nubB[i]=parcial*cuantic.NUB[i]; // Spliting	
	}

	for(i=0;i<cuantic.N_PI;i++){
		pM->nupB[i]=parcial*cuantic.NUP[i]; // Spliting			    
	}						

	for(i=0;i<cuantic.N_SIG;i++){
		pM->nurB[i]=-pM->nubB[(int)cuantic.N_SIG-(i+1)]; // Spliting
	}						

	*uuGlobal=0;
	*FGlobal=0;
	*HGlobal=0;
	
	//central component
											
	funcionComponentFor_sinrf(cuantic.N_PI,nlambda,cuantic.WEP,pM->nupB,pM->fi_p,pM->shi_p,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);
	
	//blue component
	funcionComponentFor_sinrf(cuantic.N_SIG,nlambda,cuantic.WEB,pM->nubB,pM->fi_b,pM->shi_b,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);

	//red component
	funcionComponentFor_sinrf(cuantic.N_SIG,nlambda,cuantic.WER,pM->nurB,pM->fi_r,pM->shi_r,&initModel->aa,&initModel->B,pM,&initModel->dopp,&ulos,uuGlobal,FGlobal,HGlobal);
	

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
		}
		// FOR USE CIRCULAR CONVOLUTION 

		for (i = 0; i < NPARMS; i++){
			d_convCircular<<<1,nlambda,nlambda*sizeof(REAL)+nlambda*sizeof(double)>>>(spectra + nlambda * i, pM->GMAC, nlambda,spectra + nlambda * i);	
			cudaDeviceSynchronize();
		}

    }//end if(MC > 0.0001)
	
	if(!macApplied && filter){
		REAL Ic;
		if(spectra[0]>spectra[nlambda - 1])
			Ic = spectra[0];
		else				
			Ic = spectra[nlambda - 1];
		direct_convolution_ic(spectra, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic); 
		//convolucion QUV
		//#pragma unroll
		for (i = 1; i < NPARMS; i++){
			direct_convolution(spectra + nlambda * i, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
		}
	}	


	if(slight!=NULL){  //ADDING THE STRAY-LIGHT PROFILE

		for(i=0;i<nlambda*NPARMS;i++){
			spectra_slight[i] = spectra[i];
			spectra[i] = spectra[i]*initModel->alfa+slight[i]*(1.0-initModel->alfa);
		}

	}

}

__device__  void funcionComponentFor_sinrf(const int  n_pi,int  numl,const REAL * wex,REAL *nuxB,REAL *fi_x, REAL *shi_x,PRECISION  *A,PRECISION  *MF,ProfilesMemory * pM, PRECISION * dopp, REAL * ulos,int * uuGlobal, int * FGlobal,int * HGlobal)
{
	REAL *uu,*F,*H;
	int i;
	//component
	
	for(i=0;i<n_pi;i++){

		uu=pM->uuGlobalInicial + ((*uuGlobal)*numl);
		F=pM->FGlobalInicial + ((*HGlobal)*numl);
		H=pM->HGlobalInicial + ((*FGlobal)*numl);		
		REAL r_nuxB = nuxB[i]* (*MF);

		d_fvoigt<<<1,numl>>>(*A,fi_x,shi_x,uu,F,H,r_nuxB,wex[i],i,*dopp,*ulos);
		cudaDeviceSynchronize();

		*uuGlobal= (*uuGlobal) + 1;
		*HGlobal= (*HGlobal) + 1;
		*FGlobal= (*FGlobal) + 1;		

	}//end for 
	
}





__global__ void d_fvoigt(PRECISION  damp,REAL *fi_x, REAL *shi_x, REAL * uu,REAL * F,REAL * H, REAL r_nuxB, const REAL  wex,int firstZero,PRECISION  dopp, REAL  ulos)
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
	REAL aux3_p_0=aux1_p_0+aux2_p_0;

	PRECISION maux1_p_0=znumr_p_11_0*zdenr_p_13_0;
	PRECISION maux2_p_0=znumi_p_6_0*zdeni_p_7_0;
	REAL maux3_p_0=maux1_p_0+maux2_p_0;
	PRECISION miaux1_p_0=znumi_p_6_0*zdenr_p_13_0;
	PRECISION miaux2_p_0=znumr_p_11_0*zdeni_p_7_0;
	REAL miaux3_p_0=miaux1_p_0-miaux2_p_0;
	REAL H_p_0_aux=__fdividef(maux3_p_0,aux3_p_0);
	REAL H_p_0=__fdividef(H_p_0_aux,(-1));
	REAL faux_p_0=__fdividef(miaux3_p_0,aux3_p_0);
	REAL F_p_0=faux_p_0*(0.5);

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

	REAL r_fi_p = pM->fi_p[i];
	REAL r_fi_b = pM->fi_b[i];
	REAL r_fi_r = pM->fi_r[i];
	REAL r_shi_p = pM->shi_p[i];
	REAL r_shi_b = pM->shi_b[i];
	REAL r_shi_r = pM->shi_r[i];	


	REAL r_parcial1 = r_fi_b+r_fi_r;
	REAL r_parcial2 = (E0_2)*( r_fi_p-__fdividef(r_parcial1,2) );
	REAL r_parcial3 = (E0_2)*(r_shi_p- __fdividef((r_shi_b+r_shi_r),2));

	REAL r_etain = ((E0_2)*(r_fi_p*sinis+(r_parcial1)*cosis_2));	
	REAL r_etaqn = (r_parcial2*sinis_cosa);
	REAL r_etaun = (r_parcial2*sinis_sina);
	REAL r_etavn = (r_fi_r-r_fi_b)*(E0_2*cosi);
	REAL r_rhoqn = (r_parcial3*sinis_cosa);
	REAL r_rhoun = (r_parcial3*sinis_sina);
	REAL r_rhovn = (r_shi_r-r_shi_b)*(E0_2*cosi);
	
	REAL r_etai =  1.0+r_etain;
	
	REAL r_etai_2 = r_etai*r_etai;
	
	REAL auxq,auxu,auxv;
	auxq=r_rhoqn*r_rhoqn;
	auxu=r_rhoun*r_rhoun;
	auxv=r_rhovn*r_rhovn;
	REAL r_gp1,r_gp3;
	r_gp1=r_etai_2-r_etaqn*r_etaqn-r_etaun*r_etaun-r_etavn*r_etavn+auxq+auxu+auxv;
	
	r_gp3=r_etai_2+auxq+auxu+auxv;
	REAL r_gp2 = r_etaqn*r_rhoqn+r_etaun*r_rhoun+r_etavn*r_rhovn;
	REAL r_dt = r_etai_2*r_gp1-r_gp2*r_gp2;
	REAL r_dti = __fdividef(1.0,r_dt);
	
	REAL r_gp4,r_gp5,r_gp6;

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

	pM->etaq[i]= r_etaqn;
	pM->etau[i]= r_etaun;
	pM->etav[i]= r_etavn;
	pM->rhoq[i]= r_rhoqn;
	pM->rhou[i]= r_rhoun;
	pM->rhov[i]= r_rhovn;
	
	pM->gp1[i]=r_gp1;
	pM->gp2[i]= r_gp2;
	pM->gp3[i]=r_gp3;
	pM->gp4[i] = r_gp4;
	pM->gp5[i] = r_gp5;	
	pM->gp6[i] = r_gp6;	
	pM->dt[i]=r_dt;

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
	}

}

