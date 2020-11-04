#include <time.h>
#include "definesCuda.cuh"
#include "defines.h"
#include "lib.cuh"
#include <string.h>
#include "milosUtils.cuh"
#include "convolution.cuh"


extern __constant__ PRECISION d_lambda_const [MAX_LAMBDA];
extern __constant__ PRECISION d_psfFunction_const [MAX_LAMBDA];

__device__ void funcionComponentFor(const int n_pi,PRECISION  iwlines,int  numl,const REAL * __restrict__ wex,REAL *nuxB,REAL *dfi,REAL *dshi,PRECISION  LD,PRECISION  A,int desp,ProfilesMemory * pM,REAL auxCte,int * uuGlobal, int * FGlobal,int * HGlobal);

__global__ void  d_funcionComponentFor(const int i,int  numl,const REAL * __restrict__ wex,const REAL * __restrict__ nuxB,REAL *dfi,REAL *dshi,const PRECISION  LD,const PRECISION  A,const int desp,ProfilesMemory * pM,const REAL * __restrict__ H, const REAL * __restrict__ F,REAL *uu,REAL auxCte);

//__global__ void me_der_kernel(PRECISION S1,int nlambda,REAL *d_spectra, REAL ah, ProfilesMemory * pM,PRECISION E0,REAL cosi,REAL sinis,REAL sina, REAL cosa,REAL sinda,REAL  cosda,REAL  sindi,REAL cosdi,REAL cosis_2);
__global__ void me_der_kernel(PRECISION S1,int nlambda,REAL *d_spectra, REAL ah, ProfilesMemory * pM,PRECISION E0,REAL cosi,REAL sinis,REAL sina, REAL cosa,REAL sinda,REAL  cosda,REAL  sindi,REAL cosdi,REAL cosis_2,REAL E0_2,REAL cosi_2_E0,REAL sinis_cosa,REAL sinis_sina,REAL sindi_cosa,REAL sindi_sina,REAL cosdi_E0_2,REAL cosis_2_E0_2,REAL sinis_E0_2);
__device__ void Resetear_Valores_Intermedios(int nlambda,ProfilesMemory * pM);



/*
	E00	int eta0; // 0
	MF	int B;    
	VL	double vlos;
	LD	double dopp;
	A	double aa;
	GM	int gm; //5
	AZI	int az;
	B0	double S0;
	B1	double S1;
	MC	double mac; //9
		double alfa;		
*/

/*
 * 
 */
__device__ void funcionComponentFor(const int n_pi,PRECISION  iwlines,int  numl,const REAL * __restrict__ wex,REAL *nuxB,REAL *dfi,REAL *dshi,PRECISION  LD,PRECISION  A,int desp,ProfilesMemory * pM,REAL auxCte,int * uuGlobal, int * FGlobal,int * HGlobal)
{
	
	int i;
	//int j;

	
	REAL *H,*F,*uu;
	
	/*for(j=0;j<numl;j++){
		pM->auxCte[j]=(-iwlines)/(VLIGHT*LD);
	}*/


	//component
	for(i=0;i<n_pi;i++){

		/*uu=pM->uuGlobalInicial+ (pM->uuGlobal*numl);
		F=pM->FGlobalInicial + (pM->HGlobal*numl);
		H=pM->HGlobalInicial + (pM->FGlobal*numl);*/

		uu=pM->uuGlobalInicial+ ((*uuGlobal)*numl);
		F=pM->FGlobalInicial + ((*HGlobal)*numl);
		H=pM->HGlobalInicial + ((*FGlobal)*numl);

		/*for(j=0;j<numl;j++){
			pM->dH_u[j]=((4*A*F[j])-(2*uu[j]*H[j]))*wex[i];
		}

		for(j=0;j<numl;j++){
			pM->dF_u[j]=(RR-A*H[j]-2*uu[j]*F[j])*wex[i];//
		}
		
		for(j=0;j<numl;j++){
			uu[j]=-uu[j]/LD;
		}

		//numl*4*3
		// a   b c
		//col a,fil b,desplazamiento c
		//b*a+numl+(numl*4*c)
		//dfi
		for(j=0;j<numl;j++){
			//(*,0,desp)=>0*numl+j+(numl*4*0)
			dfi[j+(numl*4*desp)]=dfi[j+(numl*4*desp)]+pM->dH_u[j]*(-nuxB[i]);
		}
		
		for(j=0;j<numl;j++){
			//(*,1,desp)=>1*numl+j+(numl*4*0)
			dfi[numl+j+(numl*4*desp)]=dfi[numl+j+(numl*4*desp)]+pM->dH_u[j]*pM->auxCte[j];
		}

		for(j=0;j<numl;j++){
			//(*,2,desp)=>1*numl+j+(numl*4*0)
			dfi[2*numl+j+(numl*4*desp)]=dfi[2*numl+j+(numl*4*desp)]+(pM->dH_u[j]*uu[j]); 											
		}								

		for(j=0;j<numl;j++){
			//(*,3,desp)=>1*numl+j+(numl*4*0)
			dfi[3*numl+j+(numl*4*desp)]=dfi[3*numl+j+(numl*4*desp)]+(-2*pM->dF_u[j]);//dH_a[j];						
		}
		
		//dshi
		for(j=0;j<numl;j++){
			//(*,0,desp)=>0*numl+j+(numl*4*0)
			dshi[j+(numl*4*desp)]=dshi[j+(numl*4*desp)]+(pM->dF_u[j])*(-nuxB[i]);
		}
						
		for(j=0;j<numl;j++){
			//(*,1,desp)=>1*numl+j+(numl*4*0)
			dshi[numl+j+(numl*4*desp)]=dshi[numl+j+(numl*4*desp)]+pM->dF_u[j]*pM->auxCte[j];
		}

		for(j=0;j<numl;j++){
			//(*,2,desp)=>1*numl+j+(numl*4*0)
			dshi[2*numl+j+(numl*4*desp)]=dshi[2*numl+j+(numl*4*desp)]+(pM->dF_u[j]*uu[j]); 											
		}								
		
		for(j=0;j<numl;j++){
			//(*,3,desp)=>1*numl+j+(numl*4*0)
			dshi[3*numl+j+(numl*4*desp)]=dshi[3*numl+j+(numl*4*desp)]+(pM->dH_u[j]/2);						
		}*/									
		
		//d_funcionComponentFor<<<1,numl,numl*sizeof(REAL)>>>(i,numl,wex,nuxB,dfi,dshi,LD,A,desp,pM,H,F,uu,auxCte);
		d_funcionComponentFor<<<1,numl>>>(i,numl,wex,nuxB,dfi,dshi,LD,A,desp,pM,H,F,uu,auxCte);
		cudaDeviceSynchronize();

		/*pM->uuGlobal=pM->uuGlobal+1;
		pM->HGlobal=pM->HGlobal+1;
		pM->FGlobal=pM->FGlobal+1;*/
		*uuGlobal= (*uuGlobal)+1;
		*HGlobal= (*HGlobal)+1;
		*FGlobal= (*FGlobal)+1;		
	}
}


__global__ void  d_funcionComponentFor(const int i,int  numl,const REAL * __restrict__ wex,const REAL * __restrict__ nuxB,REAL *dfi,REAL *dshi,const PRECISION  LD,const PRECISION  A,const int desp,ProfilesMemory * pM,const REAL * __restrict__ H, const REAL * __restrict__ F,REAL *uu, REAL auxCte)
{
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	//extern __shared__ REAL s[];
	/*REAL * s_uu = s;
	s_uu[j] = uu[j];
	__syncthreads();*/
	REAL s_uu = uu[j];
	
	REAL s_F = F[j];
	REAL s_H = H[j];
	/*REAL r_dH_u =  ((4*A*s_F)-(2*s_uu[j]*s_H))*wex[i];
	REAL r_dF_u =  (RR-A*s_H-2*s_uu[j]*s_F)*wex[i];
	s_uu[j]=-s_uu[j]/LD;
	dfi[j+(numl*4*desp)]=0+r_dH_u*(-nuxB[i]);
	dfi[numl+j+(numl*4*desp)]=0+r_dH_u*auxCte;
	dfi[2*numl+j+(numl*4*desp)]=0+(r_dH_u*s_uu[j]); 											
	dfi[3*numl+j+(numl*4*desp)]=0+(-2*r_dF_u);						

	dshi[j+(numl*4*desp)]= 0+(r_dF_u)*(-nuxB[i]);						
	dshi[numl+j+(numl*4*desp)]=0+r_dF_u*auxCte;
	dshi[2*numl+j+(numl*4*desp)]=0+(r_dF_u*s_uu[j]); 											
	dshi[3*numl+j+(numl*4*desp)]=0+(r_dH_u/2);

	uu[j]=s_uu[j];*/

	REAL r_dH_u =  ((4*A*s_F)-(2*s_uu*s_H))*wex[i];
	REAL r_dF_u =  (RR-A*s_H-2*s_uu*s_F)*wex[i];
	s_uu=-s_uu/LD;
	dfi[j+(numl*4*desp)]=0.0+r_dH_u*(-nuxB[i]);
	dfi[numl+j+(numl*4*desp)]=0.0+r_dH_u*auxCte;
	dfi[2*numl+j+(numl*4*desp)]=0.0+(r_dH_u*s_uu); 											
	dfi[3*numl+j+(numl*4*desp)]=0.0+(-2.0*r_dF_u);						

	dshi[j+(numl*4*desp)]= 0.0+(r_dF_u)*(-nuxB[i]);						
	dshi[numl+j+(numl*4*desp)]=0.0+(r_dF_u*auxCte);
	dshi[2*numl+j+(numl*4*desp)]=0.0+(r_dF_u*s_uu); 											
	dshi[3*numl+j+(numl*4*desp)]=0.0+(r_dH_u/2.0);

	uu[j]=s_uu;
}



__device__ void Resetear_Valores_Intermedios(int nlambda,ProfilesMemory * pM){
		
	//memset(pM->d_ei , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_eq , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_ev , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_eu , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_rq , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_ru , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->d_rv , 0, (nlambda*7)*sizeof(REAL));
	//memset(pM->dfi , 0,  (nlambda*4*3)*sizeof(REAL));
	//memset(pM->dshi , 0, (nlambda*4*3)*sizeof(REAL));
	
}



__device__  int me_der(const Cuantic * __restrict__ cuantic,Init_Model *initModel,const PRECISION * __restrict__ wlines,const int nlambda,REAL *d_spectra,REAL *spectra, REAL * spectra_slight, REAL ah,const REAL * __restrict__ slight,int filter,ProfilesMemory * pM, const int * __restrict__ fix, REAL cosi, REAL sinis,REAL sina, REAL cosa,REAL sinda,REAL  cosda,REAL  sindi,REAL cosdi,REAL cosis_2,int * uuGlobal, int * FGlobal,int * HGlobal)
{

	
	int il,i,j;

	//MF=initModel->B;
	//DEFINO UN VECTOR DE DERIVADAS
	//POR ORDEN SERAN param=[eta0,magnet,vlos,landadopp,aa,gamma,azi]	
	
	//Resetear_Valores_Intermedios(nlambda,pM);
    
		//Line strength
	//E0=initModel->eta0*cuantic[0].FO; //y sino se definio Fo que debe de pasar 0 o 1 ...??
	
	//central component		
	REAL auxCte = __fdividef((-wlines[0+1]),(VLIGHT*initModel->dopp));
	/*for(j=0;j<nlambda;j++){
		pM->auxCte[j]=(-wlines[0+1])/(VLIGHT*initModel->dopp);
	}*/

	funcionComponentFor(cuantic[0].N_PI,wlines[0+1],nlambda,cuantic[0].WEP,pM->nupB,pM->dfi,pM->dshi,initModel->dopp,initModel->aa,0,pM,auxCte,uuGlobal,FGlobal,HGlobal);

	//blue component
	funcionComponentFor(cuantic[0].N_SIG,wlines[0+1],nlambda,cuantic[0].WEB,pM->nubB,pM->dfi,pM->dshi,initModel->dopp,initModel->aa,1,pM,auxCte,uuGlobal,FGlobal,HGlobal);

	//red component
	funcionComponentFor(cuantic[0].N_SIG,wlines[0+1],nlambda,cuantic[0].WER,pM->nurB,pM->dfi,pM->dshi,initModel->dopp,initModel->aa,2,pM,auxCte,uuGlobal,FGlobal,HGlobal);

	/*printf("\nDFI: \n");
	for(i=0;i<nlambda*4;i++){
		printf("%f\t",pM->dfi[i]);
	}
	printf("\n");
	printf("\nDSHI: \n");
	for(i=0;i<nlambda*4;i++){
		printf("%f\t",pM->dshi[i]);
	}
	printf("\n");*/
	// call here kernel 

	//me_der_kernel<<<1,nlambda,((nlambda*7*7)*sizeof(REAL))+((nlambda * 4 * 3*2)*sizeof(REAL))>>>(initModel->S1,nlambda,d_spectra,spectra, ah, pM, initModel->eta0*cuantic[0].FO);
	//me_der_kernel<<<1,nlambda,(nlambda*7*7)*sizeof(REAL)>>>(initModel->S1,nlambda,d_spectra, ah, pM, initModel->eta0*cuantic[0].FO, cosi, sinis, sina, cosa, sinda, cosda, sindi, cosdi,cosis_2);
	REAL E0_2;
	REAL cosi_2_E0;
	REAL sindi_cosa,sindi_sina,cosdi_E0_2,cosis_2_E0_2,sinis_E0_2;
	REAL E0 = initModel->eta0*cuantic[0].FO;
	REAL sinis_cosa=__fdividef((E0*(sinis*cosa)),2);
	REAL sinis_sina=__fdividef((E0*(sinis*sina)),2);
	E0_2=__fdividef(E0,2.0);
	sindi_cosa=sindi*cosa;
	sindi_sina=sindi*sina;	
	cosdi_E0_2=(E0_2)*cosdi;
	cosi_2_E0=__fdividef((E0*cosi),2.0);
	cosis_2_E0_2=cosis_2*E0_2;
	sinis_E0_2=sinis*E0_2;

	me_der_kernel<<<1,nlambda,(nlambda*7*7)*sizeof(REAL)>>>(initModel->S1,nlambda,d_spectra, ah, pM, initModel->eta0*cuantic[0].FO, cosi, sinis, sina, cosa, sinda, cosda, sindi, cosdi,cosis_2,E0_2,cosi_2_E0,sinis_cosa,sinis_sina,sindi_cosa,sindi_sina,cosdi_E0_2,cosis_2_E0_2,sinis_E0_2);
	cudaDeviceSynchronize();


   //MACROTURBULENCIA
                
   int macApplied = 0;
   if(initModel->mac > 0.0001){
		
	   macApplied = 1;	
	   fgauss(initModel->mac,nlambda,wlines[1],0,pM);  // Gauss Function stored in global variable GMAC 

	   // VARIABLES USED TO CALCULATE DERIVATE OF G1
	   PRECISION ild = (wlines[1] * initModel->mac) / 2.99792458e5; //Sigma
	   for(i=0;i<nlambda;i++){
		   //pM->GMAC_DERIV[i] = (pM->GMAC[i] / initModel->mac * ((((lambda[i] - centro) / ild) * ((lambda[i] - centro) / ild)) - 1.0));
		   //pM->GMAC_DERIV[i] = (pM->GMAC[i] / initModel->mac * ((((d_lambda_const[i] - centro) / ild) * ((d_lambda_const[i] - centro) / ild)) - 1.0));
		   pM->GMAC_DERIV[i] = (pM->GMAC[i] / initModel->mac * ((((d_lambda_const[i] - (d_lambda_const[(int)nlambda/2])) / ild) * ((d_lambda_const[i] - (d_lambda_const[(int)nlambda/2])) / ild)) - 1.0));
		   
	   }
	   if(filter){
		   // convolve both gaussians and use this to convolve this with spectra 
		   direct_convolution_double(pM->GMAC_DERIV, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
		   
		   
		   /*d_direct_convolution_double<<<1,nlambda>>>(pM->GMAC_DERIV, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
		   cudaDeviceSynchronize();*/
		   direct_convolution_double(pM->GMAC, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
		   
		   /*d_direct_convolution_double<<<1,nlambda>>>(pM->GMAC, nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
		   cudaDeviceSynchronize();*/
	   }
	   #pragma unroll
	   for(il=0;il<4;il++){
		   //convCircular(spectra+nlambda*il, pM->GMAC_DERIV, nlambda,d_spectra+(9*nlambda)+(nlambda*NTERMS*il),pM); 
		   d_convCircular<<<1,nlambda,nlambda*sizeof(REAL)+nlambda*sizeof(double)>>>(spectra+nlambda*il, pM->GMAC_DERIV, nlambda,d_spectra+(9*nlambda)+(nlambda*NTERMS*il)); 
		   cudaDeviceSynchronize();
	   }

	   for (j = 0; j < NPARMS; j++)
	   {
			
			for (i = 0; i < 9; i++)
			{
				if (i != 7)																															 //no convolucionamos S0
					//convCircular(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), pM->GMAC, nlambda,d_spectra + (nlambda * i) + (nlambda * NTERMS * j),pM); 
					d_convCircular<<<1,nlambda,nlambda*sizeof(REAL)+nlambda*sizeof(double)>>>(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), pM->GMAC, nlambda,d_spectra + (nlambda * i) + (nlambda * NTERMS * j)); 
					cudaDeviceSynchronize();
			}
	   }


  
   }//end if(MC > 0.0001)


   // stray light factor 


   if(!macApplied && filter){
	   int h;

	   REAL Ic;
				   
	   for (i = 0; i < NTERMS; i++)
	   {
		   // invert continuous
		   if (i != 7){	
			   if(d_spectra[(nlambda * i)]>d_spectra[(nlambda * i) + (nlambda - 1)])
				   Ic = d_spectra[(nlambda * i)];	
			   else
				   Ic = d_spectra[(nlambda * i) + (nlambda - 1)];
			   /*for(h=0;h<nlambda;h++){
				   d_spectra[(nlambda * i) + h] = Ic - d_spectra[(nlambda * i) +h];
			   }*/
			   direct_convolution_ic(d_spectra + (nlambda * i), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic);
			   //direct_convolution_ic4(d_spectra + (nlambda * i), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic);
			   //direct_convolution_ic2(d_spectra + (nlambda * i), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar,Ic);
			  //d_direct_convolution<<<1,nlambda>>>(d_spectra + (nlambda * i), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar); 
			   
			   //d_direct_convolution_ic<<<1,nlambda,nlambda*sizeof(double)>>>(d_spectra + (nlambda * i), d_psfFunction_const, nlambda,Ic); 
			   
			   /*for(h=0;h<nlambda;h++){
				   d_spectra[(nlambda * i) +h] = Ic - d_spectra[(nlambda * i) + h];
			   }*/
		   }	
		   
	   }
	   
	   
	   for (j = 1; j < NPARMS; j++)
	   {
		   
		   for (i = 0; i < NTERMS; i++)
		   {
			   if (i != 7)																															 //no convolucionamos S0
				   direct_convolution(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
				   //direct_convolution2(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
				   //direct_convolution3(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), nlambda, d_psfFunction_const, nlambda,pM->dirConvPar);
				   //d_direct_convolution<<<1,nlambda,nlambda*sizeof(double)>>>(d_spectra + (nlambda * i) + (nlambda * NTERMS * j), d_psfFunction_const, nlambda);
				   //cudaDeviceSynchronize();		   
		   }
	   }
	   //cudaDeviceSynchronize();
	   //

   }

   if(slight!=NULL){
	   // Response Functions 
	  int par;
	  for(par=0;par<NPARMS;par++){
		   for(il=0;il<NTERMS;il++){
			   for(i=0;i<nlambda;i++){
				   d_spectra[(nlambda*il+nlambda*NTERMS*par)+i]=d_spectra[(nlambda*il+nlambda*NTERMS*par)+i]*initModel->alfa;
				   if(NTERMS==11){
					   if(il==10){ //Magnetic filling factor Response function
						   d_spectra[(nlambda*il+nlambda*NTERMS*par)+i]=spectra_slight[nlambda*par+i]-slight[nlambda*par+i];
					   }
				   }
				   else{
					   if(fix[9]){ // if there is mac 
						   if(il==10){ //Magnetic filling factor Response function
							   d_spectra[nlambda*il+nlambda*NTERMS*par+i]=spectra_slight[nlambda*par+i]-slight[nlambda*par+i];
						   }
					   }
					   else{
						   if(il==9){ //Magnetic filling factor Response function
							   d_spectra[nlambda*il+nlambda*NTERMS*par+i]=spectra_slight[nlambda*par+i]-slight[nlambda*par+i];
						   }
					   }
				   }
			   }
		   }
	   }
   }

	
	
}


__global__ void me_der_kernel(PRECISION S1,int nlambda,REAL *d_spectra, REAL ah, ProfilesMemory * pM,PRECISION E0,REAL cosi,REAL sinis,REAL sina, REAL cosa,REAL sinda,REAL  cosda,REAL  sindi,REAL cosdi,REAL cosis_2,REAL E0_2,REAL cosi_2_E0,REAL sinis_cosa,REAL sinis_sina,REAL sindi_cosa,REAL sindi_sina,REAL cosdi_E0_2,REAL cosis_2_E0_2,REAL sinis_E0_2){


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int j,il;
	extern __shared__ REAL d_ei[];
	REAL * d_eq = (REAL *)&d_ei[nlambda*7];
	REAL * d_eu = (REAL *)&d_eq[nlambda*7];
	REAL * d_ev = (REAL *)&d_eu[nlambda*7];
	REAL * d_rq = (REAL *)&d_ev[nlambda*7];
	REAL * d_ru = (REAL *)&d_rq[nlambda*7];
	REAL * d_rv = (REAL *)&d_ru[nlambda*7];

	//REAL * dfi =  (REAL *)&d_rv[nlambda*4*3];
	//REAL * dshi = (REAL *)&dfi[nlambda*4*3];
	
	#pragma unroll
	for(il=0;il<7;il++){
		d_ei[i+nlambda*il] =0;
		d_eq[i+nlambda*il] =0;
		d_eu[i+nlambda*il] =0;
		d_ev[i+nlambda*il] =0;
		d_rq[i+nlambda*il] =0;
		d_ru[i+nlambda*il] =0;
		d_rv[i+nlambda*il] =0;
	}

	/*#pragma unroll
	for(il=0;il<(12);il++){
		dfi[il+ i*(12)] = pM->dfi[il+ i*(12)];
		//dshi[il+ i*(12)] = pM->dshi[il+ i*(12)];
	}*/
	__syncthreads();

	//etai[i] = pM->etai[i]
	
	REAL etain = pM->etain[i];

	REAL etaq = pM->etaq[i];
	REAL etav = pM->etav[i];
	REAL rhou = pM->rhou[i];
	REAL etau = pM->etau[i];
	REAL rhov = pM->rhov[i];
	REAL rhoq = pM->rhoq[i];
	
	d_ei[i]=etain/E0;
	
	/*d_eq[i]=pM->etaqn[i]/E0;
	d_eu[i]=pM->etaun[i]/E0;
	d_ev[i]=pM->etavn[i]/E0;
	d_rq[i]=pM->rhoqn[i]/E0;
	d_ru[i]=pM->rhoun[i]/E0;
	d_rv[i]=pM->rhovn[i]/E0;*/
	
	
	d_eq[i]=__fdividef(etaq,E0);
	d_eu[i]=__fdividef(etau,E0);
	d_ev[i]=__fdividef(etav,E0);
	d_rq[i]=__fdividef(rhoq,E0);
	d_ru[i]=__fdividef(rhou,E0);
	d_rv[i]=__fdividef(rhov,E0);

	//dispersion profiles
	/*REAL E0_2;
	REAL cosi_2_E0;
	REAL sinis_cosa=__fdividef((E0*(sinis*cosa)),2);
	REAL sinis_sina=__fdividef((E0*(sinis*sina)),2);
	E0_2=__fdividef(E0,2.0);
	REAL sindi_cosa,sindi_sina,cosdi_E0_2,cosis_2_E0_2,sinis_E0_2;
	sindi_cosa=sindi*cosa;
	sindi_sina=sindi*sina;	
	cosdi_E0_2=(E0_2)*cosdi;
	cosi_2_E0=__fdividef((E0*cosi),2.0);
	cosis_2_E0_2=cosis_2*E0_2;
	sinis_E0_2=sinis*E0_2;*/
	
	//printf("\n sinis_cosa %f sinis_sina %f E0_2 %f sindi_cosa %f sindi_sina %f cosdi_E0_2 %f cosi_2_E0 %f cosis_2_E0_2 %f sinis_E0_2 %f \n",sinis_cosa,sinis_sina,E0_2,sindi_cosa,sindi_sina,cosdi_E0_2,cosi_2_E0,cosis_2_E0_2,sinis_E0_2);
	
	for(j=1;j<5;j++){
		//derivadas de los perfiles de dispersion respecto de B,VL,LDOPP,A 
		
		REAL dfisum,aux;
		dfisum=	pM->dfi[i + (j-1)*nlambda+ (nlambda*4)]+pM->dfi[i + (j-1)*nlambda + (nlambda*4*2)];
		d_ei[j*nlambda+i] = d_ei[j*nlambda+i] + (pM->dfi[i+ (j-1)*nlambda] * sinis_E0_2 + dfisum * cosis_2_E0_2);
		aux=pM->dfi[(j-1)*nlambda+i]-__fdividef(dfisum,2);
		d_eq[j*nlambda+i]=d_eq[j*nlambda+i]+(aux)*sinis_cosa;
		d_eu[j*nlambda+i]=d_eu[j*nlambda+i]+(aux)*sinis_sina;
		d_ev[j*nlambda+i]= d_ev[j*nlambda+i] +(pM->dfi[(j-1)*nlambda+i+(nlambda*4*2)]-pM->dfi[(j-1)*nlambda+i+(nlambda*4)])*cosi_2_E0;

		/*dfisum=	dfi[i + (j-1)*nlambda+ (nlambda*4)]+dfi[i + (j-1)*nlambda + (nlambda*4*2)];
		d_ei[j*nlambda+i] = d_ei[j*nlambda+i] + (dfi[i+ (j-1)*nlambda] * sinis_E0_2 + dfisum * cosis_2_E0_2);
		aux=__fdividef((dfi[(j-1)*nlambda+i]-dfisum),2);
		d_eq[j*nlambda+i]=d_eq[j*nlambda+i]+(aux)*sinis_cosa;
		d_eu[j*nlambda+i]=d_eu[j*nlambda+i]+(aux)*sinis_sina;
		d_ev[j*nlambda+i]= d_ev[j*nlambda+i] +(dfi[(j-1)*nlambda+i+(nlambda*4*2)]-dfi[(j-1)*nlambda+i+(nlambda*4)])*cosi_2_E0;*/
		
	}
	
	
	for(j=1;j<5;j++){
		REAL r_dshi_1 = pM->dshi[(j-1)*nlambda+i+(nlambda*4)];
		REAL r_dshi_2= pM->dshi[(j-1)*nlambda+i+(nlambda*4*2)];
		//REAL aux=__fdividef((pM->dshi[(j-1)*nlambda+i]-(r_dshi_1+r_dshi_2)),2.0);
		REAL aux=pM->dshi[(j-1)*nlambda+i]-__fdividef((r_dshi_1+r_dshi_2),2.0);
		//REAL aux=dshi[(j-1)*numl+i]-(dshi[(j-1)*numl+i+(numl*4)]+dshi[(j-1)*numl+i+(numl*4*2)])/2.0;
		/*REAL r_dshi_1 = dshi[(j-1)*nlambda+i+(nlambda*4)];
		REAL r_dshi_2= dshi[(j-1)*nlambda+i+(nlambda*4*2)];
		REAL aux=__fdividef((dshi[(j-1)*nlambda+i]-(r_dshi_1+r_dshi_2)),2.0);		*/

		d_rq[j*nlambda+i]=d_rq[j*nlambda+i]+(aux)*sinis_cosa;
		d_ru[j*nlambda+i]=d_ru[j*nlambda+i]+(aux)*sinis_sina;
		d_rv[j*nlambda+i]=d_rv[j*nlambda+i]+((r_dshi_2-r_dshi_1))*cosi_2_E0;
	}
	
	
	//derivadas de los perfiles de dispersion respecto de GAMMA
	REAL cosi_cosdi,sindi_E0_2;
	//cosi_cosdi=pM->cosi*pM->cosdi*E0_2;
	cosi_cosdi=cosi*cosdi*E0_2;
	sindi_E0_2=sindi*E0_2;
	REAL r_parcial2 = pM->parcial2[i];
	REAL r_parcial3 = pM->parcial3[i];

	d_ei[5*nlambda+i]=d_ei[5*nlambda+i]+pM->fi_p[i]*sindi_E0_2+(pM->parcial1[i])*cosi_cosdi;
	d_eq[5*nlambda+i]=d_eq[5*nlambda+i]+r_parcial2*sindi_cosa;
	d_eu[5*nlambda+i]=d_eu[5*nlambda+i]+r_parcial2*sindi_sina;
	d_ev[5*nlambda+i]=d_ev[5*nlambda+i]+(pM->fi_r[i]-pM->fi_b[i])*cosdi_E0_2;
	d_rq[5*nlambda+i]=d_rq[5*nlambda+i]+r_parcial3*sindi_cosa;
	d_ru[5*nlambda+i]=d_ru[5*nlambda+i]+r_parcial3*sindi_sina;
	d_rv[5*nlambda+i]=d_rv[5*nlambda+i]+(pM->shi_r[i]-pM->shi_b[i])*cosdi_E0_2;
	
	
	//derivadas de los perfiles de dispersion respecto de AZI
	REAL sinis_cosda,sinis_sinda;
	/*sinis_cosda=pM->sinis*pM->cosda;
	sinis_sinda=pM->sinis*pM->sinda;		*/
	sinis_cosda=sinis*cosda;
	sinis_sinda=sinis*sinda;			
				
	d_eq[6*nlambda+i]=d_eq[6*nlambda+i]+r_parcial2*sinis_cosda;
	d_eu[6*nlambda+i]=d_eu[6*nlambda+i]+r_parcial2*sinis_sinda;
	d_rq[6*nlambda+i]=d_rq[6*nlambda+i]+r_parcial3*sinis_cosda;
	d_ru[6*nlambda+i]=d_ru[6*nlambda+i]+r_parcial3*sinis_sinda;

	
	//bucle para derivadas de I,Q,U,V 
	//derivada de spectra respecto  E0,MF,VL,LD,A,gamma,azi	
	//REAL r_etai = pM->etai[i];
	REAL r_etai = etain + 1.0;
	REAL r_dt = pM->dt[i];
	//pM->dtaux[i]=(-((S1)* ah))/(pM->dt[i]*pM->dt[i]);
	REAL dtaux =__fdividef((-((S1)* ah)),(r_dt*r_dt));
	
	//pM->etai_gp3[i]=pM->etai[i]*pM->gp3[i];
	REAL etai_gp3 = r_etai*pM->gp3[i];
	REAL gp4_gp2_rhoq = pM->gp4_gp2_rhoq[i];
	REAL gp5_gp2_rhou = pM->gp5_gp2_rhou[i];
	REAL gp6_gp2_rhov = pM->gp6_gp2_rhov[i];

	REAL aux=2*r_etai;
	//pM->ext1[i]=aux*pM->etaq[i]+pM->etav[i]*pM->rhou[i]-pM->etau[i]*pM->rhov[i];
	//pM->ext2[i]=aux*pM->etau[i]+pM->etaq[i]*pM->rhov[i]-pM->etav[i]*pM->rhoq[i];
	//pM->ext3[i]=aux*pM->etav[i]+pM->etau[i]*pM->rhoq[i]-pM->etaq[i]*pM->rhou[i];
	//pM->ext4[i]=aux*pM->gp1[i];


	REAL ext1 = aux*etaq+etav*rhou-etau*rhov;
	REAL ext2 = aux*etau+etaq*rhov-etav*rhoq;
	REAL ext3 = aux*etav+etau*rhoq-etaq*rhou;
	REAL ext4 = aux*pM->gp1[i];
	REAL dgp1, dgp2, dgp3, dgp4,dgp5,dgp6, d_dt;
	REAL r_gp2 = pM->gp2[i];
	//REAL r_etai_2 = pM->etai_2[i];
	REAL r_etai_2 = r_etai*r_etai;
	REAL r_gp3 = pM->gp3[i];

	#pragma unroll
	for(il=0;il<7;il++){
		dgp1=2.0*(r_etai*d_ei[i+nlambda*il]-etaq*d_eq[i+nlambda*il]-etau*d_eu[i+nlambda*il]-etav*d_ev[i+nlambda*il]+rhoq*d_rq[i+nlambda*il]+rhou*d_ru[i+nlambda*il]+rhov*d_rv[i+nlambda*il]);
		dgp2=rhoq*d_eq[i+nlambda*il]+etaq*d_rq[i+nlambda*il]+rhou*d_eu[i+nlambda*il]+etau*d_ru[i+nlambda*il]+rhov*d_ev[i+nlambda*il]+etav*d_rv[i+nlambda*il];
		d_dt=ext4*d_ei[i+nlambda*il]+r_etai_2*dgp1-2*r_gp2*dgp2;
		dgp3=2.0*(r_etai*d_ei[i+nlambda*il]+rhoq*d_rq[i+nlambda*il]+rhou*d_ru[i+nlambda*il]+rhov*d_rv[i+nlambda*il]);
		dgp4=d_ei[i+nlambda*il]*(ext1)+r_etai_2*d_eq[i+nlambda*il]+r_etai*(rhou*d_ev[i+nlambda*il]+etav*d_ru[i+nlambda*il]-rhov*d_eu[i+nlambda*il]-etau*d_rv[i+nlambda*il]);
		dgp5=d_ei[i+nlambda*il]*(ext2)+r_etai_2*d_eu[i+nlambda*il]+r_etai*(rhov*d_eq[i+nlambda*il]+etaq*d_rv[i+nlambda*il]-rhoq*d_ev[i+nlambda*il]-etav*d_rq[i+nlambda*il]);
		dgp6=d_ei[i+nlambda*il]*(ext3)+r_etai_2*d_ev[i+nlambda*il]+r_etai*(rhoq*d_eu[i+nlambda*il]+etau*d_rq[i+nlambda*il]-rhou*d_eq[i+nlambda*il]-etaq*d_ru[i+nlambda*il]);
		d_spectra[i+nlambda*il]=-(((d_ei[i+nlambda*il]*r_gp3+r_etai*dgp3)*r_dt-d_dt*etai_gp3)*(dtaux));
		d_spectra[i+nlambda*il+nlambda*NTERMS]=((dgp4+d_rq[i+nlambda*il]*r_gp2+rhoq*dgp2)*r_dt-d_dt*(gp4_gp2_rhoq))*(dtaux);
		d_spectra[i+nlambda*il+(nlambda*NTERMS*2)]=((dgp5+d_ru[i+nlambda*il]*r_gp2+rhou*dgp2)*r_dt-d_dt*(gp5_gp2_rhou))*(dtaux);
		d_spectra[i+nlambda*il+(nlambda*NTERMS*3)]=((dgp6+d_rv[i+nlambda*il]*r_gp2+rhov*dgp2)*r_dt-d_dt*(gp6_gp2_rhov))*(dtaux);

		/*d_spectra[i+nlambda*il]=pM->d_spectra_backup[i+nlambda*il]=-(((d_ei[i+nlambda*il]*r_gp3+r_etai*dgp3)*r_dt-d_dt*etai_gp3)*(dtaux));
		d_spectra[i+nlambda*il+nlambda*NTERMS]=pM->d_spectra_backup[i+nlambda*il+nlambda*NTERMS]=((dgp4+d_rq[i+nlambda*il]*r_gp2+rhoq*dgp2)*r_dt-d_dt*(gp4_gp2_rhoq))*(dtaux);
		d_spectra[i+nlambda*il+(nlambda*NTERMS*2)]=pM->d_spectra_backup[i+nlambda*il+(nlambda*NTERMS*2)]=((dgp5+d_ru[i+nlambda*il]*r_gp2+rhou*dgp2)*r_dt-d_dt*(gp5_gp2_rhou))*(dtaux);
		d_spectra[i+nlambda*il+(nlambda*NTERMS*3)]=pM->d_spectra_backup[i+nlambda*il+(nlambda*NTERMS*3)]=((dgp6+d_rv[i+nlambda*il]*r_gp2+rhov*dgp2)*r_dt-d_dt*(gp6_gp2_rhov))*(dtaux);		*/
	}
	
	//LA 7-8 RESPECTO B0 Y B1
	
	
	//pM->dti[i]=-(pM->dti[i]* ah);
	//REAL dti = -(pM->dti[i]* ah);
	REAL dti = -((__fdividef(1.0,r_dt))* ah);

	d_spectra[i+nlambda*8]=-dti*etai_gp3;
	d_spectra[i+nlambda*8+(nlambda*NTERMS)]= dti*(gp4_gp2_rhoq);   		
	d_spectra[i+nlambda*8+(nlambda*NTERMS*2)]= dti*(gp5_gp2_rhou);
	d_spectra[i+nlambda*8+(nlambda*NTERMS*3)]= dti*(gp6_gp2_rhov);
	
	/*d_spectra[i+nlambda*8]=pM->d_spectra_backup[i+nlambda*8]=-dti*etai_gp3;
	d_spectra[i+nlambda*8+(nlambda*NTERMS)] = pM->d_spectra_backup[i+nlambda*8+(nlambda*NTERMS)] = dti*(gp4_gp2_rhoq);   		
	d_spectra[i+nlambda*8+(nlambda*NTERMS*2)] = pM->d_spectra_backup[i+nlambda*8+(nlambda*NTERMS*2)] = dti*(gp5_gp2_rhou);
	d_spectra[i+nlambda*8+(nlambda*NTERMS*3)] = pM->d_spectra_backup[i+nlambda*8+(nlambda*NTERMS*3)] = dti*(gp6_gp2_rhov);	*/
	//S0
	
	d_spectra[i+nlambda*7]=1;
	d_spectra[i+nlambda*7+(nlambda*NTERMS)]=0;
	d_spectra[i+nlambda*7+(nlambda*NTERMS*2)]=0;
	d_spectra[i+nlambda*7+(nlambda*NTERMS*3)]=0;
	
	/*d_spectra[i+nlambda*7]= pM->d_spectra_backup[i+nlambda*7]=1;
	d_spectra[i+nlambda*7+(nlambda*NTERMS)]=pM->d_spectra_backup[i+nlambda*7+(nlambda*NTERMS)]=0;
	d_spectra[i+nlambda*7+(nlambda*NTERMS*2)]=pM->d_spectra_backup[i+nlambda*7+(nlambda*NTERMS*2)]=0;
	d_spectra[i+nlambda*7+(nlambda*NTERMS*3)]=pM->d_spectra_backup[i+nlambda*7+(nlambda*NTERMS*3)]=0;	*/
	
	//azimuth stokes I &V
	
	d_spectra[i+nlambda*6]=0;
	d_spectra[i+nlambda*6+(nlambda*NTERMS*3)]=0;
	
	/*d_spectra[i+nlambda*6]=pM->d_spectra_backup[i+nlambda*6]=0;
	d_spectra[i+nlambda*6+(nlambda*NTERMS*3)]=pM->d_spectra_backup[i+nlambda*6+(nlambda*NTERMS*3)]=0;*/
	
}


