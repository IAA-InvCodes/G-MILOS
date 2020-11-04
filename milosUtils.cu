#include "lib.cuh"
#include "definesCuda.cuh"
#include "defines.h"
#include "milosUtils.cuh"
#include "time.h"
#include <complex.h>
#include <math.h>
#include <cuComplex.h>
#include "svdcmp.cuh"



extern __constant__ int d_fix_const[11];
extern __constant__ PRECISION d_lambda_const [MAX_LAMBDA];
//extern __constant__ PRECISION d_lambda_const_wcl  [MAX_LAMBDA];
extern __constant__ REAL d_weight_const [4];
extern __constant__ REAL d_weight_sigma_const [4];
extern __constant__ PRECISION d_wlines_const [2];
extern __constant__ Cuantic d_cuantic_const;
extern __constant__ Init_Model d_initModel_const;
extern __constant__ int d_nlambda_const;
extern __constant__ PRECISION d_toplim_const;
extern __constant__ int d_miter_const;
extern __constant__ REAL d_sigma_const;
extern __constant__ REAL d_ilambda_const;
extern __constant__ int d_use_convolution_const;
extern __constant__ REAL d_ah_const;
extern __constant__ int d_logclambda_const;


__device__ void AplicaDelta(const Init_Model * model, PRECISION *  delta, Init_Model *modelout)
{

	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
	PRECISION aux;
	if (d_fix_const[0])  // ETHA 0 
	{
		modelout->eta0 = model->eta0 - delta[0]; // 0
	}
	if (d_fix_const[1]) // B
	{
		aux= delta[1];
		if (aux < -300) //300
			aux = -300;
		else if (aux > 300)
			aux = 300;
		modelout->B = model->B - aux; //magnetic field
	}
	if (d_fix_const[2]) // VLOS
	{
		/*if (delta[2] > 2)
			delta[2] = 2;

		if (delta[2] < -2)
			delta[2] = -2;		*/
		modelout->vlos = model->vlos - delta[2];
	}

	if (d_fix_const[3]) // DOPPLER WIDTH
	{

		/*if (delta[3] > 1e-2)
			delta[3] = 1e-2;
		else if (delta[3] < -1e-2)
			delta[3] = -1e-2;*/
		modelout->dopp = model->dopp - delta[3];
	}

	if (d_fix_const[4]) // DAMPING 
		modelout->aa = model->aa - delta[4];

	if (d_fix_const[5])  // GAMMA 
	{
		aux = delta[5];
		if (aux < -30) //15
			aux = -30;
		else if (aux > 30)
			aux = 30;

		modelout->gm = model->gm - aux; //5
	}
	if (d_fix_const[6]) // AZIMUTH
	{
		/*if (delta[6] < -15)
			delta[6] = -15;
		else if (delta[6] > 15)
			delta[6] = 15;*/
		aux = delta[6];
		if (aux < -30)
			aux = -30;
		else if (aux > 30)
			aux = 30;

		modelout->az = model->az - aux;
	}
	if (d_fix_const[7])
		modelout->S0 = model->S0 - delta[7];
	if (d_fix_const[8])
		modelout->S1 = model->S1 - delta[8];
	if (d_fix_const[9]){
		modelout->mac = model->mac - delta[9]; //9
	}
	if (d_fix_const[10])
		modelout->alfa = model->alfa - delta[10];
}

__device__ void AplicaDeltaf(const Init_Model * model, float *  delta, Init_Model *modelout)
{

	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
	float aux;
	if (d_fix_const[0])  // ETHA 0 
	{
		modelout->eta0 = model->eta0 - delta[0]; // 0
	}
	if (d_fix_const[1]) // B
	{
		aux= delta[1];
		if (aux < -300) //300
			aux = -300;
		else if (aux > 300)
			aux = 300;
		modelout->B = model->B - aux; //magnetic field
	}
	if (d_fix_const[2]) // VLOS
	{
		/*if (delta[2] > 2)
			delta[2] = 2;

		if (delta[2] < -2)
			delta[2] = -2;		*/
		modelout->vlos = model->vlos - delta[2];
	}

	if (d_fix_const[3]) // DOPPLER WIDTH
	{

		/*if (delta[3] > 1e-2)
			delta[3] = 1e-2;
		else if (delta[3] < -1e-2)
			delta[3] = -1e-2;*/
		modelout->dopp = model->dopp - delta[3];
	}

	if (d_fix_const[4]) // DAMPING 
		modelout->aa = model->aa - delta[4];

	if (d_fix_const[5])  // GAMMA 
	{
		aux = delta[5];
		if (aux < -30) //15
			aux = -30;
		else if (aux > 30)
			aux = 30;

		modelout->gm = model->gm - aux; //5
	}
	if (d_fix_const[6]) // AZIMUTH
	{
		/*if (delta[6] < -15)
			delta[6] = -15;
		else if (delta[6] > 15)
			delta[6] = 15;*/
		aux = delta[6];
		if (aux < -30)
			aux = -30;
		else if (aux > 30)
			aux = 30;
		modelout->az = model->az - aux;
	}
	if (d_fix_const[7])
		modelout->S0 = model->S0 - delta[7];
	if (d_fix_const[8])
		modelout->S1 = model->S1 - delta[8];
	if (d_fix_const[9]){
		modelout->mac = model->mac - delta[9]; //9
	}
	if (d_fix_const[10])
		modelout->alfa = model->alfa - delta[10];
}
__device__ int check(Init_Model *model)
{
	//Magnetic field
	if (model->B < 0)
	{
		model->B = -(model->B);
		model->gm = 180.0 - (model->gm);
	}
	if (model->B > 5000)
		model->B = 5000;

	//Inclination
	if (model->gm < 0)
		model->gm = -(model->gm);
	if (model->gm > 180)
	{
		model->gm = 360.0 - model->gm;
	}

	//azimuth
	if (model->az < 0)
		model->az = 180 + (model->az); //model->az= 180 + (model->az);
	if (model->az > 180)
	{
		model->az = model->az - 180.0;
	}

	//RANGOS
	//Eta0
	if (model->eta0 < 1)
		model->eta0 = 1;
	if (model->eta0 > 2500) //idl 2500
		model->eta0 = 2500;

	//velocity
	if (model->vlos < (-20)) //20
		model->vlos = (-20);
	if (model->vlos > 20)
		model->vlos = 20;

	//doppler width ;Do NOT CHANGE THIS
	if (model->dopp < 0.0001)
		model->dopp = 0.0001;
	if (model->dopp > 0.6) // idl 0.6
		model->dopp = 0.6;

	// damping 
	if (model->aa < 0.0001) // idl 1e-4
		model->aa = 0.0001;
	if (model->aa > 10.0) //10
		model->aa = 10.0;

	//S0
	if (model->S0 < 0.0001)
		model->S0 = 0.0001;
	if (model->S0 > 2.00)
		model->S0 = 2.00;

	//S1
	if (model->S1 < 0.0001)
		model->S1 = 0.0001;
	if (model->S1 > 2.00)
		model->S1 = 2.00;

	//macroturbulence
	if (model->mac < 0)
		model->mac = 0;
	if (model->mac > 4)
		model->mac = 4;
	
	// filling factor 
	if(model->alfa<0)
		model->alfa = 0.0;
	if(model->alfa>1.0)
		model->alfa = 1.0;
	

	return 1;
}

/**
 * Study put as parallel with a kernel 
 * */
__device__ void FijaACeroDerivadasNoNecesarias(REAL * __restrict__ d_spectra, const int nlambda)
{

	int In, j,i;
	for (In = 0; In < NTERMS; In++)
		if (d_fix_const[In] == 0)
			for (j = 0; j < NPARMS; j++)
				for (i = 0; i < nlambda; i++)
					d_spectra[i + nlambda * In + j * nlambda * NTERMS] = 0;
					
}


/*
	Tamaño de H es 	 NTERMS x NTERMS
	Tamaño de beta es 1xNTERMS

	return en delta tam 1xNTERMS
*/
__device__ int mil_svd(PRECISION * h, PRECISION *beta, PRECISION *delta)
{

	const PRECISION epsilon = 1e-12;
	PRECISION v[NTERMS*NTERMS], w[NTERMS];
	
	int i;
	int j,k; 

	PRECISION aux2[NTERMS];
	svdcmp(h,NTERMS,NTERMS,w,v);
	


	//svdcordic(h,TAMANIO_SVD,TAMANIO_SVD,w,v,NUM_ITER_SVD_CORDIC);
	
	PRECISION sum;
		
	for ( j = 0; j < NTERMS; j++){
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS; k++){
			sum += beta[k] * v[k*NTERMS+j];
		}
		aux2[j] = sum;
	}	
	

	#pragma unroll
	for (i = 0; i < NTERMS; i++)
	{
		//aux2[i]= aux2[i]*((fabs(w[i]) > epsilon) ? (1/w[i]): 0.0);
		aux2[i]= aux2[i]*((fabs(w[i]) > epsilon) ? (1/w[i]): 0.0);
	}

	//multmatrix(v, NTERMS, NTERMS, aux2, NTERMS, 1, delta);
	for ( i = 0; i < NTERMS; i++){		
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
			sum += v[i*NTERMS+k] * aux2[k];
		}
//				printf("Sum\n");
		delta[i] = sum;

	}

	return 1;
}


__device__ void mil_svdf(float * h, float *beta, float *delta, float * v, float *w)
{

	const PRECISION epsilon = 1e-12;
	int i; 
	int j,k;

	//double aux2[NTERMS];
	float aux2[NTERMS];
	//svdcmpf(h,NTERMS,NTERMS,w,v);
	svdcordicf(h,TAMANIO_SVD,TAMANIO_SVD,w,v,NUM_ITER_SVD_CORDIC);

	//multmatrixf(beta, 1, NTERMS, v, NTERMS, NTERMS, aux2);	
	float sum;
		
	for ( j = 0; j < NTERMS; j++){
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS; k++){
			sum += beta[k] * v[k*NTERMS+j];
		}
		aux2[j] = sum;
	}

	#pragma unroll
	for (i = 0; i < NTERMS; i++)
	{
		aux2[i]= aux2[i]*((fabsf(w[i]) > epsilon) ? (1/w[i]): 0.0);
	}

	//multmatrixf(v, NTERMS, NTERMS, aux2, NTERMS, 1, delta);
	for ( i = 0; i < NTERMS; i++){		
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
			sum += v[i*NTERMS+k] * aux2[k];
		}
//				printf("Sum\n");
		delta[i] = sum;

	}
}

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
*
*
* @Author: Juan Pedro Cobos Carrascosa (IAA-CSIC)
*		   jpedro@iaa.es
* @Date:  Nov. 2011
*
*/
__host__ __device__ void estimacionesClasicas(const PRECISION  lambda_0, const PRECISION *  lambda, const int  nlambda, const float *  spectro, Init_Model *initModel, const int forInitialUse, const Cuantic *  cuantic)
{

	double x, y, aux, LM_lambda_plus, LM_lambda_minus, Blos, Ic, Vlos;
	double aux_vlos,x_vlos,y_vlos;
	//const float *spectroI, *spectroQ, *spectroU, *spectroV;
	double L, m, gamma, gamma_rad, tan_gamma, C;
	int i;

	const float * spectroI = spectro;
	const float * spectroQ = spectro + nlambda;
	const float * spectroU = spectro + nlambda * 2;
	const float * spectroV = spectro + nlambda * 3;

	//check if there is ghost lambdas in the extrems
	int beginLambda = 0;
	
	int exit=0;
	for(i=0;i<nlambda && !exit;i++){
		if(spectroI[i]<0){
			beginLambda++;
		}
		else
		{
			exit=1;
		}	
	}

	int endLambda = nlambda;
	exit=0;
	for(i=nlambda-1;i>=0 && !exit;i--){
		if(spectroI[i]<0){
			endLambda--;
		}
		else
		{
			exit=1;
		}	
	}


	Ic = spectro[endLambda - 1]; // Continuo ultimo valor de I

	/*double Icmax = spectro[0];
	int index =0;
	for (i = 0; i < nlambda; i++)
	{
		if(spectroI[i]>Ic){
			Icmax = spectroI[i];
			index = i;
		}
	}

	Ic = Icmax;*/

	x = 0;
	y = 0;
	x_vlos = 0;
	y_vlos = 0;
	for (i = beginLambda; i < endLambda-1 ; i++)
	{
		if(spectroI[i]>-1 && spectroV[i]>-1){
			aux = (Ic - (spectroI[i] + spectroV[i]));
			aux_vlos = (Ic - spectroI[i]);
			x += (aux * (lambda[i] - lambda_0));
			x_vlos += (aux_vlos * (lambda[i] - lambda_0));
			y += aux;
			y_vlos += aux_vlos;
		}
	}

	//Para evitar nan
	if (fabs(y) > 1e-15)
		LM_lambda_plus = x / y;
	else
		LM_lambda_plus = 0;

	x = 0;
	y = 0;
	for (i = beginLambda; i < endLambda-1 ; i++)
	{
		if(spectroI[i]>-1 && spectroV[i]>-1){
			aux = (Ic - (spectroI[i] - spectroV[i]));
			x += (aux * (lambda[i] - lambda_0));
			y += aux;
		}
	}

	if (fabs(y) > 1e-15)
		LM_lambda_minus = x / y;
	else
		LM_lambda_minus = 0;

	C = (CTE4_6_13 * (lambda_0*lambda_0) * cuantic->GEFF);
	

	Blos = (1 / C) * ((LM_lambda_plus - LM_lambda_minus) / 2);
	//Vlos = (VLIGHT / (lambda_0)) * ((LM_lambda_plus + LM_lambda_minus) / 2);
	Vlos = (VLIGHT / (lambda_0)) * ((x_vlos/y_vlos) / 2); // for now use the center without spectroV only spectroI 


	//------------------------------------------------------------------------------------------------------------
	// //Para probar fórmulación propuesta por D. Orozco (Junio 2017)
	//La formula es la 2.7 que proviene del paper:
	// Diagnostics for spectropolarimetry and magnetography by Jose Carlos del Toro Iniesta and Valent´ýn Mart´ýnez Pillet
	//el 0.08 Es la anchura de la línea en lugar de la resuloción del etalón.

	//Vlos = ( 2*(VLIGHT)*0.08 / (PI*lambda_0)) * atan((spectroI[0]+spectroI[1]-spectroI[3]-spectroI[4])/(spectroI[0]-spectroI[1]-spectroI[3]+spectroI[4]));

	//------------------------------------------------------------------------------------------------------------

	//inclinacion
	x = 0;
	y = 0;
	for (i = beginLambda; i < endLambda - 1; i++)
	{
		if(spectroQ[i]>-1 && spectroU[i]>-1 && spectroV[i]>-1){
			L = FABS(SQRT(spectroQ[i] * spectroQ[i] + spectroU[i] * spectroU[i]));
			m = fabs((4 * (lambda[i] - lambda_0) * L)); // / (3*C*Blos) ); //2*3*C*Blos mod abril 2016 (en test!)

			x = x + FABS(spectroV[i]) * m;
			y = y + FABS(spectroV[i]) * FABS(spectroV[i]);
		}
	}

	y = y * fabs((3 * C * Blos));

	tan_gamma = fabs(sqrt(x / y));

	gamma_rad = atan(tan_gamma); //gamma en radianes
	gamma = gamma_rad * (180 / PI); //gamma en grados

	if(forInitialUse){
		if(gamma>=85 && gamma <=90){  
			gamma_rad = 85 *(PI/180);
		}
		if(gamma>90 && gamma <=95){ 
			gamma_rad = 95 *(PI/180);
		}
	}
	//correction 
	//we use the sign of Blos to see to correct the quadrant
	if (Blos < 0)
		gamma = (180) - gamma;


	// CALCULATIONS FOR AZIMUTH 
	PRECISION tan2phi, phi;

	double sum_u =0.0, sum_q = 0.0;
	for(i=0;i<nlambda;i++){
		if(spectroU[i]>-1 && spectroQ[i]>-1){
			if( fabs(spectroU[i])>0.0001 || fabs(spectroQ[i])>0.0001  ){
				sum_u += spectroU[i];
				sum_q += spectroQ[i];
			}
		}
	}
	tan2phi = sum_u/sum_q;
	phi = (atan(tan2phi) * 180 / PI) / 2;
	if ( sum_u > 0 && sum_q > 0 )
		phi = phi;
	else if ( sum_u < 0 && sum_q > 0 )
		phi = phi + 180;
	else if ( sum_u< 0 && sum_q < 0 )
		phi = phi + 90;
	else if ( sum_u > 0 && sum_q < 0 )
		phi = phi + 90;
	
	// END CALCULATIONS FOR AZIMUTH 
	
	PRECISION B_aux;
	B_aux = fabs(Blos / cos(gamma_rad)); // 

	
	if (Vlos < (-4))
		Vlos = -4;
	if (Vlos > (4))
		Vlos = 4;


	initModel->B = (B_aux > 4000 ? 4000 : B_aux);
	initModel->vlos = Vlos;
	initModel->gm = gamma;
	initModel->az = phi;

	if(!forInitialUse) // store Blos in SO if we are in non-initialization use
		initModel->S0 = Blos;

	//Liberar memoria del vector de lambda auxiliar
	
}

/*
 *
 * nwlineas :   numero de lineas espectrales
 * wlines :		lineas spectrales
 * lambda :		wavelength axis in angstrom
			longitud nlambda
 * spectra : IQUV por filas, longitud ny=nlambda
 */

__global__ void lm_mils(const float * __restrict__ spectro,Init_Model * vInitModel, float * vChisqrf, const REAL *  slight, int * vIter, float * spectra, const int * __restrict__ displsSpectro, const int * __restrict__ sendCountPixels, const int * __restrict__ displsPixels, const int N_RTE_PARALLEL, const int numberStream)
{

	int indice = threadIdx.x + blockIdx.x * blockDim.x;

	if(indice<N_RTE_PARALLEL){
		//REAL * vSigma = (REAL *) malloc((d_nlambda_const*NPARMS)*sizeof(REAL));
		
		//extern __shared__ float shared[];	

		const REAL PARBETA_better = 5.0;
		const REAL PARBETA_worst = 10.0;

		//int iter;
		int i,j,iter;  //, n_ghots;
		int nfree = (d_nlambda_const * NPARMS) - NTERMS;
		ProfilesMemory * pM = (ProfilesMemory *) malloc(sizeof(ProfilesMemory));
		InitProfilesMemoryFromDevice(d_nlambda_const,pM,d_cuantic_const);
		float v[NTERMS*NTERMS], w[NTERMS];
		REAL covar[NTERMS * NTERMS], beta[NTERMS], delta[NTERMS];
		REAL alpha[NTERMS * NTERMS];
		//PRECISION covar[NTERMS * NTERMS], beta[NTERMS], delta[NTERMS];
		//REAL * d_eq = (REAL *)&d_ei[nlambda*7];
		
		/*REAL * delta = (REAL *) &shared[sizeof(REAL)*blockIdx.x*NTERMS];
		if(threadIdx.x==0){
			printf("\n Estoy en idx bloque %d ", blockIdx.x);
			printf("\nDireccion de memoria %d",delta);
		}*/
		REAL cosi,sinis, sina, cosa, sinda, cosda, sindi, cosdi,cosis_2;
		int uuGlobal,FGlobal,HGlobal;

		/*PRECISION covar[NTERMS * NTERMS];
		PRECISION beta[NTERMS];*/
		//REAL * covar = (REAL *)malloc(NTERMS*NTERMS*sizeof(REAL));
		//REAL beta[NTERMS];
		/*REAL * beta = (REAL *) malloc(NTERMS*sizeof(REAL));
		REAL * alpha = (REAL *)malloc(NTERMS*NTERMS*sizeof(REAL));*/
		//PRECISION * delta = (PRECISION *) malloc(NTERMS*sizeof(PRECISION));
		//REAL * delta = (REAL *) malloc(NTERMS*sizeof(REAL));
		REAL flambda;
		REAL chisqr, ochisqr,chisqr0;
		//REAL chisqr2, ochisqr2;

		/*REAL chisqr0, r_ochisqr, r_chisqr;
		REAL * ochisqr, * chisqr;
		ochisqr = (REAL *) malloc (sizeof(REAL));
		chisqr = (REAL *) malloc (sizeof(REAL));*/

		int clanda, ind;
		//REAL spectroLocal [30*NPARMS];
		//printf("\nHEBRA %d EL send count pixels  %d",indice,sendCountPixels[indice]);
		//float  * spectroInter = (float *) malloc(NPARMS*d_nlambda_const*sizeof(float));
		//float4 * spectraAux2 = (float4 * )malloc(d_nlambda_const*sizeof(float4));
		for(i=0;i<sendCountPixels[(numberStream*N_RTE_PARALLEL)+indice];i++){
			
			REAL PARBETA_FACTOR = 1.0;
			flambda = d_ilambda_const;
			clanda = 0;
			iter = 0;
			//printf("\n EL DEPSLAZAMIENTO EN EL PIXEL %d HEBRA  %d  es %d\n",i,indice, displsSpectro[indice]+(i*d_nlambda_const*NPARMS));
			const float * spectroAux = spectro+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			float * spectraAux = spectra+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			/*for(j=0;j<NPARMS;j++){
				for(h=0;h<d_nlambda_const;h++){
					spectroInter[j+ (h*NPARMS)]  = spectroAux[h+(j*d_nlambda_const)];
				}
			}*/
			/*for(j=0;j<30*NPARMS;j++){
				spectroLocal[j] = spectroAux[j];
				//spectra[i] = spectra_vect[NLAMBDA*NPARMS*indice + i];
			}*/
			//n_ghots=0;
			
			/*for(j=0;j<d_nlambda_const*NPARMS;j++){
				if(spectroAux[j]<-1){ 
					vSigma[j]= 1000000000000000000000.0;
					n_ghots++;
				}
				else{
					vSigma[j] = d_sigma_const;
				}
			}*/

			
			//nfree = nfree - n_ghots;
			//Initial Model
			Init_Model initModel,model;
			initModel=d_initModel_const;
			
			// CLASSICAL ESTIMATES TO GET B, GAMMA
			estimacionesClasicas(d_wlines_const[1], d_lambda_const, d_nlambda_const, spectroAux, &initModel,1,&d_cuantic_const);
			
			if (isnan(initModel.B))
				initModel.B = 1;
			if (isnan(initModel.vlos))
				initModel.vlos = 1e-3;
			if (isnan(initModel.gm))
				initModel.gm = 1;						
			if (isnan(initModel.az))
				initModel.az = 1;		

			mil_sinrf(d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, spectraAux, d_ah_const,slight,pM->spectra_mac, pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa,&sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);
			me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, pM->spectra_slight, d_ah_const, slight, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina, cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal);

			//FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const);
			
			//covarm(d_weight_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);
			//covarm2(d_weight_const, d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);
			covarmf(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);

			/*printf("\n BETA INICIAL: ");
			#pragma unroll
			for (j = 0; j < NTERMS; j++){
				//betad[j] = beta[j];
				printf(" %e ",beta[j]);
			}*/

			//printf("\n\n ALPHA INICIAL: ");
			#pragma unroll
			for (j = 0; j < NTERMS * NTERMS; j++){
				covar[j] = alpha[j];
				//printf(" %e ",alpha[j]);
			}
			//ochisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
			ochisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
			//ochisqr = fchisqr2( spectraAux2, d_nlambda_const, spectroAux2, d_weight_const, d_sigma_const, nfree);
			//ochisqr = fchisqr3( spectraAux2, d_nlambda_const, spectroAux2, d_weight_const, d_sigma_const, nfree);
			
			chisqr0 = ochisqr;
			model = initModel;
			
			do
			{
				// CHANGE VALUES OF DIAGONAL 
				for (j = 0; j < NTERMS; j++)
				{
					ind = j * (NTERMS + 1);
					covar[ind] = alpha[ind] * (1.0 + flambda);
				}
			
				//mil_svd(covar, beta, delta);
				mil_svdf(covar, beta, delta,v,w);
				/*if(iter==0){
					printf("\n AUTOVALORES \n");
					for ( j = 0; j < NTERMS; j++){
						printf("%f \t",w[j]);
					}
					printf("\nAutovectores \n");
					int k;
					for ( j = 0; j < NTERMS; j++){
						for ( k = 0; k < NTERMS; k++){
							printf("%f \t",v[k*NTERMS+j]);
						}
						printf("\n");
					}
				}

				printf("\n Deltas: %f %f %f %f %f %f %f %f %f %f",delta[0],delta[1],delta[2],delta[3],delta[4],delta[5],delta[6],delta[7],delta[8],delta[9]);*/

				//AplicaDelta(&initModel, delta, &model);
				AplicaDeltaf(&initModel, delta, &model);
				check(&model);
				mil_sinrf(d_cuantic_const, &model, d_wlines_const, d_nlambda_const, spectraAux , d_ah_const,slight,pM->spectra_mac,pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa, &sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);
				
				//chisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
				chisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
				//chisqr = fchisqr2(spectraAux2, d_nlambda_const, spectroAux2, d_weight_const, d_sigma_const, nfree);
				//chisqr = fchisqr3(spectraAux2, d_nlambda_const, spectroAux2, d_weight_const, d_sigma_const, nfree);
				
				/**************************************************************************/

				//printf("\n CHISQR EN LA ITERACION %d,: %e", iter,chisqr);
				
				/**************************************************************************/
				if ((FABS(((ochisqr)-(chisqr))*100/(chisqr)) < d_toplim_const) || ((chisqr) < 0.0001)) // condition to exit of the loop 
					clanda = 1;		
				if ((chisqr) - (ochisqr) < 0.)
				{

					
					flambda=flambda/(PARBETA_better*PARBETA_FACTOR);
					initModel = model;
					me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, spectraAux, d_ah_const, slight, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina,cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal);
					//FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const);	
					//covarm(d_weight_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);
					//covarm2(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);
					covarmf(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM);
					
					/*#pragma unroll
					for (j = 0; j < NTERMS; j++)
						betad[j] = beta[j];*/
					
					#pragma unroll
					for (j = 0; j < NTERMS * NTERMS; j++)
						covar[j] = alpha[j];
					ochisqr = chisqr;
				}
				else
				{
					#pragma unroll
					for (j = 0; j < NTERMS * NTERMS; j++)
						covar[j] = alpha[j];
					flambda=flambda*PARBETA_worst*PARBETA_FACTOR;
					//printf("\n%d\t%f\t increases\t______________________________",iter,flambda);
				}

				//if ((flambda > 1e+5) || (flambda < 1e-12)) 
				if ((flambda > 1e+7) || (flambda < 1e-25))
					clanda=1 ; // condition to exit of the loop 		

				iter++;
				if(d_logclambda_const) PARBETA_FACTOR = log10f(chisqr)/log10f(chisqr0);

			} while (iter < d_miter_const && !clanda);

 
			vChisqrf[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = ochisqr;
			vInitModel[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = initModel;
			vIter[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = iter;
			/*printf("\n pixel %d desplazacimiento pixel  %d",indice,i+d_displsPixels[indice]);
			printf("\neta_0               :%lf\n",vInitModel[i+d-displsPixels[indice]].eta0);
			printf("magnetic field [G]  :%lf\n",vInitModel[i+d_displsPixels[indice]].B);
			printf("LOS velocity[km/s]  :%lf\n",vInitModel[i+d_displsPixels[indice]].vlos);
			printf("Doppler width [A]   :%lf\n",vInitModel[i+d_displsPixels[indice]].dopp);
			printf("damping             :%lf\n",vInitModel[i+d_displsPixels[indice]].aa);
			printf("gamma [deg]         :%lf\n",vInitModel[i+d_displsPixels[indice]].gm);
			printf("phi  [deg]          :%lf\n",vInitModel[i+d_displsPixels[indice]].az);
			printf("S_0                 :%lf\n",vInitModel[i+d_displsPixels[indice]].S0);
			printf("S_1                 :%lf\n",vInitModel[i+d_displsPixels[indice]].S1);
			printf("v_mac [km/s]        :%lf\n",vInitModel[i+d_displsPixels[indice]].mac);
			printf("filling factor      :%lf\n",vInitModel[i+d_displsPixels[indice]].alfa);
			printf("# Iterations        :%d\n",vIter[i+displsPixels[indice]]);
			printf("\nchisqr              :%le\n",vChisqrf[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]]);*/
		}
		FreeProfilesMemoryFromDevice(pM,d_cuantic_const);
		//free(spectroInter);
		//free(alpha);
		//free(delta);
		//free(beta);
		
	}

}



/**
 * 	@param nlamda Number of nlambdas to register.
 * 
 * */
__device__ void InitProfilesMemoryFromDevice(int numl, ProfilesMemory * pM, const Cuantic   cuantic){

	
	pM->v = (float *) malloc (NTERMS*NTERMS*sizeof(float));
	pM->w = (float *) malloc (NTERMS*sizeof(float));

	/************** FGAUSS *************************************/
	pM->term = (PRECISION *) malloc(numl*sizeof(PRECISION));
 
	/************* ME DER *************************************/
	pM->u = (REAL *) malloc(numl * sizeof(REAL));		
	pM->dtiaux = (REAL *) malloc(numl * sizeof(REAL));
	//pM->dtaux = (REAL *) malloc(numl * sizeof(REAL));
	pM->etai_gp3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext1 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext2 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext4 = (REAL *) malloc(numl * sizeof(REAL));
	/**********************************************************/
	pM->AP = (REAL *) malloc(NTERMS*NTERMS*NPARMS * sizeof(REAL));
	pM->BT = (REAL *) malloc(NPARMS*NTERMS * sizeof(REAL));

	/************* funcionComponentFor *************************************/
	//pM->dH_u = (REAL *) malloc(numl * sizeof(REAL));		
	//pM->dF_u = (REAL *) malloc(numl * sizeof(REAL));
	pM->auxCte = (REAL *) malloc(numl * sizeof(REAL));	
	/**********************************************************/
	//***** VARIABLES FOR FVOIGT ****************************//
	/*pM->z = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex));
	pM->zden = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex ));
	pM->zdiv = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex ));	*/
	/*pM->z = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex));
	pM->zden = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex ));
	pM->zdiv = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex ));	*/	
	/********************************************************/
	pM->resultConv = (REAL *) malloc(numl *sizeof(REAL));

	//pM->spectra = (REAL *) malloc(numl * NPARMS * sizeof(REAL));
	pM->spectra_mac = (REAL *) malloc(numl * NPARMS * sizeof(REAL));
	pM->spectra_slight = (REAL *) malloc(numl * NPARMS * sizeof(REAL));
	pM->d_spectra = (REAL *) malloc(numl * NTERMS * NPARMS * sizeof(REAL));
	pM->GMAC = (PRECISION *) malloc(numl * sizeof(PRECISION));
	pM->GMAC_DERIV  = (PRECISION *) malloc(numl * sizeof(PRECISION));
	pM->dirConvPar = (PRECISION * )malloc((numl + numl - 1) * sizeof(PRECISION));
	memset(pM->dirConvPar , 0, (numl + numl - 1)*sizeof(PRECISION));
	pM->opa = (REAL *) malloc(numl*sizeof(REAL));
	
	//pM->d_spectra_backup = (REAL *) malloc(numl * NTERMS * NPARMS * sizeof(REAL));

	pM->gp4_gp2_rhoq = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp5_gp2_rhou = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp6_gp2_rhov = (REAL *) malloc(numl * sizeof(REAL));

	pM->gp1 = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp2 = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp4 = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp5 = (REAL *) malloc(numl * sizeof(REAL));
	pM->gp6 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dt = (REAL *) malloc(numl * sizeof(REAL));
	pM->dti = (REAL *) malloc(numl * sizeof(REAL));

	pM->etai_2 = (REAL *) malloc(numl * sizeof(REAL));

	pM->dgp1 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dgp2 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dgp3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dgp4 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dgp5 = (REAL *) malloc(numl * sizeof(REAL));
	pM->dgp6 = (REAL *) malloc(numl * sizeof(REAL));
	pM->d_dt = (REAL *) malloc(numl * sizeof(REAL));

	/*pM->d_ei = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_eq = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_eu = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_ev = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_rq = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_ru = (REAL *) malloc(numl * 7 * sizeof(REAL));
	pM->d_rv = (REAL *) malloc(numl * 7 * sizeof(REAL));*/
	pM->dfi = (REAL *) malloc(numl * 4 * 3 * sizeof(REAL));  //DNULO
	pM->dshi = (REAL *) malloc(numl * 4 * 3 * sizeof(REAL)); //DNULO

	pM->fi_p = (REAL *) malloc(numl  * sizeof(REAL));
	pM->fi_b = (REAL *) malloc(numl  * sizeof(REAL));
	pM->fi_r = (REAL *) malloc(numl  * sizeof(REAL));
	pM->shi_p = (REAL *) malloc(numl  * sizeof(REAL));
	pM->shi_b = (REAL *) malloc(numl  * sizeof(REAL));
	pM->shi_r = (REAL *) malloc(numl  * sizeof(REAL));

	pM->etain = (REAL *) malloc(numl  * sizeof(REAL));
	pM->etaqn = (REAL *) malloc(numl  * sizeof(REAL));
	pM->etaun = (REAL *) malloc(numl  * sizeof(REAL));
	pM->etavn = (REAL *) malloc(numl  * sizeof(REAL));
	pM->rhoqn = (REAL *) malloc(numl  * sizeof(REAL));
	pM->rhoun = (REAL *) malloc(numl  * sizeof(REAL));
	pM->rhovn = (REAL *) malloc(numl  * sizeof(REAL));

	pM->etai = (REAL *) malloc(numl * sizeof(REAL));
	pM->etaq = (REAL *) malloc(numl * sizeof(REAL));
	pM->etau = (REAL *) malloc(numl * sizeof(REAL));
	pM->etav = (REAL *) malloc(numl * sizeof(REAL));
	pM->rhoq = (REAL *) malloc(numl * sizeof(REAL));
	pM->rhou = (REAL *) malloc(numl * sizeof(REAL));
	pM->rhov = (REAL *) malloc(numl * sizeof(REAL));

	pM->parcial1 = (REAL *) malloc(numl * sizeof(REAL));
	pM->parcial2 = (REAL *) malloc(numl * sizeof(REAL));
	pM->parcial3 = (REAL *) malloc(numl * sizeof(REAL));

	pM->nubB = (REAL *) malloc(cuantic.N_SIG * sizeof(REAL));
	pM->nurB = (REAL *) malloc(cuantic.N_SIG * sizeof(REAL));
	pM->nupB = (REAL *) malloc(cuantic.N_PI * sizeof(REAL));

	pM->uuGlobalInicial = (REAL *) malloc( ( numl * ((int)(cuantic.N_PI + cuantic.N_SIG * 2))) * sizeof(REAL) );
	pM->uuGlobal = 0;
	/*int i = 0;
	for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		pM->uuGlobalInicial[i] = (REAL *) malloc(numl * sizeof(REAL));
	}*/

	pM->HGlobalInicial = (REAL *)  malloc(  numl * ((int)(cuantic.N_PI + cuantic.N_SIG * 2)) * sizeof(REAL *));
	pM->HGlobal = 0;
	/*for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		pM->HGlobalInicial[i] = (REAL *) malloc(numl * sizeof(REAL));
	}*/

	pM->FGlobalInicial = (REAL *) malloc( numl * ((int)(cuantic.N_PI + cuantic.N_SIG * 2)) * sizeof(REAL *));
	/*for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		pM->FGlobalInicial[i] = (REAL *) malloc(numl * sizeof(REAL));
	}*/
	pM->FGlobal = 0;

}


__device__ void FreeProfilesMemoryFromDevice(ProfilesMemory * pM,const Cuantic  cuantic){


	/************** FGAUSS *************************************/
	free(pM->term);
	/************* ME DER *************************************/

	free(pM->dtiaux);
	free(pM->u);
	//free(pM->dtaux);
	free(pM->etai_gp3);
	free(pM->ext1);
	free(pM->ext2);
	free(pM->ext3);
	free(pM->ext4);	

	free(pM->AP);
	free(pM->BT);

	/************* funcionComponentFor *************************************/
	//free(pM->dH_u);		
	//free(pM->dF_u);
	free(pM->auxCte);
	/**********************************************************/

	/*free(pM->zden);
	free(pM->zdiv);
	free(pM->z);*/

	free(pM->gp1);
	free(pM->gp2);
	free(pM->gp3);
	free(pM->gp4);
	free(pM->gp5);
	free(pM->gp6);
	free(pM->dt);
	free(pM->dti);

	free(pM->etai_2);

	free(pM->dgp1);
	free(pM->dgp2);
	free(pM->dgp3);
	free(pM->dgp4);
	free(pM->dgp5);
	free(pM->dgp6);
	free(pM->d_dt);

	/*free(pM->d_ei);
	free(pM->d_eq);
	free(pM->d_ev);
	free(pM->d_eu);
	free(pM->d_rq);
	free(pM->d_ru);
	free(pM->d_rv);*/

	free(pM->dfi);
	free(pM->dshi);

	free(pM->resultConv);

	free(pM->GMAC);
	free(pM->GMAC_DERIV);
	free(pM->dirConvPar);
	
	free(pM->spectra_mac);
	free(pM->spectra_slight);
	free(pM->d_spectra);
	//free(pM->d_spectra_backup);
	free(pM->opa);
	

	free(pM->fi_p);
	free(pM->fi_b);
	free(pM->fi_r);
	free(pM->shi_p);
	free(pM->shi_b);
	free(pM->shi_r);

	free(pM->etain);
	free(pM->etaqn);
	free(pM->etaun);
	free(pM->etavn);
	free(pM->rhoqn);
	free(pM->rhoun);
	free(pM->rhovn);

	free(pM->etai);
	free(pM->etaq);
	free(pM->etau);
	free(pM->etav);

	free(pM->rhoq);
	free(pM->rhou);
	free(pM->rhov);

	free(pM->parcial1);
	free(pM->parcial2);
	free(pM->parcial3);

	free(pM->nubB);
	free(pM->nurB);
	free(pM->nupB);

	free(pM->gp4_gp2_rhoq);
	free(pM->gp5_gp2_rhou);
	free(pM->gp6_gp2_rhov);

	/*int i;
	for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		free(pM->uuGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		free(pM->HGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic.N_PI + cuantic.N_SIG * 2); i++)
	{
		free(pM->FGlobalInicial[i]);
	}*/

	free(pM->uuGlobalInicial);
	free(pM->HGlobalInicial);
	free(pM->FGlobalInicial);

}



/**
 * 	@param nlamda Number of nlambdas to register.
 * 
 * */
 __host__ void InitProfilesMemoryFromHost(int numl, ProfilesMemory * pM, Cuantic *cuantic){


	/************** FGAUSS *************************************/
	checkCuda(cudaMalloc(&pM->term, numl * sizeof(PRECISION)));
	/************* ME DER *************************************/
	checkCuda(cudaMalloc(&pM->dtiaux, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->u, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dtaux, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etai_gp3, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->ext1, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->ext2, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->ext3, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->ext4, numl * sizeof(REAL)));
	/**********************************************************/
	checkCuda(cudaMalloc(&pM->AP, NTERMS*NTERMS*NPARMS * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->BT, NPARMS*NTERMS * sizeof(REAL)));

	/************* funcionComponentFor *************************************/
	checkCuda(cudaMalloc(&pM->dH_u, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dF_u, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->auxCte, numl * sizeof(REAL)));
	/**********************************************************/

	//***** VARIABLES FOR FVOIGT ****************************//

	pM->z = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex));
	pM->zden = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex ));
	pM->zdiv = (cuDoubleComplex *) malloc (numl * sizeof(cuDoubleComplex ));	
	/*pM->z = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex));
	pM->zden = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex ));
	pM->zdiv = (cuFloatComplex *) malloc (numl * sizeof(cuFloatComplex ));		*/

	/********************************************************/
	checkCuda(cudaMalloc(&pM->resultConv, numl*sizeof(REAL) ) );
	

	//pM->spectra = malloc(numl * NPARMS * sizeof(REAL));
	checkCuda(cudaMalloc(&pM->spectra_mac, numl * NPARMS * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->spectra_slight, numl * NPARMS * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_spectra, numl * NTERMS * NPARMS * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->GMAC, numl * sizeof(PRECISION)));
	checkCuda(cudaMalloc(&pM->GMAC_DERIV, numl * sizeof(PRECISION)));
	checkCuda(cudaMalloc(&pM->dirConvPar, (numl + numl - 1) * sizeof(PRECISION)));
	checkCuda(cudaMalloc(&pM->opa, (numl) * sizeof(PRECISION)));


	checkCuda(cudaMalloc(&pM->gp4_gp2_rhoq, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp5_gp2_rhou, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp6_gp2_rhov, numl * sizeof(REAL)));


	checkCuda(cudaMalloc(&pM->gp1, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp2, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp3, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp4, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp5, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->gp6, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dt, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dti, numl * sizeof(REAL)));


	checkCuda(cudaMalloc(&pM->etai_2, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp1, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp2, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp3, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp4, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp5, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dgp6, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_dt, numl * sizeof(REAL)));


	checkCuda(cudaMalloc(&pM->d_ei, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_eq, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_eu, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_ev, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_rq, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_ru, numl * 7 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->d_rv, numl * 7 * sizeof(REAL)));

	checkCuda(cudaMalloc(&pM->dfi, numl * 4 * 3 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->dshi, numl * 4 * 3 * sizeof(REAL)));
	
	checkCuda(cudaMalloc(&pM->fi_p, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->fi_b, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->fi_r, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->shi_p, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->shi_b, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->shi_r, numl * 2 * sizeof(REAL)));

	checkCuda(cudaMalloc(&pM->etain, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etaqn, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etaun, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etavn, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhoqn, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhoun, numl * 2 * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhovn, numl * 2 * sizeof(REAL)));


	checkCuda(cudaMalloc(&pM->etai, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etaq, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etau, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->etav, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhoq, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhoq, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhou, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->rhov, numl * sizeof(REAL)));


	checkCuda(cudaMalloc(&pM->parcial1, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->parcial2, numl * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->parcial3, numl * sizeof(REAL)));

	checkCuda(cudaMalloc(&pM->parcial3, cuantic[0].N_SIG * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->parcial3, cuantic[0].N_SIG * sizeof(REAL)));
	checkCuda(cudaMalloc(&pM->parcial3, cuantic[0].N_PI * sizeof(REAL)));

	printf("\n reservada memoria antes de intentar reservar la memoria de cuantic %d \n", ((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2)) );

	checkCuda( cudaMalloc(&pM->uuGlobalInicial,  numl * ((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2)) * sizeof(REAL *) ));
	
	pM->uuGlobal = 0;
	/*int i = 0;
	for (i = 0; i < ((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2)) ; i++)
	{
		checkCuda(cudaMalloc(&pM->uuGlobalInicial[i], numl * sizeof(REAL)));
		printf("\n reservada memoria uuGlobalInicial %d \n",i);
	}*/

	
	checkCuda(cudaMalloc(&pM->HGlobalInicial,  numl * ((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2)) * sizeof(REAL *)  ));
	pM->HGlobal = 0;
	/*for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		checkCuda(cudaMalloc(&pM->HGlobalInicial[i], numl * sizeof(REAL)));
		printf("\n reservada memoria HGlobalInicial %d \n",i);
	}*/

	
	checkCuda(cudaMalloc(&pM->FGlobalInicial, numl * ((int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2)) * sizeof(REAL *)    ));
	
	/*for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		checkCuda(cudaMalloc(&pM->FGlobalInicial[i], numl * sizeof(REAL)));
		printf("\n reservada memoria FGlobalInicial %d \n",i);
	}*/
	pM->FGlobal = 0;


}


__host__ void FreeProfilesMemoryFromHost(ProfilesMemory * pM,Cuantic * cuantic){

	/************** FGAUSS *************************************/
	cudaFree(pM->term);
	/**************************************/
	cudaFree(pM->dtiaux);
	cudaFree(pM->u);
	cudaFree(pM->dtaux);
	cudaFree(pM->etai_gp3);
	cudaFree(pM->ext1);
	cudaFree(pM->ext2);
	cudaFree(pM->ext3);
	cudaFree(pM->ext4);	

	cudaFree(pM->AP);
	cudaFree(pM->BT);	

	cudaFree(pM->dH_u);
	cudaFree(pM->dF_u);
	cudaFree(pM->auxCte);


	cudaFree(pM->zden);
	cudaFree(pM->zdiv);
	cudaFree(pM->z);

	cudaFree(pM->gp1);
	cudaFree(pM->gp2);
	cudaFree(pM->gp3);
	cudaFree(pM->gp4);
	cudaFree(pM->gp5);
	cudaFree(pM->gp6);
	cudaFree(pM->dt);
	cudaFree(pM->dti);

	cudaFree(pM->etai_2);

	cudaFree(pM->dgp1);
	cudaFree(pM->dgp2);
	cudaFree(pM->dgp3);
	cudaFree(pM->dgp4);
	cudaFree(pM->dgp5);
	cudaFree(pM->dgp6);
	cudaFree(pM->d_dt);

	cudaFree(pM->d_ei);
	cudaFree(pM->d_eq);
	cudaFree(pM->d_ev);
	cudaFree(pM->d_eu);
	cudaFree(pM->d_rq);
	cudaFree(pM->d_ru);
	cudaFree(pM->d_rv);

	cudaFree(pM->dfi);
	cudaFree(pM->dshi);

	cudaFree(pM->resultConv);

	cudaFree(pM->GMAC);
	cudaFree(pM->GMAC_DERIV);
	cudaFree(pM->dirConvPar);
	cudaFree(pM->opa);

	cudaFree(pM->spectra_mac);
	cudaFree(pM->spectra_slight);
	cudaFree(pM->d_spectra);

	cudaFree(pM->fi_p);
	cudaFree(pM->fi_b);
	cudaFree(pM->fi_r);
	cudaFree(pM->shi_p);
	cudaFree(pM->shi_b);
	cudaFree(pM->shi_r);

	cudaFree(pM->etain);
	cudaFree(pM->etaqn);
	cudaFree(pM->etaun);
	cudaFree(pM->etavn);
	cudaFree(pM->rhoqn);
	cudaFree(pM->rhoun);
	cudaFree(pM->rhovn);

	cudaFree(pM->etai);
	cudaFree(pM->etaq);
	cudaFree(pM->etau);
	cudaFree(pM->etav);

	cudaFree(pM->rhoq);
	cudaFree(pM->rhou);
	cudaFree(pM->rhov);

	cudaFree(pM->parcial1);
	cudaFree(pM->parcial2);
	cudaFree(pM->parcial3);

	cudaFree(pM->nubB);
	cudaFree(pM->nurB);
	cudaFree(pM->nupB);

	cudaFree(pM->gp4_gp2_rhoq);
	cudaFree(pM->gp5_gp2_rhou);
	cudaFree(pM->gp6_gp2_rhov);

	/*int i;
	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		cudaFree(pM->uuGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		cudaFree(pM->HGlobalInicial[i]);
	}

	for (i = 0; i < (int)(cuantic[0].N_PI + cuantic[0].N_SIG * 2); i++)
	{
		cudaFree(pM->FGlobalInicial[i]);
	}*/

	cudaFree(pM->uuGlobalInicial);
	cudaFree(pM->HGlobalInicial);
	cudaFree(pM->FGlobalInicial);

}




/*__device__ void createMemoryFFTFromDevice(CUFFT_Memory * cu, int  nlambda, int  usePSF){

	cu->inFilterMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
	cu->outFilterMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
	
	cu->inFilterMAC_DERIV = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
	cu->outFilterMAC_DERIV = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

	cu->inSpectraFwMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
	cu->outSpectraFwMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

	cu->inSpectraBwMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
	cu->outSpectraBwMAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

	if(usePSF){
		cu->inSpectraFwPSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->outSpectraFwPSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

		cu->inSpectraBwPSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->outSpectraBwPSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

		cu->inPSF_MAC = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->fftw_G_MAC_PSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

		cu->inMulMacPSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->outConvFilters = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

		cu->inPSF_MAC_DERIV = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->fftw_G_MAC_DERIV_PSF = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));

		cu->inMulMacPSFDeriv = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		cu->outConvFiltersDeriv = (cufftDoubleComplex *) malloc( nlambda * sizeof (cufftDoubleComplex));
		
	}

	//cufftPlan1d(&cu->plan1D, nlambda, CUFFT_Z2Z, 1);
}*/

/*__device__ void  FreeMemoryFFTFromDevice(CUFFT_Memory * cu, int  usePSF){

	free(cu->inFilterMAC);
	free(cu->outFilterMAC);

	free(cu->inFilterMAC_DERIV);
	free(cu->outFilterMAC_DERIV);

	free(cu->inSpectraFwMAC);
	free(cu->outSpectraFwMAC);

	free(cu->inSpectraBwMAC);
	free(cu->outSpectraBwMAC);

	if(usePSF){
		free(cu->inSpectraFwPSF);
		free(cu->outSpectraFwPSF);

		free(cu->inSpectraBwPSF);
		free(cu->outSpectraBwPSF);

		free(cu->inPSF_MAC);
		free(cu->fftw_G_MAC_PSF);

		free(cu->inMulMacPSF);
		free(cu->outConvFilters);

		free(cu->inPSF_MAC_DERIV);
		free(cu->fftw_G_MAC_DERIV_PSF);

		free(cu->inMulMacPSFDeriv);
		free(cu->outConvFiltersDeriv);

	}
	//cufftDestroy(cu->plan1D);
}*/

/*__host__ void createMemoryFFTFromHost(CUFFT_Memory * cu, int  nlambda, int usePSF){

	checkCuda(cudaMalloc(&cu->inFilterMAC, nlambda * sizeof (cufftDoubleComplex)));
	checkCuda(cudaMalloc(&cu->outFilterMAC, nlambda * sizeof (cufftDoubleComplex)));

	checkCuda(cudaMalloc(&cu->inFilterMAC_DERIV, nlambda * sizeof (cufftDoubleComplex)));
	checkCuda(cudaMalloc(&cu->outFilterMAC_DERIV, nlambda * sizeof (cufftDoubleComplex)));
	
	checkCuda(cudaMalloc(&cu->inSpectraFwMAC, nlambda * sizeof (cufftDoubleComplex)));
	checkCuda(cudaMalloc(&cu->outSpectraFwMAC, nlambda * sizeof (cufftDoubleComplex)));

	checkCuda(cudaMalloc(&cu->inSpectraBwMAC, nlambda * sizeof (cufftDoubleComplex)));
	checkCuda(cudaMalloc(&cu->outSpectraBwMAC, nlambda * sizeof (cufftDoubleComplex)));

	if(usePSF){

		checkCuda(cudaMalloc(&cu->inSpectraFwPSF, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->outSpectraFwPSF, nlambda * sizeof (cufftDoubleComplex)));

		checkCuda(cudaMalloc(&cu->inSpectraBwPSF, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->outSpectraBwPSF, nlambda * sizeof (cufftDoubleComplex)));

		checkCuda(cudaMalloc(&cu->inPSF_MAC, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->fftw_G_MAC_PSF, nlambda * sizeof (cufftDoubleComplex)));

		checkCuda(cudaMalloc(&cu->inMulMacPSF, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->outConvFilters, nlambda * sizeof (cufftDoubleComplex)));

		checkCuda(cudaMalloc(&cu->inPSF_MAC_DERIV, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->fftw_G_MAC_DERIV_PSF, nlambda * sizeof (cufftDoubleComplex)));

		checkCuda(cudaMalloc(&cu->inMulMacPSFDeriv, nlambda * sizeof (cufftDoubleComplex)));
		checkCuda(cudaMalloc(&cu->outConvFiltersDeriv, nlambda * sizeof (cufftDoubleComplex)));
		
	}

	cufftPlan1d(&cu->plan1D, nlambda, CUFFT_Z2Z, 1);
}*/

/*__host__ void  FreeMemoryFFTFromHost(CUFFT_Memory * cu, int usePSF){

	cudaFree(cu->inFilterMAC);
	cudaFree(cu->outFilterMAC);

	cudaFree(cu->inFilterMAC_DERIV);
	cudaFree(cu->outFilterMAC_DERIV);

	cudaFree(cu->inSpectraFwMAC);
	cudaFree(cu->outSpectraFwMAC);

	cudaFree(cu->inSpectraBwMAC);
	cudaFree(cu->outSpectraBwMAC);

	if(usePSF){
		cudaFree(cu->inSpectraFwPSF);
		cudaFree(cu->outSpectraFwPSF);

		cudaFree(cu->inSpectraBwPSF);
		cudaFree(cu->outSpectraBwPSF);

		cudaFree(cu->inPSF_MAC);
		cudaFree(cu->fftw_G_MAC_PSF);

		cudaFree(cu->inMulMacPSF);
		cudaFree(cu->outConvFilters);

		cudaFree(cu->inPSF_MAC_DERIV);
		cudaFree(cu->fftw_G_MAC_DERIV_PSF);

		cudaFree(cu->inMulMacPSFDeriv);
		cudaFree(cu->outConvFiltersDeriv);

	}
	cufftDestroy(cu->plan1D);
}*/



/*__global__ void load_sigma(__constant__ REAL * __restrict__ spectro, REAL * __restrict__ vSigma, int * n_ghots){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(spectro[i]<-1){ 
		vSigma[i]= 1000000000000000000000.0;
		n_ghots++;
	}
	else{
		vSigma[i] = d_sigma_const;
	}

}*/