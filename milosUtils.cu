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
		modelout->vlos = model->vlos - delta[2];
	}

	if (d_fix_const[3]) // DOPPLER WIDTH
	{

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

		modelout->vlos = model->vlos - delta[2];
	}

	if (d_fix_const[3]) // DOPPLER WIDTH
	{
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
__device__ void FijaACeroDerivadasNoNecesarias(REAL * __restrict__ d_spectra, const int nlambda,const int nterms)
{

	int In, j,i;
	for (In = 0; In < nterms; In++)
		if (d_fix_const[In] == 0)
			for (j = 0; j < NPARMS; j++)
				for (i = 0; i < nlambda; i++)
					d_spectra[i + nlambda * In + j * nlambda * nterms] = 0;
					
}


/*
	Tamaño de H es 	 NTERMS_11 x NTERMS_11
	Tamaño de beta es 1xNTERMS_11

	return en delta tam 1xNTERMS_11
*/
__device__ int mil_svd(PRECISION * h, PRECISION *beta, PRECISION *delta)
{

	const PRECISION epsilon = 1e-12;
	PRECISION v[NTERMS_11*NTERMS_11], w[NTERMS_11];
	
	int i;
	int j,k; 

	PRECISION aux2[NTERMS_11];
	svdcmp(h,NTERMS_11,NTERMS_11,w,v);
	


	//svdcordic(h,TAMANIO_SVD,TAMANIO_SVD,w,v,NUM_ITER_SVD_CORDIC);
	
	PRECISION sum;
		
	for ( j = 0; j < NTERMS_11; j++){
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS_11; k++){
			sum += beta[k] * v[k*NTERMS_11+j];
		}
		aux2[j] = sum;
	}	
	

	#pragma unroll
	for (i = 0; i < NTERMS_11; i++)
	{
		aux2[i]= aux2[i]*((fabs(w[i]) > epsilon) ? (1/w[i]): 0.0);
	}

	for ( i = 0; i < NTERMS_11; i++){		
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS_11; k++){
			sum += v[i*NTERMS_11+k] * aux2[k];
		}
		delta[i] = sum;

	}

	return 1;
}


__device__ void mil_svdf(float * h, float *beta, float *delta, float * v, float *w)
{

	const PRECISION epsilon = 1e-12;
	int i; 
	int j,k;

	float aux2[NTERMS];
	svdcordicf(h,TAMANIO_SVD,TAMANIO_SVD,w,v,NUM_ITER_SVD_CORDIC);

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


	for ( i = 0; i < NTERMS; i++){		
		sum=0;
		#pragma unroll
		for ( k = 0;  k < NTERMS; k++){
			sum += v[i*NTERMS+k] * aux2[k];
		}
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

__global__ void lm_mils(const float * __restrict__ spectro,Init_Model * vInitModel, float * vChisqrf, REAL *  slight, int * vIter, float * spectra, const int * __restrict__ displsSpectro, const int * __restrict__ sendCountPixels, const int * __restrict__ displsPixels, const int N_RTE_PARALLEL, const int numberStream, const int mapStrayLight)
{

	int indice = threadIdx.x + blockIdx.x * blockDim.x;

	if(indice<N_RTE_PARALLEL){

		const REAL PARBETA_better = 5.0;
		const REAL PARBETA_worst = 10.0;

		int i,j,iter;  //, n_ghots;
		int nfree = (d_nlambda_const * NPARMS) - NTERMS;
		ProfilesMemory * pM = (ProfilesMemory *) malloc(sizeof(ProfilesMemory));
		InitProfilesMemoryFromDevice(d_nlambda_const,pM,d_cuantic_const,NTERMS);
		float v[NTERMS*NTERMS], w[NTERMS];
		REAL covar[NTERMS * NTERMS], beta[NTERMS], delta[NTERMS];
		REAL alpha[NTERMS * NTERMS];
		REAL cosi,sinis, sina, cosa, sinda, cosda, sindi, cosdi,cosis_2;
		int uuGlobal,FGlobal,HGlobal;

		REAL flambda;
		REAL chisqr, ochisqr,chisqr0;

		int clanda, ind;
		for(i=0;i<sendCountPixels[(numberStream*N_RTE_PARALLEL)+indice];i++){
			
			REAL PARBETA_FACTOR = 1.0;
			flambda = d_ilambda_const;
			clanda = 0;
			iter = 0;

			const float * spectroAux = spectro+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			float * spectraAux = spectra+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			float * slight_pixel; 
			if(mapStrayLight){
				slight_pixel = slight+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			}
			else{
				slight_pixel = slight;
			}
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

			mil_sinrf(d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, spectraAux, d_ah_const,slight_pixel,pM->spectra_mac, pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa,&sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);
			me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, pM->spectra_slight, d_ah_const, slight_pixel, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina, cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal,NTERMS);
			FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const,NTERMS);
			covarmf(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM,NTERMS);

			#pragma unroll
			for (j = 0; j < NTERMS * NTERMS; j++){
				covar[j] = alpha[j];
			}

			ochisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
			
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
				mil_svdf(covar, beta, delta,v,w);


				AplicaDeltaf(&initModel, delta, &model);
				check(&model);
				mil_sinrf(d_cuantic_const, &model, d_wlines_const, d_nlambda_const, spectraAux , d_ah_const,slight_pixel,pM->spectra_mac,pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa, &sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);

				chisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
				
				/**************************************************************************/

				//printf("\n CHISQR EN LA ITERACION %d,: %e", iter,chisqr);
				
				/**************************************************************************/
				if ((FABS(((ochisqr)-(chisqr))*100/(chisqr)) < d_toplim_const) || ((chisqr) < 0.0001)) // condition to exit of the loop 
					clanda = 1;		
				if ((chisqr) - (ochisqr) < 0.)
				{

					
					flambda=flambda/(PARBETA_better*PARBETA_FACTOR);
					initModel = model;
					me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, spectraAux, d_ah_const, slight_pixel, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina,cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal,NTERMS);
					FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const,NTERMS);	
					covarmf(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM,NTERMS);
					
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
				}

				if ((flambda > 1e+7) || (flambda < 1e-25))
					clanda=1 ; // condition to exit of the loop 		

				iter++;
				if(d_logclambda_const) PARBETA_FACTOR = log10f(chisqr)/log10f(chisqr0);

			} while (iter < d_miter_const && !clanda);

 
			vChisqrf[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = ochisqr;
			vInitModel[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = initModel;
			vIter[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = iter;

		}
		FreeProfilesMemoryFromDevice(pM,d_cuantic_const);

	}

}


__global__ void lm_mils_11(const float * __restrict__ spectro,Init_Model * vInitModel, float * vChisqrf, REAL *  slight, int * vIter, float * spectra, const int * __restrict__ displsSpectro, const int * __restrict__ sendCountPixels, const int * __restrict__ displsPixels, const int N_RTE_PARALLEL, const int numberStream, const int mapStrayLight)
{

	int indice = threadIdx.x + blockIdx.x * blockDim.x;

	if(indice<N_RTE_PARALLEL){

		const REAL PARBETA_better = 5.0;
		const REAL PARBETA_worst = 10.0;

		int i,j,iter;  //, n_ghots;
		int nfree = (d_nlambda_const * NPARMS) - NTERMS_11;
		ProfilesMemory * pM = (ProfilesMemory *) malloc(sizeof(ProfilesMemory));
		InitProfilesMemoryFromDevice(d_nlambda_const,pM,d_cuantic_const,NTERMS_11);
		PRECISION covar[NTERMS_11 * NTERMS_11], beta[NTERMS_11], delta[NTERMS_11];
		REAL alpha[NTERMS_11 * NTERMS_11];
		REAL cosi,sinis, sina, cosa, sinda, cosda, sindi, cosdi,cosis_2;
		int uuGlobal,FGlobal,HGlobal;

		REAL flambda;
		REAL chisqr, ochisqr,chisqr0;

		int clanda, ind;
		for(i=0;i<sendCountPixels[(numberStream*N_RTE_PARALLEL)+indice];i++){
			
			REAL PARBETA_FACTOR = 1.0;
			flambda = d_ilambda_const;
			clanda = 0;
			iter = 0;

			const float * spectroAux = spectro+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			float * spectraAux = spectra+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			float * slight_pixel; 
			if(mapStrayLight){
				slight_pixel = slight+displsSpectro[(numberStream*N_RTE_PARALLEL)+indice]+(i*d_nlambda_const*NPARMS);
			}
			else{
				slight_pixel = slight;
			}
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

			mil_sinrf(d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, spectraAux, d_ah_const,slight_pixel,pM->spectra_mac, pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa,&sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);
			me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, pM->spectra_slight, d_ah_const, slight_pixel, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina, cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal,NTERMS_11);
			FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const,NTERMS_11);
			covarm(d_weight_const, d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM,NTERMS_11);
			

			#pragma unroll
			for (j = 0; j < NTERMS_11 * NTERMS_11; j++){
				covar[j] = alpha[j];
			}

			ochisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
			
			chisqr0 = ochisqr;
			model = initModel;
			
			do
			{
				// CHANGE VALUES OF DIAGONAL 
				for (j = 0; j < NTERMS_11; j++)
				{
					ind = j * (NTERMS_11 + 1);
					covar[ind] = alpha[ind] * (1.0 + flambda);
				}
				mil_svd(covar, beta, delta);
				


				AplicaDelta(&initModel, delta, &model);
				check(&model);
				mil_sinrf(d_cuantic_const, &model, d_wlines_const, d_nlambda_const, spectraAux , d_ah_const,slight_pixel,pM->spectra_mac,pM->spectra_slight, d_use_convolution_const,pM,&cosi,&sinis,&sina,&cosa, &sinda, &cosda, &sindi, &cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);

				chisqr = fchisqr(spectraAux, d_nlambda_const, spectroAux, d_weight_const, d_sigma_const, nfree);
				
				/**************************************************************************/

				//printf("\n CHISQR EN LA ITERACION %d,: %e", iter,chisqr);
				
				/**************************************************************************/
				if ((FABS(((ochisqr)-(chisqr))*100/(chisqr)) < d_toplim_const) || ((chisqr) < 0.0001)) // condition to exit of the loop 
					clanda = 1;		
				if ((chisqr) - (ochisqr) < 0.)
				{

					
					flambda=flambda/(PARBETA_better*PARBETA_FACTOR);
					initModel = model;
					me_der(&d_cuantic_const, &initModel, d_wlines_const, d_nlambda_const, pM->d_spectra, pM->spectra_mac, spectraAux, d_ah_const, slight_pixel, d_use_convolution_const, pM, d_fix_const,cosi,sinis,sina,cosa,sinda, cosda, sindi, cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal,NTERMS_11);
					FijaACeroDerivadasNoNecesarias(pM->d_spectra, d_nlambda_const,NTERMS_11);	
					covarm(d_weight_const,d_weight_sigma_const, d_sigma_const, spectroAux, d_nlambda_const, spectraAux, pM->d_spectra, beta, alpha,pM,NTERMS_11);
					
					#pragma unroll
					for (j = 0; j < NTERMS_11 * NTERMS_11; j++)
						covar[j] = alpha[j];
					ochisqr = chisqr;
				}
				else
				{
					#pragma unroll
					for (j = 0; j < NTERMS_11 * NTERMS_11; j++)
						covar[j] = alpha[j];
					flambda=flambda*PARBETA_worst*PARBETA_FACTOR;
				}

				if ((flambda > 1e+7) || (flambda < 1e-25))
					clanda=1 ; // condition to exit of the loop 		

				iter++;
				if(d_logclambda_const) PARBETA_FACTOR = log10f(chisqr)/log10f(chisqr0);

			} while (iter < d_miter_const && !clanda);

 
			vChisqrf[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = ochisqr;
			vInitModel[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = initModel;
			vIter[i+displsPixels[(numberStream*N_RTE_PARALLEL)+indice]] = iter;

		}
		FreeProfilesMemoryFromDevice(pM,d_cuantic_const);

	}

}


/**
 * 	@param nlamda Number of nlambdas to register.
 * 
 * */
__device__ void InitProfilesMemoryFromDevice(int numl, ProfilesMemory * pM, const Cuantic   cuantic,const int nterms){

	
	pM->v = (float *) malloc (nterms*nterms*sizeof(float));
	pM->w = (float *) malloc (nterms*sizeof(float));

	/************** FGAUSS *************************************/
	pM->term = (PRECISION *) malloc(numl*sizeof(PRECISION));
 
	/************* ME DER *************************************/
	pM->u = (REAL *) malloc(numl * sizeof(REAL));		
	pM->dtiaux = (REAL *) malloc(numl * sizeof(REAL));
	pM->etai_gp3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext1 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext2 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext3 = (REAL *) malloc(numl * sizeof(REAL));
	pM->ext4 = (REAL *) malloc(numl * sizeof(REAL));
	/**********************************************************/
	pM->AP = (REAL *) malloc(nterms*nterms*NPARMS * sizeof(REAL));
	pM->BT = (REAL *) malloc(NPARMS*nterms * sizeof(REAL));

	/************* funcionComponentFor *************************************/
	pM->auxCte = (REAL *) malloc(numl * sizeof(REAL));	
	/**********************************************************/	
	/********************************************************/
	pM->resultConv = (REAL *) malloc(numl *sizeof(REAL));


	pM->spectra_mac = (REAL *) malloc(numl * NPARMS * sizeof(REAL));
	pM->spectra_slight = (REAL *) malloc(numl * NPARMS * sizeof(REAL));
	pM->d_spectra = (REAL *) malloc(numl * nterms * NPARMS * sizeof(REAL));
	pM->GMAC = (PRECISION *) malloc(numl * sizeof(PRECISION));
	pM->GMAC_DERIV  = (PRECISION *) malloc(numl * sizeof(PRECISION));
	pM->dirConvPar = (PRECISION * )malloc((numl + numl - 1) * sizeof(PRECISION));
	memset(pM->dirConvPar , 0, (numl + numl - 1)*sizeof(PRECISION));
	pM->opa = (REAL *) malloc(numl*sizeof(REAL));

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

	pM->HGlobalInicial = (REAL *)  malloc(  numl * ((int)(cuantic.N_PI + cuantic.N_SIG * 2)) * sizeof(REAL *));
	pM->HGlobal = 0;

	pM->FGlobalInicial = (REAL *) malloc( numl * ((int)(cuantic.N_PI + cuantic.N_SIG * 2)) * sizeof(REAL *));
	pM->FGlobal = 0;

}


__device__ void FreeProfilesMemoryFromDevice(ProfilesMemory * pM,const Cuantic  cuantic){


	/************** FGAUSS *************************************/
	free(pM->term);
	/************* ME DER *************************************/

	free(pM->dtiaux);
	free(pM->u);
	free(pM->etai_gp3);
	free(pM->ext1);
	free(pM->ext2);
	free(pM->ext3);
	free(pM->ext4);	

	free(pM->AP);
	free(pM->BT);

	/************* funcionComponentFor *************************************/
	free(pM->auxCte);
	/**********************************************************/

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

	free(pM->dfi);
	free(pM->dshi);

	free(pM->resultConv);

	free(pM->GMAC);
	free(pM->GMAC_DERIV);
	free(pM->dirConvPar);
	
	free(pM->spectra_mac);
	free(pM->spectra_slight);
	free(pM->d_spectra);
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



	free(pM->uuGlobalInicial);
	free(pM->HGlobalInicial);
	free(pM->FGlobalInicial);

}

