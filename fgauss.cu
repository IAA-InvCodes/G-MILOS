
#include "definesCuda.cuh"

/*
 * 
 * deriv : 1 true, 0 false
 */

//;this function builds a gauss function
//;landa(amstrong) ;Central wavelength
//;eje(amstrong) ;Wavelength axis
//;macro ;Macroturbulence in km/s

extern __constant__ PRECISION d_lambda_const [MAX_LAMBDA];

__device__ void fgauss(const PRECISION  MC, const int  neje, const PRECISION  landa, const int deriv,ProfilesMemory * pM)
{
	//int fgauss(PRECISION MC, PRECISION * eje,int neje,PRECISION landa,int deriv,PRECISION * mtb,int nmtb){

	//PRECISION centro;
	PRECISION ild;
	int i;
	PRECISION cte;
	//centro = d_lambda_const[(int)neje / 2];
	ild = (landa * MC) / 2.99792458e5; //Sigma

	//	printf("ild-> %f  ...\n",ild);

	#pragma unroll
	for (i = 0; i < neje; i++)
	{

		//PRECISION aux = ((d_lambda_const[i] - centro) / ild);
		PRECISION aux = ((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild);
		
		pM->term[i] = ( aux * aux) / 2; //exponent
		//printf("term (%d) %f  ...\n",i,term[i]);
	}
	#pragma unroll
	for (i = 0; i < neje; i++)
	{
		if(pM->term[i]< 1e30)
			pM->GMAC[i] = exp(-pM->term[i]);
		else 
			pM->GMAC[i] = 0;
	}

	cte = 0;
	//normalization
	#pragma unroll
	for (i = 0; i < neje; i++)
	{
		cte += pM->GMAC[i];
	}
	#pragma unroll
	for (i = 0; i < neje; i++)
	{
		pM->GMAC[i] /= cte;
	}

	//In case we need the deriv of f gauss /deriv
	if (deriv == 1)
	{
		for (i = 0; i < neje; i++)
		{
			//mtb2=mtb/macro*(((eje-centro)/ILd)^2d0-1d0)
			//pM->GMAC[i] = pM->GMAC[i] / MC * ((((eje[i] - centro) / ild) * ((eje[i] - centro) / ild)) - 1.0);			
			//pM->GMAC[i] = pM->GMAC[i] / MC * ((((d_lambda_const[i] - centro) / ild) * ((d_lambda_const[i] - centro) / ild)) - 1.0);
			pM->GMAC[i] = pM->GMAC[i] / MC * ((((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild) * ((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild)) - 1.0);
			
		}
	}
	//free(term);
}


/*__global__ void fgauss(PRECISION  MC, int  neje, PRECISION  landa, int deriv,ProfilesMemory * pM)
{
	//int fgauss(PRECISION MC, PRECISION * eje,int neje,PRECISION landa,int deriv,PRECISION * mtb,int nmtb){
	int indice = threadIdx.x + blockIdx.x * blockDim.x;

	PRECISION centro;
	PRECISION ild;
	int i;
	PRECISION cte;
	centro = d_lambda_const[(int)neje / 2];
	ild = (landa * MC) / 2.99792458e5; //Sigma

	//	printf("ild-> %f  ...\n",ild);


	for (i = 0; i < neje; i++)
	{

		PRECISION aux = ((d_lambda_const[i] - centro) / ild);
		pM->term[i] = ( aux * aux) / 2; //exponent
		//printf("term (%d) %f  ...\n",i,term[i]);
	}

	for (i = 0; i < neje; i++)
	{
		if(pM->term[i]< 1e30)
			pM->GMAC[i] = exp(-pM->term[i]);
		else 
			pM->GMAC[i] = 0;
	}

	cte = 0;
	//normalization
	for (i = 0; i < neje; i++)
	{
		cte += pM->GMAC[i];
	}
	for (i = 0; i < neje; i++)
	{
		pM->GMAC[i] /= cte;
	}

	//In case we need the deriv of f gauss /deriv
	if (deriv == 1)
	{
		for (i = 0; i < neje; i++)
		{
			//mtb2=mtb/macro*(((eje-centro)/ILd)^2d0-1d0)
			//pM->GMAC[i] = pM->GMAC[i] / MC * ((((eje[i] - centro) / ild) * ((eje[i] - centro) / ild)) - 1.0);			
			pM->GMAC[i] = pM->GMAC[i] / MC * ((((d_lambda_const[i] - centro) / ild) * ((d_lambda_const[i] - centro) / ild)) - 1.0);
		}
	}
	//free(term);
}*/