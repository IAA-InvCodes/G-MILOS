
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
	PRECISION ild;
	int i;
	PRECISION cte;
	
	ild = (landa * MC) / 2.99792458e5; //Sigma

	#pragma unroll
	for (i = 0; i < neje; i++)
	{

		PRECISION aux = ((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild);
		
		pM->term[i] = ( aux * aux) / 2; //exponent
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
			pM->GMAC[i] = pM->GMAC[i] / MC * ((((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild) * ((d_lambda_const[i] - (d_lambda_const[(int)neje / 2])) / ild)) - 1.0);
			
		}
	}
}
