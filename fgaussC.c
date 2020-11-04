#include "defines.h"


/*
 * 
 * deriv : 1 true, 0 false
 */

//;this function builds a gauss function
//;landa(amstrong) ;Central wavelength
//;eje(amstrong) ;Wavelength axis
//;macro ;Macroturbulence in km/s

PRECISION * fgauss_WL(PRECISION FWHM, PRECISION step_between_lw, PRECISION lambda0, PRECISION lambdaCentral, int nLambda)
{

	REAL *mtb_final;
	PRECISION *mtb ;
	PRECISION *term, *loai;
	int i;
	int nloai;
	int nmtb;
	PRECISION cte;
	
	PRECISION sigma=FWHM*0.42466090/1000.0; // in Angstroms

	term = (PRECISION *)calloc(nLambda, sizeof(PRECISION));

	for (i = 0; i <nLambda; i++)
	{
		PRECISION lambdaX = lambda0 +i*step_between_lw;
		PRECISION aux = ((lambdaX - lambdaCentral) / sigma);
		term[i] = ( aux * aux) / 2; //exponent
	}

	nloai = 0;
	loai = (PRECISION *) calloc(nLambda, sizeof(PRECISION));
	for (i = 0; i <nLambda; i++)
	{
		if (term[i] < 1e30)
		{
			nloai++;
			loai[i] = 1;
		}
	}

	if (nloai > 0)
	{
		nmtb = nloai;
		mtb = (PRECISION * ) calloc(nmtb, sizeof(PRECISION));
		for (i = 0; i <nLambda; i++)
		{
			if (loai[i])
			{
				mtb[i] = exp(-term[i]);
				//printf("term (%d) %f  ...\n",i,mtb[i]);
			}
		}
	}
	else
	{

		nmtb = nLambda;
		mtb = (PRECISION *) malloc ( (nLambda)* sizeof(PRECISION));
		for (i = 0; i < nLambda; i++)
		{
			mtb[i] = exp(-term[i]);
			//printf("term (%d) %f  ...\n",i,mtb[i]);
		}
	}

	cte = 0;
	//normalization
	for (i = 0; i < nmtb; i++)
	{
		cte += mtb[i];
	}
	for (i = 0; i < nLambda; i++)
	{
		mtb[i] /= cte;
	}

	free(loai);
	free(term);

	return mtb;
}
