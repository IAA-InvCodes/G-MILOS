
#include "defines.h"

int Cuanten(Cuantic *cuantic, PRECISION sl, PRECISION ll, PRECISION jl, PRECISION su, PRECISION lu, PRECISION ju, PRECISION fos,int log);

//data -> [lines, sl,ll,jl,su,lu,ju, fos]  (fos only if lines>1)

Cuantic *create_cuantic(PRECISION *dat,int log)
{
	Cuantic *cuantic;

	int i;
	PRECISION lines;

	lines = dat[0];

	cuantic = (Cuantic *) calloc(lines, sizeof(Cuantic));

	for (i = 0; i < lines; i++)
	{
		Cuanten(&(cuantic[i]), dat[i * 7 + 1], dat[i * 7 + 2], dat[i * 7 + 3],
				  dat[i * 7 + 4],
				  dat[i * 7 + 5],
				  dat[i * 7 + 6],
				  (lines > 1 ? dat[i * 7 + 7] : 1),log);
	}

	return cuantic;
}

int Cuanten(Cuantic *cuantic, PRECISION sl, PRECISION ll, PRECISION jl, PRECISION su, PRECISION lu, PRECISION ju, PRECISION fos, int log)
{
	//SL(I),     SU(I),     LL(I),    LU(I),    JL(I),   JU(I)
	//,NUB,NUP,NUR,WEB,WEP,WER,GLO,GUP,GEF
	//	;1-> LOW
	//	;2-> UP

	int *m1, *m2, im, i, j;
	int jj, n_pi, n_sig;
	REAL g1, g2, geff;
	//REAL *pi, *sig1, *sig2, *mpi, *msig1, *msig2;
	int ipi, isig1, isig2;
	REAL sumpi, sumsig1, sumsig2;

	//lande factors (with 'if' because j could be cero)
	if (jl != 0)
		g1 = (3.0 * jl * (jl + 1) + sl * (sl + 1) - ll * (ll + 1)) / (2 * jl * (jl + 1));
	else
		g1 = 0;

	if (ju != 0)
		g2 = (3.0 * ju * (ju + 1.0) + su * (su + 1.0) - lu * (lu + 1.0)) / (2.0 * ju * (ju + 1.0));
	else
		g2 = 0;

	geff = (g1 + g2) / 2 + (g1 - g2) * (jl * (jl + 1.) - ju * (ju + 1.)) / 4;

	//	printf("g1 : %f\n",g1);
	//	printf("g2 : %f\n",g2);

	//magnetic quanten number Mlo Mup
	m1 = (int *) calloc(2 * jl + 1, sizeof(int));
	for (i = 0; i < 2 * jl + 1; i++)
	{
		m1[i] = i - (int)jl;
		//		printf("m1 (%d) : %d \n",i,m1[i]);
	}
	m2 = (int *) calloc(2 * ju + 1, sizeof(int));
	for (i = 0; i < 2 * ju + 1; i++)
	{
		m2[i] = i - (int)ju;
		//		printf("m2 (%d) :%d\n",i,m2[i]);
	}

	n_pi = (2 * (jl < ju ? jl : ju)) + 1; //Number of pi components
	n_sig = jl + ju;							  //Number of sigma components

	//	printf("n_pi :%d\n",n_pi);
	//	printf("n_sig :%d\n",n_sig);

	//BLUE COMPONENT => Mlo-Mup = +1
	//RED COMPONENT => Mlo-Mup = -1
	//CENTRAL COMPONENT => Mlo-Mup = 0

	//pi = calloc(n_pi, sizeof(REAL));
	//sig1 = calloc(n_sig, sizeof(REAL));
	//sig2 = calloc(n_sig, sizeof(REAL));
	//mpi = calloc(n_pi, sizeof(REAL));
	//msig1 = calloc(n_sig, sizeof(REAL));
	//msig2 = calloc(n_sig, sizeof(REAL));

	//counters for the components
	ipi = 0;
	isig1 = 0;
	isig2 = 0;

	jj = ju - jl;

	for (j = 0; j <= 2 * jl; j++)
	{
		for (i = 0; i <= 2 * ju; i++)
		{
			im = m2[i] - m1[j];
			switch (im)
			{
			case 0: //M -> M  ;CENTRAL COMPONENT
				switch (jj)
				{
				case -1:
					//j -> j-1
					cuantic->WEP[ipi] = jl * jl - m1[j] * m1[j];
					break;
				case 0:
					//  j -> j
					cuantic->WEP[ipi] = m1[j] * m1[j];
					break;
				case 1:
					//  j -> j+1
					cuantic->WEP[ipi] = (jl + 1) * (jl + 1) - m1[j] * m1[j];
					break;
				}
				cuantic->NUP[ipi] = g1 * m1[j] - g2 * m2[i];
				ipi = ipi + 1;
				break;
			case 1: //M -> M+1  ;BLUE COMPONENT
				switch (jj)
				{
				case -1:
					//j -> j-1
					cuantic->WEB[isig1] = (jl - m1[j]) * (jl - m1[j] - 1) / 4;
					break;
				case 0:
					//  j -> j
					cuantic->WEB[isig1] = (jl - m1[j]) * (jl + m1[j] + 1) / 4;
					break;
				case 1:
					//  j -> j+1
					cuantic->WEB[isig1] = (jl + m1[j] + 1) * (jl + m1[j] + 2) / 4;
					break;
				}
				cuantic->NUB[isig1] = g1 * m1[j] - g2 * m2[i];
				isig1 = isig1 + 1;
				break;
			case -1: //M -> M-1   ;RED COMPONENT
				switch (jj)
				{
				case -1:
					//j -> j-1
					cuantic->WER[isig2] = (jl + m1[j]) * (jl + m1[j] - 1) / 4;
					break;
				case 0:
					//  j -> j
					cuantic->WER[isig2] = (jl + m1[j]) * (jl - m1[j] + 1) / 4;
					break;
				case 1:
					//  j -> j+1
					cuantic->WER[isig2] = (jl - m1[j] + 1) * (jl - m1[j] + 2) / 4;
					break;
				}
				cuantic->NUR[isig2] = g1 * m1[j] - g2 * m2[i];
				isig2 = isig2 + 1;
				break;
			} //end switch
		}
	}

	//normalization OF EACH COMPONENT

	sumpi = 0;
	for (i = 0; i < n_pi; i++)
	{
		sumpi = sumpi + cuantic->WEP[i];
	}

	//C_N(IL).wep(i)
	for (i = 0; i < n_pi; i++)
	{
		cuantic->WEP[i] = cuantic->WEP[i] / sumpi;
	}

	sumsig1 = 0;
	for (i = 0; i < n_sig; i++)
	{
		sumsig1 = sumsig1 + cuantic->WEB[i];
	}
	for (i = 0; i < n_sig; i++)
	{
		cuantic->WEB[i] = cuantic->WEB[i] / sumsig1;
	}

	sumsig2 = 0;
	for (i = 0; i < n_sig; i++)
	{
		sumsig2 = sumsig2 + cuantic->WER[i];
	}
	for (i = 0; i < n_sig; i++)
	{
		cuantic->WER[i] = cuantic->WER[i] / sumsig2;
	}

	//Cuanten,S1,S2,L1,L2,J1,J2,msig1,mpi,msig2,sig1,pi, sig2,g1, g2 , geff
	//CUANTEN,SL,SU,LL,LU,JL,JU,NUB,  NUP,NUR,  WEB, WEP,WER, GLO,GUP,GEF

	cuantic->N_PI = n_pi;
	cuantic->N_SIG = n_sig;
	//cuantic->NUB = msig1;
	//cuantic->NUP = mpi;
	//cuantic->NUR = msig2;
	//cuantic->WEB = sig1;
	//cuantic->WEP = pi;
	//cuantic->WER = sig2;
	cuantic->GL = g1;
	cuantic->GU = g2;
	cuantic->GEFF = geff;
	cuantic->FO = fos;

	if(log){
		printf("\n\n------------    QUANTUM NUMBERS   --------------");
		printf("\n------------------------------------------------\n");
		printf("\nNumber of Pi components:\t\t %d",cuantic->N_PI);
		printf("\nNumber of Sigma components:\t\t %d",cuantic->N_SIG);
		printf("\nLower level lande factor:\t\t %lf",cuantic->GL);
		printf("\nUpper level lande factor:\t\t %lf",cuantic->GU);
		printf("\nEffective lande factor:\t\t\t   %lf",cuantic->GEFF);
		printf("\nShifts principal component:\t\t");
		for(i=0;i<cuantic->N_PI;i++){
			if(i<(cuantic->N_PI-1) )
				printf(" %lf\t",cuantic->NUP[i]);
			else
				printf(" %lf ",cuantic->NUP[i]);
		}
		printf("\nShifts blue component:\t\t\t ");
		for(i=0;i<cuantic->N_SIG;i++){
			if(i<(cuantic->N_SIG-1) )
				printf(" %lf\t",cuantic->NUB[i]);
			else
				printf(" %lf ",cuantic->NUB[i]);
		}
		printf("\nShifts red component:\t\t\t ");
		for(i=0;i<cuantic->N_SIG;i++){
			if(i<(cuantic->N_SIG-1) )
				printf(" %lf\t",cuantic->NUR[i]);
			else
				printf(" %lf ",cuantic->NUR[i]);
		}
		printf("\nStrength principal component:\t\t ");
		for(i=0;i<cuantic->N_PI;i++){
			if(i<(cuantic->N_PI-1) )
				printf(" %lf\t",cuantic->WEP[i]);
			else
				printf(" %lf ",cuantic->WEP[i]);
		}
		printf("\nStrength blue component:\t\t ");
		for(i=0;i<cuantic->N_SIG;i++){
			if(i<(cuantic->N_SIG-1) )
				printf(" %lf\t",cuantic->WEB[i]);
			else
				printf(" %lf ",cuantic->WEB[i]);
		}
		printf("\nStrength red component:\t\t\t ");
		for(i=0;i<cuantic->N_SIG;i++){
			if(i<(cuantic->N_SIG-1) )
				printf(" %lf\t",cuantic->WER[i]);
			else
				printf(" %lf ",cuantic->WER[i]);
		}
		printf("\nRelative line strength:\t\t\t %lf",cuantic->FO);
		printf("\n\n------------------------------------------------\n");	
	}

	free(m1);
	free(m2);

	return 1;
}
