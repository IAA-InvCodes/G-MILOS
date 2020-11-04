#include <math.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include "convolution.cuh"
#include "defines.h"
#include "definesCuda.cuh"

extern __constant__ PRECISION d_psfFunction_const  [MAX_LAMBDA];

/*
	Convolucion para el caso Sophi: convolucion central de x con h.
	
*/

__global__ void d_direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	extern __shared__ double d_h[];	
	//__shared__ double d_dirConvPar[MAX_LAMBDA];
	int j;
	dirConvPar[i +  (nh / 2)] = x[i];
	//d_dirConvPar[i] = x[i];
	d_h[i]=h[i];
	__syncthreads();
	
	// vamos a tomar solo la convolucion central
	double aux = 0;
	//int N_start_point=i-(nh/2);
	for (j = 0; j < nh; j++)
	{
		/*if(((N_start_point+j)>=0) && ((N_start_point+j)<nh))
			aux += d_h[j] * d_dirConvPar[N_start_point+j];*/
			//aux += d_h[j] * x[N_start_point+j];
		aux += d_h[j] * dirConvPar[j + i];
	}
	//__syncthreads();
	x[i] = aux;
	
		
}

/**
	@param x signal 
	@param h kernel convolution
	@param nh size of x and h 
	
*/
__global__ void d_direct_convolution(REAL * __restrict__ x,const double * __restrict__ h, int nh)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*extern __shared__ double d_h[];	
	double * d_dirConvPar = (double *)&d_h[nh];*/
	extern __shared__ double d_dirConvPar[];
	int j;
	//dirConvPar[i +  (nh / 2)] = x[i];
	d_dirConvPar[i] = x[i];
	__syncthreads();
	//d_h[i]=h[i];
	//__syncthreads();
	// vamos a tomar solo la convolucion central
	double aux = 0;
	int N_start_point=i-(nh/2);
	#pragma unroll
	for (j = 0; j < nh; j++)
	{
		if(((N_start_point+j)>=0) && ((N_start_point+j)<nh))
			aux += d_psfFunction_const[j] * d_dirConvPar[N_start_point+j];
		//aux += d_h[j] * dirConvPar[j + i];
	}
	x[i] = aux;
		
}

/**
	@param x signal 
	@param h kernel convolution
	@param nh size of x and h 
	@param Ic value to rest in kernel convolution 
	
*/
__global__ void d_direct_convolution_ic(REAL * __restrict__ x, const double * __restrict__ h, int nh, REAL Ic)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*extern __shared__ double d_h[];	
	double * d_dirConvPar = (double *)&d_h[nh];*/
	extern __shared__ double d_dirConvPar[];
	int j;
	//dirConvPar[i +  (nh / 2)] = x[i];
	d_dirConvPar[i] = Ic - x[i];
	__syncthreads();
	/*d_h[i]=h[i];
	__syncthreads();*/
	// vamos a tomar solo la convolucion central
	double aux = 0;
	int N_start_point=i-(nh/2);
	#pragma unroll
	for (j = 0; j < nh; j++)
	{
		if(((N_start_point+j)>=0) && ((N_start_point+j)<nh))
			aux += d_psfFunction_const[j] * d_dirConvPar[N_start_point+j];
		//aux += d_h[j] * dirConvPar[j + i];
	}
	x[i] = Ic - aux;
		
}

__global__ void d_convCircular(REAL * __restrict__ x, const double * __restrict__ h, const int size, REAL * __restrict__ result)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ REAL s[];
	REAL * d_x = s;
	double * d_h = (double *)&d_x[size];
	/*__shared__ double d_h[MAX_LAMBDA];
	__shared__ REAL d_x[MAX_LAMBDA];*/
	
	d_h[i]=h[i];
	d_x[i]=x[i];
	__syncthreads();
	int j;
	int startShift = size/2;
	if(size%2) startShift+=1;
	double aux = 0.f;
	
	for(j=0; j < size; j++){
		if( (i-j)<0 )
			aux += d_h[j] * d_x[ (i-j)+size];
		else
			aux += d_h[j] * d_x[(i-j)];
		//printf("\n %f",aux);
	}
	//d_r[i] = aux;
	//__syncthreads();

	if(i < size/2){
		result[startShift+i] = aux;
	}
	else{
		result[i-(size/2)] = aux;		
	}

	
}

__device__ void direct_convolution_double(PRECISION * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar)
{

	int nx_aux;
	int k, j;

	nx_aux = nx + nh - 1; // tamano de toda la convolucion
	int mitad_nh = nh / 2;

	// rellenamos el vector auxiliar
	//#pragma unroll
	for (k = 0; k < nx_aux; k++)
	{
		dirConvPar[k] = 0;
	}

	
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = x[k];
	}

	// vamos a tomar solo la convolucion central

	for (k = 0; k < nx; k++)
	{
		//x[k] = 0;
		double aux = 0;

		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}
		x[k] = aux;
	}
}

__device__ void direct_convolution(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar)
{

	//int nx_aux;
	int k, j;

	//nx_aux = nx + nh - 1; // tamano de toda la convolucion
	
	//PRECISION  * dirConvPar = (PRECISION * )malloc(nx_aux * sizeof(PRECISION));
	int mitad_nh = nh / 2;

	// rellenamos el vector auxiliar
	/*for (k = 0; k < nx_aux; k++)
	{
		dirConvPar[k] = 0;
	}*/

	//#pragma unroll
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = x[k];
	}

	// vamos a tomar solo la convolucion central
	//double aux,aux2,aux3,aux4;
	double aux,aux2;
	for (k = 0; k < nx; k++)
	{
		//x[k] = 0;
		aux = 0;
		aux2 = 0;
		/*aux3 = 0;
		aux4 = 0;
		
		for (j = 0; j < nh/4; j++)
		{
			aux += h[j] * dirConvPar[j + k];
			aux2 += h[j+1] * dirConvPar[(j+1) + k];
			aux3 += h[j+2] * dirConvPar[(j+2) + k];
			aux4 += h[j+3] * dirConvPar[(j+3) + k];
		}
		int reminder = nh%4;
		
		for(j=0;j<reminder;j++){
			aux += h[j+ (nh-reminder)] * dirConvPar[j+ (nh-reminder) + k];
		}
		x[k] = aux+aux2+aux3+aux4;*/
		if(nh%2==0){
			
			for (j = 0; j < nh; j=j+2)
			{
				aux += h[j] * dirConvPar[j + k];
				aux2 += h[j+1] * dirConvPar[(j+1) + k];
			}			
		}
		else{
			for (j = 0; j < nh-1; j=j+2)
			{
				aux += h[j] * dirConvPar[j + k];
				aux2 += h[j+1] * dirConvPar[(j+1) + k];
			}				
			aux += h[nh-1] * dirConvPar[(nh-1) + k];
		}

		/*for (j = 0; j < nh/2; j++)
		{
			aux += h[j] * dirConvPar[j + k];
			aux2 += h[j+1] * dirConvPar[(j+1) + k];
		}		
		if(nh%2)*/
			
		x[k] = aux+aux2;
	}
	//free(dirConvPar);
}


__device__ void direct_convolution2(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar)
{

	//int nx_aux;
	int k, j;
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;
		int N_start_point = k - (nh / 2);
		for (j = 0; j < nh; j++)
		{
			//aux += h[j] * dirConvPar[j + k];
			if (N_start_point + j >= 0 && N_start_point + j < nx) {
				aux += h[j] * x[N_start_point+ j];
			}
		}			
		x[k] = aux;
	}
	
}

__device__ void direct_convolution3(REAL *  x, int nx, const double *  h, int nh,PRECISION  *  dirConvPar)
{

	//int nx_aux;
	int k, j;
	int mitad_nh = nh / 2;

	//#pragma unroll
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = x[k];
	}
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;			
		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}			
		x[k] = aux;
	}
	
}


__device__ void direct_convolution_ic(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar,REAL Ic)
{

	//int nx_aux;
	int k, j;

	//nx_aux = nx + nh - 1; // tamano de toda la convolucion
	
	//PRECISION  * dirConvPar = (PRECISION * )malloc(nx_aux * sizeof(PRECISION));
	int mitad_nh = nh / 2;

	// rellenamos el vector auxiliar
	/*for (k = 0; k < nx_aux; k++)
	{
		dirConvPar[k] = 0;
	}*/

	
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = Ic - x[k];
	}

	// vamos a tomar solo la convolucion central
	//double aux,aux2,aux3,aux4;
	double aux,aux2;
	for (k = 0; k < nx; k++)
	{
		//x[k] = 0;
		aux = 0;
		aux2 = 0;
		/*aux3 = 0;
		aux4 = 0;
		
		for (j = 0; j < nh/4; j++)
		{
			aux += h[j] * dirConvPar[j + k];
			aux2 += h[j+1] * dirConvPar[(j+1) + k];
			aux3 += h[j+2] * dirConvPar[(j+2) + k];
			aux4 += h[j+3] * dirConvPar[(j+3) + k];
		}
		int reminder = nh%4;
		
		for(j=0;j<reminder;j++){
			aux += h[j+ (nh-reminder)] * dirConvPar[j+ (nh-reminder) + k];
		}
		x[k] = aux+aux2+aux3+aux4;*/
		if(nh%2==0){
			
			for (j = 0; j < nh; j=j+2)
			{
				aux += h[j] * dirConvPar[j + k];
				aux2 += h[j+1] * dirConvPar[(j+1) + k];
			}			
		}
		else{
			for (j = 0; j < nh-1; j=j+2)
			{
				aux += h[j] * dirConvPar[j + k];
				aux2 += h[j+1] * dirConvPar[(j+1) + k];
			}				
			aux += h[nh-1] * dirConvPar[(nh-1) + k];
		}

		/*for (j = 0; j < nh/2; j++)
		{
			aux += h[j] * dirConvPar[j + k];
			aux2 += h[j+1] * dirConvPar[(j+1) + k];
		}		
		if(nh%2)*/
			
		x[k] = Ic - (aux+aux2);
	}
	//free(dirConvPar);
}





__device__ void direct_convolution_ic2(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,PRECISION  * __restrict__ dirConvPar,REAL Ic)
{

	//int nx_aux;
	int k, j;
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;
		int N_start_point = k - (nh / 2);
		for (j = 0; j < nh; j++)
		{
			//aux += h[j] * dirConvPar[j + k];
			if (N_start_point + j >= 0 && N_start_point + j < nx) {
				aux += h[j] * (Ic - x[N_start_point+ j]);
			}
		}			
		x[k] = Ic - (aux);
	}
}


__device__ void direct_convolution_ic3(REAL * __restrict__ x, int nx, const double * __restrict__ h, int nh,REAL Ic)
{

	//int nx_aux;
	int k, j;
	double auxI,auxQ,auxU,auxV;
	for (k = 0; k < nx; k++)
	{
		auxI = 0;
		auxQ = 0;
		auxU = 0;
		auxV = 0;
		int N_start_point = k - (nh / 2);
		for (j = 0; j < nh; j++)
		{
			//aux += h[j] * dirConvPar[j + k];
			if (N_start_point + j >= 0 && N_start_point + j < nx) {
				double aux_h = h[j];
				auxI += aux_h * (Ic - x[N_start_point+ j]);
				auxQ += aux_h * x[N_start_point+ j + 1];
				auxU += aux_h * x[N_start_point+ j + 2];
				auxV += aux_h * x[N_start_point+ j + 3];
			}
		}			
		x[k] = Ic - (auxI);
		x[k+1] = auxQ;
		x[k+2] = auxU;
		x[k+3] = auxV;
	}
}

__device__ void direct_convolution_ic4(REAL * x, int nx, const double * h, int nh,PRECISION  * dirConvPar,REAL Ic)
{
	int k, j;
	int mitad_nh = nh / 2;
	
	for (k = 0; k < nx; k++)
	{
		dirConvPar[k + mitad_nh] = Ic - x[k];
	}
	double aux;
	for (k = 0; k < nx; k++)
	{
		aux = 0;	
		for (j = 0; j < nh; j++)
		{
			aux += h[j] * dirConvPar[j + k];
		}					
		x[k] = Ic - (aux);
	}
}


/**
 * Method to do circular convolution over signal 'x'. We assume signal 'x' and 'h' has the same size. 
 * The result is stored in array 'result'
 * 
 * 
 * */

__device__ void convCircular(const REAL * __restrict__ x, const double * __restrict__ h, int size, REAL * __restrict__ result,REAL * __restrict__ resultConv)
{
	int i,j,mod;
	double aux;

	int startShift = size/2;
	if(size%2) startShift+=1;	
	
	for(i=0; i < size ; i++){
		aux = 0;
    	for(j=0; j < size; j++){
			mod = i-j;
			if((i-j)<0)
				aux += h[j] * x[size + (i-j)];
			else
				aux += h[j] * x[mod];
		}
		
		if(i < size/2){
			resultConv[startShift+i] = aux;
		}
		else{
			resultConv[i-(size/2)] = aux;		
		}
	}

	for(i=0;i<size;i++){
		result[i] = resultConv[i];
	}
	
}

