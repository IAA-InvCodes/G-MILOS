#include "defines.h"
#include "definesCuda.cuh"
#include <string.h>
#include "lib.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*

 el tamaño de w es 	nlambda*NPARMS;

return 
	- beta de tam 1 x NTERMS
	- alpha de tam NTERMS x NTERMS

*/

__device__ void covarm(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM){	
	

	int j,i,k;
	int h;
	REAL sum,sum2;
	REAL *BTaux,*APaux;

	for(j=0;j<NPARMS;j++){
		REAL w_aux = w[j];
		REAL w_d_aux = w_d[j];
		BTaux=pM->BT+(j*NTERMS);
		APaux=pM->AP+(j*NTERMS*NTERMS);
		for ( i = 0; i < NTERMS; i++){
			//#pragma unroll
			for ( h = 0; h < NTERMS; h++){
				sum=0;
				if(i==0)
					sum2=0;
				
				for ( k = 0;  k < nspectro; k++){
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					sum += __ldg(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					if(i==0){
						sum2 += (w_aux*( __ldg(spectra+k+nspectro*j)-__ldg(spectro+k+nspectro*j) )) * dAux;
					}
				}
	
				APaux[(NTERMS)*i+h] = (sum)*w_d_aux;
				if(i==0){
					BTaux[h] = __fdividef(sum2,sig);
				}
			} 
		}
	}

	REAL sum3,sum4;
	#pragma unroll
	for(i=0;i<NTERMS;i++){
		sum=pM->BT[i];
		sum2=pM->BT[NTERMS+i];
		sum3=pM->BT[2*NTERMS+i];
		sum4=pM->BT[3*NTERMS+i];
		beta[i] = sum + sum2 + sum3 + sum4;
	}	
	totalParcialMatrixf(pM->AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	
}

__device__ void covarmf(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM){	
	

	int j,i,k;
	int h;
	REAL sum;
	REAL sum2;
	REAL *APaux;
	REAL *BTaux;
	/*for(i=0;i<NTERMS;i++)
		beta[i]=0;*/

	for(j=0;j<NPARMS;j++){
		

		REAL w_aux = w[j];
		REAL w_d_aux = w_d[j];
		BTaux=pM->BT+(j*NTERMS);
		APaux=pM->AP+(j*NTERMS*NTERMS);
	
		for ( i = 0; i < NTERMS; i++){
			for ( h = 0; h < NTERMS; h++){
				sum=0;
				if(i==0)
					sum2=0;
				
				for ( k = 0;  k < nspectro; k++){
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					sum += __ldg(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					
					if(i==0){
						sum2 += (w_aux*( __ldg(spectra+k+nspectro*j)-__ldg(spectro+k+nspectro*j) )) * dAux;
					}
				}
	
				APaux[(NTERMS)*i+h] = (sum)*w_d_aux;
				if(i==0){
					BTaux[h] = __fdividef(sum2,sig);
				}
			} 
		}
	}

	REAL sum3,sum4;
	#pragma unroll
	for(i=0;i<NTERMS;i++){
		sum=pM->BT[i];
		sum2=pM->BT[NTERMS+i];
		sum3=pM->BT[2*NTERMS+i];
		sum4=pM->BT[3*NTERMS+i];
		beta[i] = sum + sum2 + sum3 + sum4;
	}	
	totalParcialMatrixf(pM->AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
}

__device__ REAL fchisqr(const REAL * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree){
	
	REAL TOT,dif1,dif2,dif3,dif4;	
	REAL opa1,opa2,opa3,opa4;

	int i;

	TOT=0;
	opa1=0;
	opa2=0;
	opa3=0;
	opa4=0;
	REAL w_0 = w[0];
	REAL w_1 = w[1];
	REAL w_2 = w[2];
	REAL w_3 = w[3];


	
	for(i=0;i<nspectro;i++){
		dif1=spectra[i]-spectro[i];
		dif2=spectra[i+nspectro]-spectro[i+nspectro];
		dif3=spectra[i+nspectro*2]-spectro[i+nspectro*2];
		dif4=spectra[i+nspectro*3]-spectro[i+nspectro*3];
		
		opa1+= __fdividef(((dif1*dif1)*w_0),(sig));
		opa2+= __fdividef(((dif2*dif2)*w_1),(sig));
		opa3+= __fdividef(((dif3*dif3)*w_2),(sig));
		opa4+= __fdividef(((dif4*dif4)*w_3),(sig));		
		
	}
	TOT= opa1+opa2+opa3+opa4;
	
	return TOT/nfree;
	
}

/*

	Multiplica la matriz a (tamaño naf,nac)
	por la matriz b (de tamaño nbf,nbc)	
	al estilo IDL, es decir, filas de a por columnas de b,
	el resultado se almacena en resultOut (de tamaño fil,col)

	El tamaño de salida (fil,col) corresponde con (nbf,nac).

	El tamaño de columnas de b, nbc, debe de ser igual al de filas de a, naf.

*/
__device__ void multmatrixIDLValue(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value){
    
   int i,k;
   REAL sum;

	for ( i = 0; i < NTERMS; i++){		
		sum=0;
		for ( k = 0;  k < naf; k++){
			//printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
			sum += a[k] * b[i*nbc+k];
		}
		result[i] = sum/value;	
	}		
		
	

}


/**
 * 
 * @param A --> Matrix of size fxc
 * @param f --> num of Rows of A  --> We assume that f will be always 4. 
 * @param c --> num of Cols of A 
 * @param result --> Array of size c
 * 
 * Method to realize the sum of A by columns and store the results of each sum in array result. 
 * */
__device__ void totalParcialf(const REAL * __restrict__ A, int f,int c,PRECISION *  result){

	int i;
	REAL sum,sum2,sum3,sum4;
	#pragma unroll
	for(i=0;i<c;i++){
		sum=A[i];
		sum2=A[c+i];
		sum3=A[2*c+i];
		sum4=A[3*c+i];
		result[i] = sum + sum2 + sum3 + sum4;
	}
}



/**
 * @param A --> Matrix of tam fXcXp 3D 
 * @param f --> num of rows of A and result
 * @param c --> num of cols of A and result 
 * @param p --> num of depth of A 
 * @param result --> Matrix to store the result. Size fXc
 * 
 * Method to realize the sumatory in the axis depth for each element en (x,y) of A and store this sumatory in result(x,y)
 * */
__device__ void totalParcialMatrixf(const REAL * __restrict__ A, int f,int c,int p,REAL * result){

	int i,j;
	REAL sum,sum2,sum3,sum4;
	#pragma unroll
	for(i=0;i<f;i++)
		#pragma unroll
		for(j=0;j<c;j++){

			sum = A[i*c+j];
			sum2 = A[i*c+j+f*c];
			sum3 = A[i*c+j+f*c*2];
			sum4 = A[i*c+j+f*c*3];

			result[i*c+j] = sum + sum2 + sum3 + sum4;
		}

//	return result;
}

/*
	Multiplica la matriz a (tamaño naf,nac)
	por la matriz b (de tamaño nbf,nbc)
	al estilo multiplicación algebraica de matrices, es decir, columnas de a por filas de b,
	el resultado se almacena en resultOut (de tamaño fil,col)

	El tamaño de salida (fil,col) corresponde con (nbf,nac).

	El tamaño de columnas de a, nac, debe de ser igual al de filas de b, nbf.
*/

__device__ int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result){
    
    int i,j,k;
	PRECISION sum;	
	
	for ( i = 0; i < naf; i++)
		
		for ( j = 0; j < nbc; j++){
			sum=0;
			#pragma unroll
			for ( k = 0;  k < nbf; k++){
//					printf("i: %d,j:%d,k=%d .. a[%d][%d]  .. b[%d][%d]\n",i,j,k,i,k,k,j);
				sum += a[i*nac+k] * b[k*nbc+j];
			}
//				printf("Sum\n");
			result[(nbc)*i+j] = sum;

		} 

	return 1;
	


}



__device__ int multmatrix_transpose(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL value){
    int i,j,k;
    REAL sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		
		for ( i = 0; i < naf; i++){
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
					sum += a[i*nac+k] * b[j*nbc+k];
				}

				result[(*col)*i+j] = (sum)*value;
     		} 
		
		}
		return 1;
	}else{
		printf("\n \n Error en multmatrix_transpose no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}




__global__ void d_multmatrix_transpose(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,const REAL value){
    
    
    
	int i = blockIdx.x * blockDim.x + threadIdx.x; 	// row
	int j = blockIdx.x * blockDim.x + threadIdx.y; // col
    int k;
	REAL sum=0;

	extern __shared__ REAL d_a[];

	
	//#pragma unroll
	//for ( k = 0;  k < nbc; k++){
	//#pragma unroll
	for ( k = j;  k < nbc; k=k+(nbc/nbf)){
		d_a[i*nbc+k] = a[i*nbc+k];
	}
	__syncthreads();

	
	#pragma unroll
	for ( k = 0;  k < nbc; k++){
		sum += (d_a[i*nac+k] * d_a[j*nbc+k]);
	}	

	result[(nbf)*i+j] = (sum)*value;

}



