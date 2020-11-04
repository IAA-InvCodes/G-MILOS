#include "defines.h"
#include "definesCuda.cuh"
#include <string.h>
#include "lib.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "multShare.cuh"

/*

 el tamaño de w es 	nlambda*NPARMS;

return 
	- beta de tam 1 x NTERMS
	- alpha de tam NTERMS x NTERMS

*/

__device__ int covarm(const REAL * __restrict__ w,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM){	
	
	//int j,i,bt_nf,bt_nc,
	//int aux_nf,aux_nc,bt_nf,bt_nc;
	int j,i;
	REAL *BTaux,*APaux;
	//REAL AP[NTERMS*NTERMS*NPARMS],
	//REAL BT[NPARMS*NTERMS];	
	//REAL opa[30];
	for(j=0;j<NPARMS;j++){
		
		/*for(i=0;i<nspectro;i++){
			opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
		}*/
		if(nspectro%2==0){
			for(i=0;i<nspectro;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}						
		}
		else{
			for(i=0;i<nspectro-1;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}			
			pM->opa[nspectro-1]= w[j]*(spectra[(nspectro-1)+nspectro*j]-spectro[(nspectro-1)+nspectro*j]);
		}	

		BTaux=pM->BT+(j*NTERMS);
		//BTaux=pM->BT(j*NTERMS);
		APaux=pM->AP+(j*NTERMS*NTERMS);
		
		multmatrixIDLValue(pM->opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,sig); //bt de tam NTERMS x 1
		//multmatrixIDLValueSigma(pM->opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,&bt_nf,&bt_nc,sig); //bt de tam NTERMS x 1
		/*dim3 dimBlock1(NTERMS,1);
		d_multmatrixIDLValueSigma<<<1,dimBlock1>>>(pM->opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,sig); //bt de tam NTERMS x 1
		cudaDeviceSynchronize();*/
		//multmatrix_transpose(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,&aux_nf,&aux_nc,w[j]/sig);//ap de tam NTERMS x NTERMS
		//multmatrix_transpose_sigma(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,&aux_nf,&aux_nc,w[j], sig);//ap de tam NTERMS x NTERMS
		dim3 dimBlock2(NTERMS,NTERMS);
		d_multmatrix_transpose<<<1,dimBlock2>>>(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,__fdividef(w[j],sig));//ap de tam NTERMS x NTERMS
		cudaDeviceSynchronize();
	}

	totalParcialf(pM->BT,NPARMS,NTERMS,beta); //beta de tam 1 x NTERMS
	totalParcialMatrixf(pM->AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	/*dim3 dimBlock3(NTERMS,NTERMS);
	d_totalParcialMatrixf<<<1,dimBlock3>>>(pM->AP,NTERMS,NTERMS,NPARMS,alpha); //alpha de tam NTERMS x NTERMS
	cudaDeviceSynchronize();*/

	return 1;
}

__device__ void covarm2(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,PRECISION *beta,REAL *alpha,ProfilesMemory * pM){	
	

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
					//REAL dAux = (*(d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					//sum += *(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					sum += __ldg(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					//sum += __ldg(pM->d_spectra_backup+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					
					if(i==0){
						//sum2 += (w_aux*(spectra[k+nspectro*j]-spectro[k+nspectro*j])) * dAux;
						sum2 += (w_aux*( __ldg(spectra+k+nspectro*j)-__ldg(spectro+k+nspectro*j) )) * dAux;
						//sum2 +=  __ldg(pM->opa+k) * dAux;
						
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
		
		/*if(nspectro%2==0){
			for(i=0;i<nspectro;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}						
		}
		else{
			for(i=0;i<nspectro-1;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}			
			pM->opa[nspectro-1]= w[j]*(spectra[(nspectro-1)+nspectro*j]-spectro[(nspectro-1)+nspectro*j]);
		}*/
				
		/*for(i=0;i<nspectro;i++){
			//pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
			pM->opa[i]= w[j]*( __ldg(spectra+i+nspectro*j)-__ldg(spectro+i+nspectro*j) );
		}*/

		REAL w_aux = w[j];
		REAL w_d_aux = w_d[j];
		BTaux=pM->BT+(j*NTERMS);
		APaux=pM->AP+(j*NTERMS*NTERMS);
		//REAL div_aux = __fdividef(w[j],sig);
		
		//multmatrixIDLValue(pM->opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,sig); //bt de tam NTERMS x 1
		/*for ( i = 0; i < NTERMS; i++){		
			sum=0;
			for ( k = 0;  k < nspectro; k++){
				//printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
				sum += (w[j]*(spectra[k+nspectro*j]-spectro[k+nspectro*j])) * (*(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k));
			}
			//printf("Sum, result[%d][%d] : %f \n",i,j,sum);
			BTaux[i] = sum/sig;	
		}*/
		
		//multmatrix_transpose2(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,w[j]/sig);//ap de tam NTERMS x NTERMS
		//multmatrix_transposeCUBLAS(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,w[j]/sig);
		for ( i = 0; i < NTERMS; i++){
			//#pragma unroll
			for ( h = 0; h < NTERMS; h++){
				sum=0;
				if(i==0)
					sum2=0;
				
				for ( k = 0;  k < nspectro; k++){
					//REAL dAux = (*(d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					//sum += *(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					sum += __ldg(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					//sum += __ldg(pM->d_spectra_backup+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					
					if(i==0){
						//sum2 += (w_aux*(spectra[k+nspectro*j]-spectro[k+nspectro*j])) * dAux;
						sum2 += (w_aux*( __ldg(spectra+k+nspectro*j)-__ldg(spectro+k+nspectro*j) )) * dAux;
						//sum2 +=  __ldg(pM->opa+k) * dAux;
						
					}
				}
	
				APaux[(NTERMS)*i+h] = (sum)*w_d_aux;
				if(i==0){
					BTaux[h] = __fdividef(sum2,sig);
				}
			} 
		}
		/*dim3 dimBlock2(NTERMS,NTERMS);
		//d_multmatrix_transpose<<<1,dimBlock2>>>(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,w[j]/sig);//ap de tam NTERMS x NTERMS
		d_multmatrix_transpose<<<1,dimBlock2,NTERMS*nspectro*sizeof(REAL)>>>(d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,APaux,w[j]/sig);//ap de tam NTERMS x NTERMS		
		cudaDeviceSynchronize();*/
	}

	//totalParcialff(pM->BT,NPARMS,NTERMS,beta); //beta de tam 1 x NTERMS
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
	/*#pragma unroll
	for(i=0;i<NTERMS;i++)
		#pragma unroll
		for(j=0;j<NTERMS;j++){
			sum = pM->AP[i*NTERMS+j];
			sum2 = pM->AP[i*NTERMS+j+NTERMS*NTERMS];
			sum3 = pM->AP[i*NTERMS+j+NTERMS*NTERMS*2];
			sum4 = pM->AP[i*NTERMS+j+NTERMS*NTERMS*3];

			alpha[i*NTERMS+j] = sum + sum2 + sum3 + sum4;
		}	*/

}


__device__ void covarmf2(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM){	
	

	int j,i,k;
	int h;
	REAL sum;
	REAL sum2;
	REAL *APaux;
	REAL *BTaux;
	/*for(i=0;i<NTERMS;i++)
		beta[i]=0;*/

	for(j=0;j<NPARMS;j++){
		
		/*if(nspectro%2==0){
			for(i=0;i<nspectro;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}						
		}
		else{
			for(i=0;i<nspectro-1;i=i+2){
				pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
				pM->opa[i+1]= w[j]*(spectra[(i+1)+nspectro*j]-spectro[(i+1)+nspectro*j]);
			}			
			pM->opa[nspectro-1]= w[j]*(spectra[(nspectro-1)+nspectro*j]-spectro[(nspectro-1)+nspectro*j]);
		}*/		
		/*for(i=0;i<nspectro;i++){
			pM->opa[i]= w[j]*(spectra[i+nspectro*j]-spectro[i+nspectro*j]);
		}*/						

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
					//REAL dAux = (*(d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					//sum += *(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
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


__device__ void covarmf3(const REAL * __restrict__ w,const REAL * __restrict__ w_d,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM){	
	

	int j,i,k;
	int h;
	REAL sum;
	REAL sum2;
	REAL *APaux;
	REAL *BTaux;


	for(i=0;i<nspectro;i++){
		pM->opa[i*NPARMS]= w[0]*( __ldg(spectra+i+nspectro*j)-__ldg(spectro+i+nspectro*j) );
		pM->opa[i]= w[1]*( __ldg(spectra+i+nspectro*j)-__ldg(spectro+i+nspectro*j) );
		pM->opa[i]= w[2]*( __ldg(spectra+i+nspectro*j)-__ldg(spectro+i+nspectro*j) );
		pM->opa[i]= w[3]*( __ldg(spectra+i+nspectro*j)-__ldg(spectro+i+nspectro*j) );
	}

	BTaux=pM->BT+(j*NTERMS);
	multmatrixIDLValue(pM->opa,nspectro,1,d_spectra+j*nspectro*NTERMS,NTERMS,nspectro,BTaux,sig); //bt de tam NTERMS x 1

	for(j=0;j<NPARMS;j++){

		REAL w_d_aux = w_d[j];
		
		APaux=pM->AP+(j*NTERMS*NTERMS);
		for ( i = 0; i < NTERMS; i++){
			for ( h = 0; h < NTERMS; h++){
				sum=0;
				for ( k = 0;  k < nspectro; k++){
					//REAL dAux = (*(d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					REAL dAux = __ldg((d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					//sum += *(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					sum += __ldg(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					//sum += __ldg(pM->d_spectra_backup+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
				}
				APaux[(NTERMS)*i+h] = (sum)*w_d_aux;
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

/*
__global__ void d_covarmf(const REAL * __restrict__ w,const REAL sig,const float * __restrict__ spectro,int  nspectro,const REAL * __restrict__ spectra,const REAL * __restrict__ d_spectra,REAL *beta,REAL *alpha,ProfilesMemory * pM){	
	

	int i = blockIdx.x * blockDim.x + threadIdx.x; 	// row NTERMS
	int j = blockIdx.x * blockDim.x + threadIdx.y; // col NSPECTRO

	extern __shared__ float s_d_spectra; 



	REAL sum,sum2;
	REAL *APaux;
	REAL *BTaux;


	for(j=0;j<NPARMS;j++){
		BTaux=pM->BT+(j*NTERMS);
		APaux=pM->AP+(j*NTERMS*NTERMS);

		for ( i = 0; i < NTERMS; i++){
			for ( h = 0; h < NTERMS; h++){
				sum=0;
				if(i==0)
					sum2=0;
				for ( k = 0;  k < nspectro; k++){
					REAL dAux = (*(d_spectra+(j*nspectro*NTERMS)+(h*nspectro)+k));
					sum += *(d_spectra+(j*nspectro*NTERMS)+(i*nspectro)+k) * dAux;
					if(i==0){
						sum2 += (w[j]*(spectra[k+nspectro*j]-spectro[k+nspectro*j])) * dAux;
					}
				}
	
				APaux[(NTERMS)*i+h] = (sum)*(w[j]/sig);
				if(i==0){
					BTaux[h] = sum2/sig;
				}
			} 
		}
	}
}
*/
__global__ void d_fchisqr(const REAL * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL sig, const REAL nfree, REAL * TOT){
	
	REAL dif1,dif2,dif3,dif4;
	REAL opa1,opa2,opa3,opa4;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	dif1=spectra[i]-spectro[i];
	dif2=spectra[i+nspectro]-spectro[i+nspectro];
	dif3=spectra[i+nspectro*2]-spectro[i+nspectro*2];
	dif4=spectra[i+nspectro*3]-spectro[i+nspectro*3];
	//printf("\n DIF SPECTRA: %f ; sigma %f ; value opa %f", dif,sig[i+nspectro*j] , (((dif*dif)*w[j])/(sig[i+nspectro*j])));
	opa1= (((dif1*dif1)*w[0])/(sig));
	opa2= (((dif2*dif2)*w[1])/(sig));
	opa3= (((dif3*dif3)*w[2])/(sig));
	opa4= (((dif4*dif4)*w[3])/(sig));


	atomicAdd(TOT,opa1+opa2+opa3+opa4);
		
	
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

__device__ REAL fchisqr2(const float * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree){
	
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
		/*float4 spectraAux = ((float4 *) spectra)[(i*NPARMS)];
		float4 spectroAux = ((float4 *) spectro)[(i*NPARMS)];
		dif1=spectraAux.x - spectroAux.x;
		dif2=spectraAux.y - spectroAux.y;
		dif3=spectraAux.z - spectroAux.z;
		dif4=spectraAux.w - spectroAux.w;

		printf("\n spectroAux.x %f spectroAux.y %f spectroAux.z %f spectroAux.w %f spectro +2  %f \n",spectroAux.x,spectroAux.y,spectroAux.z,spectroAux.w,spectro[(i*NPARMS)]);*/

		dif1=spectra[(i*NPARMS)]-spectro[(i*NPARMS)];
		dif2=spectra[(i*NPARMS)+1]-spectro[(i*NPARMS)+1];
		dif3=spectra[(i*NPARMS)+2]-spectro[(i*NPARMS)+2];
		dif4=spectra[(i*NPARMS)+3]-spectro[(i*NPARMS)+3];
		
		opa1+= __fdividef(((dif1*dif1)*w_0),(sig));
		opa2+= __fdividef(((dif2*dif2)*w_1),(sig));
		opa3+= __fdividef(((dif3*dif3)*w_2),(sig));
		opa4+= __fdividef(((dif4*dif4)*w_3),(sig));		
		
	}
	TOT= opa1+opa2+opa3+opa4;
	
	return TOT/nfree;
	
}


__device__ REAL fchisqr3(const float4 * __restrict__ spectra,const int  nspectro,const float * __restrict__ spectro, const REAL *  w, const REAL  sig, const REAL nfree){
	
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
		
		float4 spectraAux = spectra[i];
		dif1=spectraAux.x-spectro[(i*NPARMS)];
		dif2=spectraAux.y-spectro[(i*NPARMS)+1];
		dif3=spectraAux.z-spectro[(i*NPARMS)+2];
		dif4=spectraAux.w-spectro[(i*NPARMS)+3];
		
		//printf("\n fschiqr3");
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
	

		
		/*for ( i = 0; i < NTERMS; i++){
		    for ( j = 0; j < nac; j++){
				sum=0;
				#pragma unroll
				for ( k = 0;  k < naf; k++){
					//printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
					//sum += a[k*nac+j] * b[i*nbc+k];
					//sum += a[k+j] * b[i*nbc+k];
					sum += a[k] * b[i*nbc+k];
				}
				//printf("Sum, result[%d][%d] : %f \n",i,j,sum);
				//result[((nac)*i)+j] = sum/value;
				//result[i+j] = sum/value;
				result[i] = sum/value;
      	} 
		}*/

	
	for ( i = 0; i < NTERMS; i++){		
		sum=0;
		for ( k = 0;  k < naf; k++){
			//printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
			sum += a[k] * b[i*nbc+k];
		}
		//printf("Sum, result[%d][%d] : %f \n",i,j,sum);
		result[i] = sum/value;	
	}		
		
	

}

__device__ int multmatrixIDLValueSigma(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,int *fil,int *col, const REAL sigma){
    
	int i,j,k;
	REAL sum;
	 
	 if(naf==nbc){
		 (*fil)=nbf;
		 (*col)=nac;
 
		 for ( i = 0; i < nbf; i++){
			 for ( j = 0; j < nac; j++){
				 sum=0;
				 for ( k = 0;  k < naf; k++){
					//printf("i: %d,j:%d,k=%d .. a[%d][%d]:%f  .. b[%d][%d]:%f\n",i,j,k,k,j,a[k*nac+j],i,k,b[i*nbc+k]);
					sum += (((a[k*nac+j] * b[i*nbc+k])))/sigma;
				 }
				 //printf("Sum, result[%d][%d] : %f \n",i,j,sum);
				 result[((nac)*i)+j] = sum;
			   } 
		 }
		 return 1;
	 }
	 else
		 printf("\n \n Error en multmatrixIDLValue no coinciden nac y nbf!!!! ..\n\n");
	 return 0;
}

__global__ void d_multmatrixIDLValueSigma(REAL *a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL *result,const REAL sigma){
    
	int k;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 	// row
	int j = blockIdx.x * blockDim.x + threadIdx.y; // col
	REAL sum=0;
	for ( k = 0;  k < naf; k++){
		sum += (((a[k*nac+j] * b[i*nbc+k])))/sigma;
	}
	result[((nac)*i)+j] = sum;

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
		//sum = 0;
		
		/*for(j=0;j<f;j++){
			sum+=A[j*c+i];
		}*/

		sum=A[i];
		sum2=A[c+i];
		sum3=A[2*c+i];
		sum4=A[3*c+i];
		result[i] = sum + sum2 + sum3 + sum4;
		//result[i] = sum;
	}
}

__device__ void totalParcialff(const REAL * __restrict__ A, int f,int c,float *  result){

	int i;
	REAL sum,sum2,sum3,sum4;
	#pragma unroll
	for(i=0;i<c;i++){
		//sum = 0;
		
		/*for(j=0;j<f;j++){
			sum+=A[j*c+i];
		}*/

		sum=A[i];
		sum2=A[c+i];
		sum3=A[2*c+i];
		sum4=A[3*c+i];
		//result[i] = A[i] + A[c+i] + A[2*c+i] + A[3*c+i];
		result[i] = sum + sum2 + sum3 + sum4;
		//result[i] = sum;
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
			/*for(k=0;k<p;k++)
				sum+=A[i*c+j+f*c*k];*/
			sum = A[i*c+j];
			sum2 = A[i*c+j+f*c];
			sum3 = A[i*c+j+f*c*2];
			sum4 = A[i*c+j+f*c*3];

			//result[i*c+j] = sum;
			result[i*c+j] = sum + sum2 + sum3 + sum4;
		}

//	return result;
}

__global__ void d_totalParcialMatrixf(REAL * A, int f,int c,int p,REAL *result){

	int k;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 	// row
	int j = blockIdx.x * blockDim.x + threadIdx.y; // col
	/*REAL sum1=0;
	REAL sum2=0;
	REAL sum3=0;
	REAL sum4=0;*/
	if(i<f && j<c){
		REAL sum=0;
		#pragma unroll
		for(k=0;k<p;k++)
			sum+=A[i*c+j+f*c*k];

		/*sum1=A[i*c+j+f*c*0];
		sum2=A[i*c+j+f*c*1];
		sum3=A[i*c+j+f*c*2];
		sum4=A[i*c+j+f*c*3];
		result[i*c+j] = sum1 + sum2 + sum3 + sum4;*/
		result[i*c+j] = sum;
	}
	

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

__device__ void multmatrixf(float *a,int naf,int nac, float *b,int nbf,int nbc,float *result){
    
   int i,j,k;
	float sum;	
	
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

}



/**
* In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
* C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
* implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
* If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
* multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
* is a column-based cublas matrix, which means C(T) in C/C++, we need extra
* transpose code to convert it to a row-based C/C++ matrix.

* To solve the problem, let's consider our desired result C, a row-major matrix.
* In cublas format, it is C(T) actually (because of the implicit transpose).
* C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
* happen to be C/C++ matrice B and A (still because of the implicit transpose)!
* We don't need extra transpose code, we only need alter the input order!
*/
__device__ int multmatrixCUBLAS(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col){


	if(nac==nbf){
		(*fil)=naf;
		(*col)=nbc;
		
		/*cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,naf, nbc, nac, 1.0, a, nac, b, nbc, 0.0, result, nbc);*/
        const double alfa = 1.0f;
        const double beta  = 0.0f;
		cublasHandle_t handle;
		cublasCreate(& handle ); // initialize CUBLAS context
		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,nbc,naf,nac,&alfa,b,nbc,a,nac,&beta,result,nbc);
		cublasDestroy(handle);
		//cudaDeviceSynchronize();
		//
		/*
			A[m][k] B[k][n], C=A*B
			m=naf
			k=nac=nbf
			n=nbc
		*/
		//cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n );
		//cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n)

		return 1;
	}
	return 0;

}


/*
	* CUBLAS works with column-major order. Our matrix are stored in row-major and we need to do C = A * B(T)
	* The C result in cublas is column-major too, so we need to do the next: C(T) = (A * B(T)) (T), this is equal to 
	* C(T) = (B(T))(T) * A(T) --> C(T) = B * A(T) , in column-major 
	* in row major C = B(T) * A 
*/
__device__ void multmatrix_transposeCUBLAS(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value){
    
	
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,naf, nbf, nac, value, a, nac, a, nac, 0.0, result, naf);		
	
	//const float alfa = 1.0f;
	//const float beta  = 0.0f;
	cublasHandle_t handle;
	cublasCreate(&handle); // initialize CUBLAS context
	// m = nbc , n =

	/*
		A[m][k] B[k][n], C=A*B
		m=naf
		k=nac=nbf
		n=nbc
	*/
	//cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,nbf,naf,nac,&value,b,nbf,a,nac,&beta,result,naf);
	//cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,naf,naf,nac,&value,b,naf,a,nac,&beta,result,naf);
	
	cublasDestroy(handle);
		
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

__device__ void multmatrix_transpose2(const REAL *a,int naf,int nac,const REAL *b,int nbf,int nbc,REAL *result,REAL value){
    int i,j,k;
    REAL sum;
	for ( i = 0; i < naf; i++){
		for ( j = 0; j < nbf; j++){
			sum=0;
			for ( k = 0;  k < nbc; k++){
				sum += a[i*nac+k] * b[j*nbc+k];
			}

			result[(nbf)*i+j] = (sum)*value;
		} 
	}

}

__device__ int multmatrix_transpose_sigma(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL *result,int *fil,int *col,REAL weigth, const REAL sigma){
    
    int i,j,k;
    REAL sum;
    
	if(nac==nbc){
		(*fil)=naf;
		(*col)=nbf;
		REAL aux_mul = (weigth/sigma);
		for ( i = 0; i < naf; i++){
		    for ( j = 0; j < nbf; j++){
				sum=0;
				for ( k = 0;  k < nbc; k++){
					sum += (a[i*nac+k] * b[j*nbc+k]) * aux_mul;
				}

				result[(*col)*i+j] = sum;
     		} 
		
		}
		return 1;
	}else{
		printf("\n \n Error en multmatrix_transpose no coinciden nac y nbc!!!! ..\n\n");
	}

	return 0;
}

//__global__ void d_multmatrix_transpose_sigma(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,REAL weigth, const REAL sigma){
__global__ void d_multmatrix_transpose(const REAL * __restrict__ a,int naf,int nac,const REAL * __restrict__ b,int nbf,int nbc,REAL * __restrict__ result,const REAL value){
    
    
    
	int i = blockIdx.x * blockDim.x + threadIdx.x; 	// row
	int j = blockIdx.x * blockDim.x + threadIdx.y; // col
    int k;
	REAL sum=0;
	//REAL sum2=0;
	extern __shared__ REAL d_a[];
	//REAL * d_b = (REAL *) &d_a[NTERMS*nbc];
	
	//#pragma unroll
	//for ( k = 0;  k < nbc; k++){
	//#pragma unroll
	for ( k = j;  k < nbc; k=k+(nbc/nbf)){
		//d_a[i*nbc+k] = d_b[i*nbc+k] = a[i*nbc+k];
		d_a[i*nbc+k] = a[i*nbc+k];
	}
	__syncthreads();
	
	/*if(nbc%2==0){
		
		for ( k = 0;  k < nbc; k =k+2){
			sum += (a[i*nac+k] * b[j*nbc+k]);
			sum2 += (a[i*nac+(k+1)] * b[j*nbc+(k+1)]);
		}
	}
	else{
		for ( k = 0;  k < nbc-1; k =k+2){
			sum += (a[i*nac+k] * b[j*nbc+k]);
			sum2 += (a[i*nac+(k+1)] * b[j*nbc+(k+1)]);
		}		
		sum += (a[i*nac+ (nbc-1)] * b[j*nbc+ (nbc-1)]) ;
	result[(nbf)*i+j] = (sum+sum2)*value;		
	}*/
	
	#pragma unroll
	for ( k = 0;  k < nbc; k++){
		//sum += (a[i*nac+k] * b[j*nbc+k]);
		sum += (d_a[i*nac+k] * d_a[j*nbc+k]);
	}	

	result[(nbf)*i+j] = (sum)*value;

}



