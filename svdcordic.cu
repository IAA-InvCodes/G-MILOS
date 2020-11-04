#include "defines.h"
#include "definesCuda.cuh"
#include <math.h>

extern __constant__ int cordicPosFila[TAMANIO_SVD*TAMANIO_SVD];
extern __constant__ int cordicPosCol[TAMANIO_SVD*TAMANIO_SVD];
extern __constant__ PRECISION LIMITE_INFERIOR_PRECISION_SVD;
extern __constant__ PRECISION LIMITE_INFERIOR_PRECISION_TRIG;
extern __constant__ PRECISION LIMITE_INFERIOR_PRECISION_SINCOS;

#define limita_precision_fxp(number) ( fabs(number) > LIMITE_INFERIOR_PRECISION_SVD ? number : 0 )
#define limita_precision_trig(number) ( fabs(number) > LIMITE_INFERIOR_PRECISION_TRIG ? number : 0 )
#define limita_precision_SINCOS(number) ( fabs(number) > LIMITE_INFERIOR_PRECISION_TRIG ? number : 0 )

__device__ void ROTACION(PRECISION angulo,PRECISION in1_1,PRECISION in1_2,PRECISION in2_1,PRECISION in2_2,PRECISION* out1_1,PRECISION* out1_2,PRECISION* out2_1,PRECISION* out2_2){

	//PRECISION seno, coseno;
	float seno, coseno;

	/*seno=limita_precision_SINCOS(sin(angulo));	
	coseno=limita_precision_SINCOS(cos(angulo));*/

	/*seno=sin(angulo);	
	coseno=cos(angulo);*/
	//sincos(angulo,&seno,&coseno);
	sincosf(angulo,&seno,&coseno);

	//rotacion vector columna 1
	/*(*out1_1)=limita_precision_fxp(in1_1*coseno-in2_1*seno);
	(*out2_1)=limita_precision_fxp(in1_1*seno  +in2_1*coseno);

	//rotacion vector columna 2
	(*out1_2)=limita_precision_fxp(in1_2*coseno-in2_2*seno);
	(*out2_2)=limita_precision_fxp(in1_2*seno  +in2_2*coseno);*/

	//rotacion vector columna 1
	(*out1_1)=in1_1*coseno-in2_1*seno;
	(*out2_1)=in1_1*seno  +in2_1*coseno;

	//rotacion vector columna 2
	(*out1_2)=in1_2*coseno-in2_2*seno;
	(*out2_2)=in1_2*seno  +in2_2*coseno;



}



__device__ void ROTACIONf(float angulo,float in1_1,float in1_2,float in2_1,float in2_2,float* out1_1,float* out1_2,float* out2_1,float* out2_2){

	//PRECISION seno, coseno;
	float seno, coseno;

	/*seno=limita_precision_SINCOS(sin(angulo));	
	coseno=limita_precision_SINCOS(cos(angulo));*/

	/*seno=sin(angulo);	
	coseno=cos(angulo);*/
	//sincos(angulo,&seno,&coseno);
	__sincosf(angulo,&seno,&coseno);

	//rotacion vector columna 1
	/*(*out1_1)=limita_precision_fxp(in1_1*coseno-in2_1*seno);
	(*out2_1)=limita_precision_fxp(in1_1*seno  +in2_1*coseno);

	//rotacion vector columna 2
	(*out1_2)=limita_precision_fxp(in1_2*coseno-in2_2*seno);
	(*out2_2)=limita_precision_fxp(in1_2*seno  +in2_2*coseno);*/

	//rotacion vector columna 1
	(*out1_1)=in1_1*coseno-in2_1*seno;
	(*out2_1)=in1_1*seno  +in2_1*coseno;

	//rotacion vector columna 2
	(*out1_2)=in1_2*coseno-in2_2*seno;
	(*out2_2)=in1_2*seno  +in2_2*coseno;



}

__device__ PRECISION calculaAngulo(PRECISION in1_1,PRECISION in1_2,PRECISION in2_2)
{

	PRECISION angulo,in1_2b,dif,arc;


	if(in1_1==in2_2){  //Si los elementos de la diagonal son iguales rotamos PI/4.
		//angulo=limita_precision_fxp(PI/4);
		angulo=PI/4;
	}
	else{
		/*in1_2b=limita_precision_fxp(2*in1_2);
		dif=limita_precision_fxp(in2_2-in1_1);*/
		in1_2b=2*in1_2;
		dif=in2_2-in1_1;


		arc=in1_2b/dif;

		//angulo=limita_precision_trig(atan(arc))/2;
		angulo=atan(arc)/2;

//		printf("\nAngulo: %e \n",angulo);
		
		if(angulo>(PI/4)){
			//angulo=limita_precision_fxp(PI/4);
			angulo=PI/4;
		}

		if(angulo<(-PI/4)){
			//angulo=limita_precision_fxp(-PI/4);
			angulo=-PI/4;
		}

	}

	return angulo;

}


__device__ float calculaAngulof(float in1_1,float in1_2,float in2_2)
{

	float angulo,in1_2b,dif,arc;


	if(in1_1==in2_2){  //Si los elementos de la diagonal son iguales rotamos PI/4.
		//angulo=limita_precision_fxp(PI/4);
		angulo=PI/4;
	}
	else{
		/*in1_2b=limita_precision_fxp(2*in1_2);
		dif=limita_precision_fxp(in2_2-in1_1);*/
		in1_2b=2*in1_2;
		dif=in2_2-in1_1;


		arc=__fdividef(in1_2b,dif);

		//angulo=limita_precision_trig(atan(arc))/2;
		angulo=__fdividef(atanf(arc),2);

//		printf("\nAngulo: %e \n",angulo);
		
		if(angulo>__fdividef(PI,4)){
			//angulo=limita_precision_fxp(PI/4);
			angulo=__fdividef(PI,4);
		}

		if(angulo<__fdividef(-PI,4)){
			//angulo=limita_precision_fxp(-PI/4);
			angulo=__fdividef(-PI,4);
		}

	}

	return angulo;

}

__device__ PRECISION PD(PRECISION in1_1,PRECISION in1_2,PRECISION in2_2,PRECISION *out1_1,PRECISION *out1_2,PRECISION *out2_2){
	
	PRECISION angulo;

	PRECISION kk;


	//printf("PD: Se calcula el angulo de : %e  ,  %e   ,   %e  \n",in1_1,in1_2,in2_2);


	angulo=calculaAngulo(in1_1,in1_2,in2_2);
	
	//////////////////
	
	//Para probar a normalizar submatriz a submatriz, hay q resolver el problema del calculo del angulo cuando la division antes de atan es cero
	
	// printf("PD: Se calcula el angulo de \t\t: %e  ,  %e   ,   %e  -> %e \n",in1_1,in1_2,in2_2,angulo);
	
	// PRECISION max= fabs(in1_1);
	
	// if(fabs(in1_2) > max)
		// max = in1_2;
	
	// if(fabs(in2_2) > max)
		// max = in2_2;
	
	// in1_1 = in1_1 / fabs(max);
	// in1_2 = in1_2 / fabs(max);
	// in2_2 = in2_2 / fabs(max);
	
	// angulo1=calculaAngulo(in1_1,in1_2,in2_2);
	
	// printf("PD: Se calcula el angulo NORMALIZADO de \t\t: %e  ,  %e   ,   %e  -> %e \n",in1_1,in1_2,in2_2,angulo1);
	
	// if(fabs(angulo-angulo1)>10e-7){
		// printf("---------------------------------------------------->>>\n");		
	// }
	
	
	////////////////////////////

	ROTACION(angulo,in1_1,in1_2,in1_2,in2_2,out1_1,out1_2,&kk,out2_2);
	ROTACION(angulo,*out1_1,kk,*out1_2,*out2_2,out1_1,out1_2,&kk,out2_2);

	//forzamos los no diagonles a cero
	*out1_2=0;

	return angulo;

}

__device__ float PDf(float in1_1,float in1_2,float in2_2,float *out1_1,float *out1_2,float *out2_2){
	
	float angulo;

	float kk;

	angulo=calculaAngulof(in1_1,in1_2,in2_2);
	
	////////////////////////////

	ROTACIONf(angulo,in1_1,in1_2,in1_2,in2_2,out1_1,out1_2,&kk,out2_2);
	ROTACIONf(angulo,*out1_1,kk,*out1_2,*out2_2,out1_1,out1_2,&kk,out2_2);

	//forzamos los no diagonles a cero
	*out1_2=0;

	return angulo;

}

__device__ void PND(PRECISION angulofila,PRECISION angulocolumna,PRECISION in1_1,PRECISION in1_2,PRECISION in2_1,PRECISION in2_2,PRECISION *out1_1,PRECISION *out1_2,PRECISION *out2_1,PRECISION *out2_2){

	PRECISION aux;
	
	ROTACION(angulofila,in1_1,in1_2,in2_1,in2_2,out1_1,out1_2,out2_1,out2_2);
	ROTACION(angulocolumna,*out1_1,*out2_1,*out1_2,*out2_2,out1_1,out1_2,out2_1,out2_2);

	aux=*out1_2;
	*out1_2=*out2_1;
	*out2_1=aux;
		
}


__device__ void PNDf(float angulofila,float angulocolumna,float in1_1,float in1_2,float in2_1,float in2_2,float *out1_1,float *out1_2,float *out2_1,float *out2_2){

	float aux;
	
	ROTACIONf(angulofila,in1_1,in1_2,in2_1,in2_2,out1_1,out1_2,out2_1,out2_2);
	ROTACIONf(angulocolumna,*out1_1,*out2_1,*out1_2,*out2_2,out1_1,out1_2,out2_1,out2_2);

	aux=*out1_2;
	*out1_2=*out2_1;
	*out2_1=aux;
		
}

__device__ void PV(PRECISION angulo,PRECISION in1_1,PRECISION in1_2,PRECISION in2_1,PRECISION in2_2,PRECISION* out1_1,PRECISION* out1_2,PRECISION* out2_1,PRECISION* out2_2){

	PRECISION kk1,kk2,kk3,kk4;
	
	ROTACION(angulo,in1_1,in2_1,in1_2,in2_2,&kk1,&kk2,&kk3,&kk4);

	*out1_1 = kk1;
	*out1_2 = kk3;
	*out2_1 = kk2;
	*out2_2 = kk4;
}
__device__ void PVf(float angulo,float in1_1,float in1_2,float in2_1,float in2_2,float* out1_1,float* out1_2,float* out2_1,float* out2_2){

	//float kk1,kk2,kk3,kk4;
	
	//ROTACIONf(angulo,in1_1,in2_1,in1_2,in2_2,&kk1,&kk2,&kk3,&kk4);
	ROTACIONf(angulo,in1_1,in2_1,in1_2,in2_2,out1_1,out2_1,out1_2,out2_2);

	/**out1_1 = kk1;
	*out1_2 = kk3;
	*out2_1 = kk2;
	*out2_2 = kk4;*/
}

__device__ void InicializarMatrizIdentidad(PRECISION A[TAMANIO_SVD][TAMANIO_SVD],int tamanio){
	int i=0,j;

	for(i=0;i<tamanio;i++){
		for(j=0;j<tamanio;j++)
			A[i][j] = 0;			
	}
	
	for(i=0;i<tamanio;i++){
			A[i][i] = 1;
	}
}

__device__ void InicializarMatrizIdentidad2(PRECISION A[TAMANIO_SVD * TAMANIO_SVD],int tamanio){
	int i=0,j;

	
	#pragma unroll
	for(i=0;i<tamanio*tamanio;i++){
		A[i] = 0;
	}
	#pragma unroll
	for (j = 0; j < tamanio; j++)
	{
		i = j * (tamanio + 1);
		A[i] = 1;
	}	
}

__device__ void InicializarMatrizIdentidadf(float A[TAMANIO_SVD * TAMANIO_SVD],int tamanio){
	int i=0,j;

	
	#pragma unroll
	for(i=0;i<tamanio*tamanio;i++){
		A[i] = 0;
	}
	#pragma unroll
	for (j = 0; j < tamanio; j++)
	{
		i = j * (tamanio + 1);
		A[i] = 1;
	}	
}

__device__ void InicializarMatrizCeros(PRECISION A[TAMANIO_SVD][TAMANIO_SVD],int tamanio){
	int i=0,j;


	for(i=0;i<tamanio;i++){
		for(j=0;j<tamanio;j++)
			A[i][j]=0;
	}
	
}

__device__ void InicializarMatrizCerosf(float A[TAMANIO_SVD][TAMANIO_SVD],int tamanio){
	int i=0,j;


	for(i=0;i<tamanio;i++){
		for(j=0;j<tamanio;j++)
			A[i][j]=0;
	}
	
}


__device__ void InicializarMatrizCeros2(PRECISION A[TAMANIO_SVD*TAMANIO_SVD],int tamanio){
	int i;
	for(i=0;i<tamanio*tamanio;i++){
		A[i] = 0;
	}
	
}

__device__ void svdcordic(PRECISION *a, int m, int n, PRECISION w[TAMANIO_SVD], PRECISION v[TAMANIO_SVD * TAMANIO_SVD],int max_iter)
{

	int ind_f,ind_c;
	int iter;
	int tamanio=TAMANIO_SVD;

	PRECISION angulos[(int)TAMANIO_SVD/2];
	//PRECISION* a_out=(PRECISION*)calloc(TAMANIO_SVD*TAMANIO_SVD,sizeof(PRECISION));
	
	PRECISION a_out[TAMANIO_SVD*TAMANIO_SVD]; 
	


	//PRECISION autovectores[TAMANIO_SVD][TAMANIO_SVD];
	PRECISION autovectores_out[TAMANIO_SVD][TAMANIO_SVD];

	//PRECISION autovectores [TAMANIO_SVD * TAMANIO_SVD];
	//PRECISION autovectores_out[TAMANIO_SVD * TAMANIO_SVD];


	//InicializarMatrizIdentidad(autovectores,TAMANIO_SVD);
	//InicializarMatrizIdentidad2(autovectores,TAMANIO_SVD);
	InicializarMatrizIdentidad2(v,TAMANIO_SVD);
	//InicializarMatrizCeros(autovectores_out,TAMANIO_SVD);


	PRECISION kk1,kk2,kk3,kk4;

	for(iter=0;iter<max_iter;iter++){

		/*angulos[0]=PD(*(a+tamanio*0+0),*(a+tamanio*0+1),*(a+tamanio*1+1),
						(a_out+tamanio*0+0),(a_out+tamanio*0+2),(a_out+tamanio*2+2));

		angulos[1]=PD(*(a+tamanio*2+2),*(a+tamanio*2+3),*(a+tamanio*3+3),
						(a_out+tamanio*4+4),(a_out+tamanio*1+4),(a_out+tamanio*1+1));

		angulos[2]=PD(*(a+tamanio*4+4),*(a+tamanio*4+5),*(a+tamanio*5+5),
						(a_out+tamanio*6+6),(a_out+tamanio*3+6),(a_out+tamanio*3+3));


		angulos[3]=PD(*(a+tamanio*6+6),*(a+tamanio*6+7),*(a+tamanio*7+7),
						(a_out+tamanio*8+8),(a_out+tamanio*5+8),(a_out+tamanio*5+5));


		angulos[4]=PD(*(a+tamanio*8+8),*(a+tamanio*8+9),*(a+tamanio*9+9),
						(a_out+tamanio*9+9),(a_out+tamanio*7+9),(a_out+tamanio*7+7));*/


		angulos[0]=PD(*(a+tamanio*0+0),*(a+tamanio*0+1),*(a+tamanio*1+1),
						(a_out+tamanio*0+0),(a_out+tamanio*0+2),(a_out+tamanio*2+2));

		angulos[1]=PD(*(a+tamanio*2+2),*(a+tamanio*2+3),*(a+tamanio*3+3),
						(a_out+tamanio*4+4),(a_out+tamanio*1+4),(a_out+tamanio*1+1));

		angulos[2]=PD(*(a+tamanio*4+4),*(a+tamanio*4+5),*(a+tamanio*5+5),
						(a_out+tamanio*6+6),(a_out+tamanio*3+6),(a_out+tamanio*3+3));


		angulos[3]=PD(*(a+tamanio*6+6),*(a+tamanio*6+7),*(a+tamanio*7+7),
						(a_out+tamanio*8+8),(a_out+tamanio*5+8),(a_out+tamanio*5+5));


		angulos[4]=PD(*(a+tamanio*8+8),*(a+tamanio*8+9),*(a+tamanio*9+9),
						(a_out+tamanio*9+9),(a_out+tamanio*7+9),(a_out+tamanio*7+7));

		
		//PND for filas
		//fila 0
		PND(angulos[0],angulos[1],*(a+tamanio*0+2),*(a+tamanio*0+3),*(a+tamanio*1+2),*(a+tamanio*1+3),
						(a_out+tamanio*0+4),(a_out+tamanio*0+1),(a_out+tamanio*2+4),(a_out+tamanio*1+2));

		PND(angulos[0],angulos[2],*(a+tamanio*0+4),*(a+tamanio*0+5),*(a+tamanio*1+4),*(a+tamanio*1+5),
						(a_out+tamanio*0+6),(a_out+tamanio*0+3),(a_out+tamanio*2+6),(a_out+tamanio*2+3));

		PND(angulos[0],angulos[3],*(a+tamanio*0+6),*(a+tamanio*0+7),*(a+tamanio*1+6),*(a+tamanio*1+7),
						(a_out+tamanio*0+8),(a_out+tamanio*0+5),(a_out+tamanio*2+8),(a_out+tamanio*2+5));

		PND(angulos[0],angulos[4],*(a+tamanio*0+8),*(a+tamanio*0+9),*(a+tamanio*1+8),*(a+tamanio*1+9),
						(a_out+tamanio*0+9),(a_out+tamanio*0+7),(a_out+tamanio*2+9),(a_out+tamanio*2+7));


		//fila 1
		PND(angulos[1],angulos[2],*(a+tamanio*2+4),*(a+tamanio*2+5),*(a+tamanio*3+4),*(a+tamanio*3+5),
						(a_out+tamanio*4+6),(a_out+tamanio*3+4),(a_out+tamanio*1+6),(a_out+tamanio*1+3));

		PND(angulos[1],angulos[3],*(a+tamanio*2+6),*(a+tamanio*2+7),*(a+tamanio*3+6),*(a+tamanio*3+7),
						(a_out+tamanio*4+8),(a_out+tamanio*4+5),(a_out+tamanio*1+8),(a_out+tamanio*1+5));

		PND(angulos[1],angulos[4],*(a+tamanio*2+8),*(a+tamanio*2+9),*(a+tamanio*3+8),*(a+tamanio*3+9),
						(a_out+tamanio*4+9),(a_out+tamanio*4+7),(a_out+tamanio*1+9),(a_out+tamanio*1+7));

		//fila 2
		PND(angulos[2],angulos[3],*(a+tamanio*4+6),*(a+tamanio*4+7),*(a+tamanio*5+6),*(a+tamanio*5+7),
						(a_out+tamanio*6+8),(a_out+tamanio*5+6),(a_out+tamanio*3+8),(a_out+tamanio*3+5));

		PND(angulos[2],angulos[4],*(a+tamanio*4+8),*(a+tamanio*4+9),*(a+tamanio*5+8),*(a+tamanio*5+9),
						(a_out+tamanio*6+9),(a_out+tamanio*6+7),(a_out+tamanio*3+9),(a_out+tamanio*3+7));

		//fila 3
		PND(angulos[3],angulos[4],*(a+tamanio*6+8),*(a+tamanio*6+9),*(a+tamanio*7+8),*(a+tamanio*7+9),
						(a_out+tamanio*8+9),(a_out+tamanio*7+8),(a_out+tamanio*5+9),(a_out+tamanio*5+7));
				

		//-----------------------------------------------------------------------------

		//PV por columnas		
		int ind_angulo;
		ind_angulo=0;
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c=ind_c+2){			
			for(ind_f=0;ind_f<TAMANIO_SVD;ind_f=ind_f+2){

				/*PV(angulos[ind_angulo],autovectores[ind_f][ind_c],autovectores[ind_f][ind_c+1],
											 autovectores[ind_f+1][ind_c],autovectores[ind_f+1][ind_c+1],
													&kk1,&kk2,&kk3,&kk4);

				PV(angulos[ind_angulo],autovectores[ind_f][ind_c],autovectores[ind_f][ind_c+1],
					autovectores[ind_f+1][ind_c],autovectores[ind_f+1][ind_c+1],
							&kk1,&kk2,&kk3,&kk4);*/
		   
				//PV(angulos[ind_angulo],autovectores[(ind_f*TAMANIO_SVD) + ind_c],autovectores[ (ind_f*TAMANIO_SVD) + (ind_c+1)],autovectores[((ind_f+1)*TAMANIO_SVD) + ind_c],autovectores[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);
				PV(angulos[ind_angulo],v[(ind_f*TAMANIO_SVD) + ind_c],v[ (ind_f*TAMANIO_SVD) + (ind_c+1)],v[((ind_f+1)*TAMANIO_SVD) + ind_c],v[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);

				//PV(angulos[ind_angulo],autovectores[(ind_f*TAMANIO_SVD) + ind_c],autovectores[ (ind_f*TAMANIO_SVD) + (ind_c+1)],autovectores[((ind_f+1)*TAMANIO_SVD) + ind_c],autovectores[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);

				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ (ind_f*TAMANIO_SVD)+ (ind_c)]]=kk1;								
				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + ( (ind_c+1))]] [cordicPosCol[ (ind_f*TAMANIO_SVD) + ((ind_c+1) ) ]]=kk2;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]]=kk3;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]][cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]]=kk4;				

			}
			ind_angulo++;
		}

	
		

		//copiar a_out en a
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				a[ind_f*TAMANIO_SVD+ind_c] = a_out[ind_f*TAMANIO_SVD+ind_c];
			}
		}

		//copiar autovectores_aux en autovectores
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				//autovectores[ind_f][ind_c] = autovectores_out[ind_f][ind_c];
				//autovectores[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
				v[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
			}
		}

	}  //end for max_iter


	//copiar autovectores en v
	/*for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			//v[ind_f][ind_c] = autovectores[ind_f][ind_c];
			//v[ind_c + (ind_f*TAMANIO_SVD)] = autovectores[ind_f] [ind_c];
			v[ind_c + (ind_f*TAMANIO_SVD)] = autovectores[ (ind_f*TAMANIO_SVD) + ind_c];
		}
	}*/
	#pragma unroll
	for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			w[ind_c]=a[ind_c*TAMANIO_SVD+ind_c];
	}


	
	//free(a_out);	


}




__device__ void svdcordicf(float *a, int m, int n, float w[TAMANIO_SVD], float v[TAMANIO_SVD * TAMANIO_SVD],int max_iter)
{

	int ind_f,ind_c;
	int iter;
	int tamanio=TAMANIO_SVD;

	float angulos[(int)TAMANIO_SVD/2];
	float a_out[TAMANIO_SVD*TAMANIO_SVD]; 
	


	//float autovectores[TAMANIO_SVD * TAMANIO_SVD];
	float autovectores_out[TAMANIO_SVD][TAMANIO_SVD];

	//InicializarMatrizIdentidadf(autovectores,TAMANIO_SVD);
	InicializarMatrizIdentidadf(v,TAMANIO_SVD);

	float kk1,kk2,kk3,kk4;

	for(iter=0;iter<max_iter;iter++){

		angulos[0]=PDf(*(a+tamanio*0+0),*(a+tamanio*0+1),*(a+tamanio*1+1),
						(a_out+tamanio*0+0),(a_out+tamanio*0+2),(a_out+tamanio*2+2));

		angulos[1]=PDf(*(a+tamanio*2+2),*(a+tamanio*2+3),*(a+tamanio*3+3),
						(a_out+tamanio*4+4),(a_out+tamanio*1+4),(a_out+tamanio*1+1));

		angulos[2]=PDf(*(a+tamanio*4+4),*(a+tamanio*4+5),*(a+tamanio*5+5),
						(a_out+tamanio*6+6),(a_out+tamanio*3+6),(a_out+tamanio*3+3));


		angulos[3]=PDf(*(a+tamanio*6+6),*(a+tamanio*6+7),*(a+tamanio*7+7),
						(a_out+tamanio*8+8),(a_out+tamanio*5+8),(a_out+tamanio*5+5));


		angulos[4]=PDf(*(a+tamanio*8+8),*(a+tamanio*8+9),*(a+tamanio*9+9),
						(a_out+tamanio*9+9),(a_out+tamanio*7+9),(a_out+tamanio*7+7));

		
		//PND for filas
		//fila 0
		PNDf(angulos[0],angulos[1],*(a+tamanio*0+2),*(a+tamanio*0+3),*(a+tamanio*1+2),*(a+tamanio*1+3),
						(a_out+tamanio*0+4),(a_out+tamanio*0+1),(a_out+tamanio*2+4),(a_out+tamanio*1+2));

		PNDf(angulos[0],angulos[2],*(a+tamanio*0+4),*(a+tamanio*0+5),*(a+tamanio*1+4),*(a+tamanio*1+5),
						(a_out+tamanio*0+6),(a_out+tamanio*0+3),(a_out+tamanio*2+6),(a_out+tamanio*2+3));

		PNDf(angulos[0],angulos[3],*(a+tamanio*0+6),*(a+tamanio*0+7),*(a+tamanio*1+6),*(a+tamanio*1+7),
						(a_out+tamanio*0+8),(a_out+tamanio*0+5),(a_out+tamanio*2+8),(a_out+tamanio*2+5));

		PNDf(angulos[0],angulos[4],*(a+tamanio*0+8),*(a+tamanio*0+9),*(a+tamanio*1+8),*(a+tamanio*1+9),
						(a_out+tamanio*0+9),(a_out+tamanio*0+7),(a_out+tamanio*2+9),(a_out+tamanio*2+7));


		//fila 1
		PNDf(angulos[1],angulos[2],*(a+tamanio*2+4),*(a+tamanio*2+5),*(a+tamanio*3+4),*(a+tamanio*3+5),
						(a_out+tamanio*4+6),(a_out+tamanio*3+4),(a_out+tamanio*1+6),(a_out+tamanio*1+3));

		PNDf(angulos[1],angulos[3],*(a+tamanio*2+6),*(a+tamanio*2+7),*(a+tamanio*3+6),*(a+tamanio*3+7),
						(a_out+tamanio*4+8),(a_out+tamanio*4+5),(a_out+tamanio*1+8),(a_out+tamanio*1+5));

		PNDf(angulos[1],angulos[4],*(a+tamanio*2+8),*(a+tamanio*2+9),*(a+tamanio*3+8),*(a+tamanio*3+9),
						(a_out+tamanio*4+9),(a_out+tamanio*4+7),(a_out+tamanio*1+9),(a_out+tamanio*1+7));

		//fila 2
		PNDf(angulos[2],angulos[3],*(a+tamanio*4+6),*(a+tamanio*4+7),*(a+tamanio*5+6),*(a+tamanio*5+7),
						(a_out+tamanio*6+8),(a_out+tamanio*5+6),(a_out+tamanio*3+8),(a_out+tamanio*3+5));

		PNDf(angulos[2],angulos[4],*(a+tamanio*4+8),*(a+tamanio*4+9),*(a+tamanio*5+8),*(a+tamanio*5+9),
						(a_out+tamanio*6+9),(a_out+tamanio*6+7),(a_out+tamanio*3+9),(a_out+tamanio*3+7));

		//fila 3
		PNDf(angulos[3],angulos[4],*(a+tamanio*6+8),*(a+tamanio*6+9),*(a+tamanio*7+8),*(a+tamanio*7+9),
						(a_out+tamanio*8+9),(a_out+tamanio*7+8),(a_out+tamanio*5+9),(a_out+tamanio*5+7));
				

		//-----------------------------------------------------------------------------

		//PV por columnas		
		int ind_angulo;
		ind_angulo=0;
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c=ind_c+2){			
			for(ind_f=0;ind_f<TAMANIO_SVD;ind_f=ind_f+2){
				//PVf(angulos[ind_angulo],autovectores[(ind_f*TAMANIO_SVD) + ind_c],autovectores[ (ind_f*TAMANIO_SVD) + (ind_c+1)],autovectores[((ind_f+1)*TAMANIO_SVD) + ind_c],autovectores[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);
				PVf(angulos[ind_angulo],v[(ind_f*TAMANIO_SVD) + ind_c],v[ (ind_f*TAMANIO_SVD) + (ind_c+1)],v[((ind_f+1)*TAMANIO_SVD) + ind_c],v[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);
				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ (ind_f*TAMANIO_SVD)+ (ind_c)]]=kk1;								
				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + ( (ind_c+1))]] [cordicPosCol[ (ind_f*TAMANIO_SVD) + ((ind_c+1) ) ]]=kk2;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]]=kk3;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]][cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]]=kk4;				

			}
			ind_angulo++;
		}

		//copiar a_out en a
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			//#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				a[ind_f*TAMANIO_SVD+ind_c] = a_out[ind_f*TAMANIO_SVD+ind_c];
			}
		}

		//copiar autovectores_aux en autovectores
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			//#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				
				//autovectores[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
				v[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
			}
		}

	}  //end for max_iter

	//copiar autovectores en v
	/*for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			v[ind_c + (ind_f*TAMANIO_SVD)] = autovectores[ (ind_f*TAMANIO_SVD) + ind_c];
		}
	}*/
	//#pragma unroll
	for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			w[ind_c]=a[ind_c*TAMANIO_SVD+ind_c];
	}

	//free(a_out);	


}



/*__device__ void svdcordicf2(float *a, int m, int n, float w[TAMANIO_SVD], float v[TAMANIO_SVD * TAMANIO_SVD],int max_iter)
{

	int ind_f,ind_c;
	int iter;
	int tamanio=TAMANIO_SVD;

	float angulos[(int)TAMANIO_SVD/2];
	float a_out[TAMANIO_SVD*TAMANIO_SVD]; 

	float autovectores[TAMANIO_SVD][TAMANIO_SVD];
	float autovectores_out[TAMANIO_SVD][TAMANIO_SVD];

	//InicializarMatrizIdentidadf(autovectores,TAMANIO_SVD);
	InicializarMatrizIdentidadf(autovectores,TAMANIO_SVD);
	InicializarMatrizCeros(autovectores_out,TAMANIO_SVD);

	float kk1,kk2,kk3,kk4;

	for(iter=0;iter<max_iter;iter++){

		angulos[0]=PDf(*(a+tamanio*0+0),*(a+tamanio*0+1),*(a+tamanio*1+1),
						(a_out+tamanio*0+0),(a_out+tamanio*0+2),(a_out+tamanio*2+2));

		angulos[1]=PDf(*(a+tamanio*2+2),*(a+tamanio*2+3),*(a+tamanio*3+3),
						(a_out+tamanio*4+4),(a_out+tamanio*1+4),(a_out+tamanio*1+1));

		angulos[2]=PDf(*(a+tamanio*4+4),*(a+tamanio*4+5),*(a+tamanio*5+5),
						(a_out+tamanio*6+6),(a_out+tamanio*3+6),(a_out+tamanio*3+3));


		angulos[3]=PDf(*(a+tamanio*6+6),*(a+tamanio*6+7),*(a+tamanio*7+7),
						(a_out+tamanio*8+8),(a_out+tamanio*5+8),(a_out+tamanio*5+5));


		angulos[4]=PDf(*(a+tamanio*8+8),*(a+tamanio*8+9),*(a+tamanio*9+9),
						(a_out+tamanio*9+9),(a_out+tamanio*7+9),(a_out+tamanio*7+7));

		
		//PND for filas
		//fila 0
		PNDf(angulos[0],angulos[1],*(a+tamanio*0+2),*(a+tamanio*0+3),*(a+tamanio*1+2),*(a+tamanio*1+3),
						(a_out+tamanio*0+4),(a_out+tamanio*0+1),(a_out+tamanio*2+4),(a_out+tamanio*1+2));

		PNDf(angulos[0],angulos[2],*(a+tamanio*0+4),*(a+tamanio*0+5),*(a+tamanio*1+4),*(a+tamanio*1+5),
						(a_out+tamanio*0+6),(a_out+tamanio*0+3),(a_out+tamanio*2+6),(a_out+tamanio*2+3));

		PNDf(angulos[0],angulos[3],*(a+tamanio*0+6),*(a+tamanio*0+7),*(a+tamanio*1+6),*(a+tamanio*1+7),
						(a_out+tamanio*0+8),(a_out+tamanio*0+5),(a_out+tamanio*2+8),(a_out+tamanio*2+5));

		PNDf(angulos[0],angulos[4],*(a+tamanio*0+8),*(a+tamanio*0+9),*(a+tamanio*1+8),*(a+tamanio*1+9),
						(a_out+tamanio*0+9),(a_out+tamanio*0+7),(a_out+tamanio*2+9),(a_out+tamanio*2+7));


		//fila 1
		PNDf(angulos[1],angulos[2],*(a+tamanio*2+4),*(a+tamanio*2+5),*(a+tamanio*3+4),*(a+tamanio*3+5),
						(a_out+tamanio*4+6),(a_out+tamanio*3+4),(a_out+tamanio*1+6),(a_out+tamanio*1+3));

		PNDf(angulos[1],angulos[3],*(a+tamanio*2+6),*(a+tamanio*2+7),*(a+tamanio*3+6),*(a+tamanio*3+7),
						(a_out+tamanio*4+8),(a_out+tamanio*4+5),(a_out+tamanio*1+8),(a_out+tamanio*1+5));

		PNDf(angulos[1],angulos[4],*(a+tamanio*2+8),*(a+tamanio*2+9),*(a+tamanio*3+8),*(a+tamanio*3+9),
						(a_out+tamanio*4+9),(a_out+tamanio*4+7),(a_out+tamanio*1+9),(a_out+tamanio*1+7));

		//fila 2
		PNDf(angulos[2],angulos[3],*(a+tamanio*4+6),*(a+tamanio*4+7),*(a+tamanio*5+6),*(a+tamanio*5+7),
						(a_out+tamanio*6+8),(a_out+tamanio*5+6),(a_out+tamanio*3+8),(a_out+tamanio*3+5));

		PNDf(angulos[2],angulos[4],*(a+tamanio*4+8),*(a+tamanio*4+9),*(a+tamanio*5+8),*(a+tamanio*5+9),
						(a_out+tamanio*6+9),(a_out+tamanio*6+7),(a_out+tamanio*3+9),(a_out+tamanio*3+7));

		//fila 3
		PNDf(angulos[3],angulos[4],*(a+tamanio*6+8),*(a+tamanio*6+9),*(a+tamanio*7+8),*(a+tamanio*7+9),
						(a_out+tamanio*8+9),(a_out+tamanio*7+8),(a_out+tamanio*5+9),(a_out+tamanio*5+7));
				

		//-----------------------------------------------------------------------------

		//PV por columnas		
		int ind_angulo;
		ind_angulo=0;
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c=ind_c+2){			
			for(ind_f=0;ind_f<TAMANIO_SVD;ind_f=ind_f+2){
				//PVf(angulos[ind_angulo],autovectores[(ind_f*TAMANIO_SVD) + ind_c],autovectores[ (ind_f*TAMANIO_SVD) + (ind_c+1)],autovectores[((ind_f+1)*TAMANIO_SVD) + ind_c],autovectores[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);
				PVf(angulos[ind_angulo],v[(ind_f*TAMANIO_SVD) + ind_c],v[ (ind_f*TAMANIO_SVD) + (ind_c+1)],v[((ind_f+1)*TAMANIO_SVD) + ind_c],v[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);
				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ (ind_f*TAMANIO_SVD)+ (ind_c)]]=kk1;								
				autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + ( (ind_c+1))]] [cordicPosCol[ (ind_f*TAMANIO_SVD) + ((ind_c+1) ) ]]=kk2;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]]=kk3;
				autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]][cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]]=kk4;				

			}
			ind_angulo++;
		}

		//copiar a_out en a
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			//#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				a[ind_f*TAMANIO_SVD+ind_c] = a_out[ind_f*TAMANIO_SVD+ind_c];
			}
		}

		//copiar autovectores_aux en autovectores
		for(ind_f=0;ind_f<TAMANIO_SVD;ind_f++){
			//#pragma unroll
			for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
				
				//autovectores[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
				v[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
			}
		}

	}  //end for max_iter


	//#pragma unroll
	for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			w[ind_c]=a[ind_c*TAMANIO_SVD+ind_c];
	}

	//free(a_out);	


}*/



/** kernel launch whith TAMANIO_SVD THREADS
*/
__global__ void svdcordic_kernel(float *a, int m, int n, float w[TAMANIO_SVD], float v[TAMANIO_SVD * TAMANIO_SVD],int max_iter)
{

	int iThread = blockIdx.x * blockDim.x + threadIdx.x;  // index columns 
	int ind_f = iThread *2;
	int ind_c;
	int iter;
	int i;

	__shared__ float angulos[(int)TAMANIO_SVD/2];
	__shared__ float a_aux [TAMANIO_SVD*TAMANIO_SVD];
	__shared__ float a_out[TAMANIO_SVD*TAMANIO_SVD]; 
	__shared__ float autovectores[TAMANIO_SVD * TAMANIO_SVD];
	__shared__ float autovectores_out[TAMANIO_SVD][TAMANIO_SVD];

	#pragma unroll
	for(i=0;i<TAMANIO_SVD;i++){
		autovectores[i+iThread*TAMANIO_SVD] = 0;
	}
	__syncthreads();
	i = iThread * (TAMANIO_SVD + 1);
	autovectores[i] = 1;

	#pragma unroll
	for(i=0;i<TAMANIO_SVD;i++){
		a_aux[i+iThread*TAMANIO_SVD] = a[i+iThread*TAMANIO_SVD];
	}
	__syncthreads();

	float kk1,kk2,kk3,kk4;

	for(iter=0;iter<max_iter;iter++){

		if(iThread==0){
			angulos[0]=PDf(*(a_aux+TAMANIO_SVD*0+0),*(a_aux+TAMANIO_SVD*0+1),*(a_aux+TAMANIO_SVD*1+1),
						(a_out+TAMANIO_SVD*0+0),(a_out+TAMANIO_SVD*0+2),(a_out+TAMANIO_SVD*2+2));
		}
		else if(iThread==1){
			angulos[1]=PDf(*(a_aux+TAMANIO_SVD*2+2),*(a_aux+TAMANIO_SVD*2+3),*(a_aux+TAMANIO_SVD*3+3),
						(a_out+TAMANIO_SVD*4+4),(a_out+TAMANIO_SVD*1+4),(a_out+TAMANIO_SVD*1+1));
		}
		else if(iThread==2){
			angulos[2]=PDf(*(a_aux+TAMANIO_SVD*4+4),*(a_aux+TAMANIO_SVD*4+5),*(a_aux+TAMANIO_SVD*5+5),
						(a_out+TAMANIO_SVD*6+6),(a_out+TAMANIO_SVD*3+6),(a_out+TAMANIO_SVD*3+3));
		}
		else if(iThread==3){
			angulos[3]=PDf(*(a_aux+TAMANIO_SVD*6+6),*(a_aux+TAMANIO_SVD*6+7),*(a_aux+TAMANIO_SVD*7+7),
						(a_out+TAMANIO_SVD*8+8),(a_out+TAMANIO_SVD*5+8),(a_out+TAMANIO_SVD*5+5));
		}
		else if(iThread==4){
			angulos[4]=PDf(*(a_aux+TAMANIO_SVD*8+8),*(a_aux+TAMANIO_SVD*8+9),*(a_aux+TAMANIO_SVD*9+9),
						(a_out+TAMANIO_SVD*9+9),(a_out+TAMANIO_SVD*7+9),(a_out+TAMANIO_SVD*7+7));
		}
		__syncthreads();
		//PND for filas
		//fila 0
		if(iThread==0){
			PNDf(angulos[0],angulos[1],*(a_aux+TAMANIO_SVD*0+2),*(a_aux+TAMANIO_SVD*0+3),*(a_aux+TAMANIO_SVD*1+2),*(a_aux+TAMANIO_SVD*1+3),
							(a_out+TAMANIO_SVD*0+4),(a_out+TAMANIO_SVD*0+1),(a_out+TAMANIO_SVD*2+4),(a_out+TAMANIO_SVD*1+2));

			PNDf(angulos[0],angulos[2],*(a_aux+TAMANIO_SVD*0+4),*(a_aux+TAMANIO_SVD*0+5),*(a_aux+TAMANIO_SVD*1+4),*(a_aux+TAMANIO_SVD*1+5),
							(a_out+TAMANIO_SVD*0+6),(a_out+TAMANIO_SVD*0+3),(a_out+TAMANIO_SVD*2+6),(a_out+TAMANIO_SVD*2+3));
			
		}
		else if(iThread==1){
			PNDf(angulos[0],angulos[3],*(a_aux+TAMANIO_SVD*0+6),*(a_aux+TAMANIO_SVD*0+7),*(a_aux+TAMANIO_SVD*1+6),*(a_aux+TAMANIO_SVD*1+7),
							(a_out+TAMANIO_SVD*0+8),(a_out+TAMANIO_SVD*0+5),(a_out+TAMANIO_SVD*2+8),(a_out+TAMANIO_SVD*2+5));

			PNDf(angulos[0],angulos[4],*(a_aux+TAMANIO_SVD*0+8),*(a_aux+TAMANIO_SVD*0+9),*(a_aux+TAMANIO_SVD*1+8),*(a_aux+TAMANIO_SVD*1+9),
							(a_out+TAMANIO_SVD*0+9),(a_out+TAMANIO_SVD*0+7),(a_out+TAMANIO_SVD*2+9),(a_out+TAMANIO_SVD*2+7));
		}
		else if(iThread==2){
		//fila 1
			PNDf(angulos[1],angulos[2],*(a_aux+TAMANIO_SVD*2+4),*(a_aux+TAMANIO_SVD*2+5),*(a_aux+TAMANIO_SVD*3+4),*(a_aux+TAMANIO_SVD*3+5),
							(a_out+TAMANIO_SVD*4+6),(a_out+TAMANIO_SVD*3+4),(a_out+TAMANIO_SVD*1+6),(a_out+TAMANIO_SVD*1+3));

			PNDf(angulos[1],angulos[3],*(a_aux+TAMANIO_SVD*2+6),*(a_aux+TAMANIO_SVD*2+7),*(a_aux+TAMANIO_SVD*3+6),*(a_aux+TAMANIO_SVD*3+7),
							(a_out+TAMANIO_SVD*4+8),(a_out+TAMANIO_SVD*4+5),(a_out+TAMANIO_SVD*1+8),(a_out+TAMANIO_SVD*1+5));
		}
		else if(iThread==3){
			PNDf(angulos[1],angulos[4],*(a_aux+TAMANIO_SVD*2+8),*(a_aux+TAMANIO_SVD*2+9),*(a_aux+TAMANIO_SVD*3+8),*(a_aux+TAMANIO_SVD*3+9),
							(a_out+TAMANIO_SVD*4+9),(a_out+TAMANIO_SVD*4+7),(a_out+TAMANIO_SVD*1+9),(a_out+TAMANIO_SVD*1+7));

			//fila 2
			PNDf(angulos[2],angulos[3],*(a_aux+TAMANIO_SVD*4+6),*(a_aux+TAMANIO_SVD*4+7),*(a_aux+TAMANIO_SVD*5+6),*(a_aux+TAMANIO_SVD*5+7),
							(a_out+TAMANIO_SVD*6+8),(a_out+TAMANIO_SVD*5+6),(a_out+TAMANIO_SVD*3+8),(a_out+TAMANIO_SVD*3+5));
		}
		else if(iThread==4){
			PNDf(angulos[2],angulos[4],*(a_aux+TAMANIO_SVD*4+8),*(a_aux+TAMANIO_SVD*4+9),*(a_aux+TAMANIO_SVD*5+8),*(a_aux+TAMANIO_SVD*5+9),
							(a_out+TAMANIO_SVD*6+9),(a_out+TAMANIO_SVD*6+7),(a_out+TAMANIO_SVD*3+9),(a_out+TAMANIO_SVD*3+7));

			//fila 3
			PNDf(angulos[3],angulos[4],*(a_aux+TAMANIO_SVD*6+8),*(a_aux+TAMANIO_SVD*6+9),*(a_aux+TAMANIO_SVD*7+8),*(a_aux+TAMANIO_SVD*7+9),
							(a_out+TAMANIO_SVD*8+9),(a_out+TAMANIO_SVD*7+8),(a_out+TAMANIO_SVD*5+9),(a_out+TAMANIO_SVD*5+7));
		}
		__syncthreads();
		//-----------------------------------------------------------------------------

		//PV por columnas		
		int ind_angulo;
		ind_angulo=0;
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c=ind_c+2){			
			
			PVf(angulos[ind_angulo],autovectores[(ind_f*TAMANIO_SVD) + ind_c],autovectores[ (ind_f*TAMANIO_SVD) + (ind_c+1)],autovectores[((ind_f+1)*TAMANIO_SVD) + ind_c],autovectores[ ((ind_f+1)*TAMANIO_SVD) + (ind_c+1)],	&kk1,&kk2,&kk3,&kk4);

			autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ (ind_f*TAMANIO_SVD)+ (ind_c)]]=kk1;								
			autovectores_out[cordicPosFila[ (ind_f*TAMANIO_SVD) + ( (ind_c+1))]] [cordicPosCol[ (ind_f*TAMANIO_SVD) + ((ind_c+1) ) ]]=kk2;
			autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]] [cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + (ind_c)]]=kk3;
			autovectores_out[cordicPosFila[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]][cordicPosCol[ ((ind_f+1)*TAMANIO_SVD) + ((ind_c+1))]]=kk4;				

			
			ind_angulo++;
		}

		__syncthreads();
		

		//copiar a_out en a
		
		#pragma unroll
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			a_aux[ind_f*TAMANIO_SVD+ind_c] = a_out[ind_f*TAMANIO_SVD+ind_c];
			a_aux[((ind_f+1)*TAMANIO_SVD)+ind_c] = a_out[((ind_f+1)*TAMANIO_SVD)+ind_c];
		}
		

		//copiar autovectores_aux en autovectores

		#pragma unroll
		for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			autovectores[ (ind_f*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f][ind_c];
			autovectores[ ((ind_f+1)*TAMANIO_SVD) + ind_c] = autovectores_out[ind_f+1][ind_c];
		}

		__syncthreads();
	}  //end for max_iter


	//copiar autovectores en v
	
	for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
		v[ind_c + (ind_f*TAMANIO_SVD)] = autovectores[ (ind_f*TAMANIO_SVD) + ind_c];
		v[ind_c + ((ind_f+1)*TAMANIO_SVD)] = autovectores[ ((ind_f+1)*TAMANIO_SVD) + ind_c];
	}
	
	w[ind_f]=a_aux[ind_f*TAMANIO_SVD+ind_f];
	w[ind_f+1]=a_aux[((ind_f+1)*TAMANIO_SVD)+(ind_f+1)];

	/*#pragma unroll
	for(ind_c=0;ind_c<TAMANIO_SVD;ind_c++){
			w[ind_c]=a[ind_c*TAMANIO_SVD+ind_c];
	}*/

}