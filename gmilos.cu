
//    _______             _______ _________ _        _______  _______
//   (  ____ \           (       )\__   __/( \      (  ___  )(  ____ \
//   | (    \/           | () () |   ) (   | (      | (   ) || (    \/
//   | |         _____   | || || |   | |   | |      | |   | || (_____
//   | |        (_____)  | |(_)| |   | |   | |      | |   | |(_____  )
//   | |                 | |   | |   | |   | |      | |   | |      ) |
//   | (____/\           | )   ( |___) (___| (____/\| (___) |/\____) |
//   (_______/           |/     \|\_______/(_______/(_______)\_______)
//
//
// CMILOS v1.0 (2020)
// RTE INVERSION C code (based on the ILD code MILOS by D. Orozco)
// Manuel (IAA-CSIC)
//

/*
;      eta0 = line-to-continuum absorption coefficient ratio
;      B = magnetic field strength       [Gauss]
;      vlos = line-of-sight velocity     [km/s]
;      dopp = Doppler width              [Angstroms]
;      aa = damping parameter
;      gm = magnetic field inclination   [deg]
;      az = magnetic field azimuth       [deg]
;      S0 = source function constant
;      S1 = source function gradient
;      mac = macroturbulent velocity     [km/s]
;      alpha = filling factor of the magnetic component [0->1]

*/

#include <assert.h>
#include <time.h>
#include "defines.h"
#include "definesCuda.cuh"
#include <string.h>
#include <stdio.h>
#include "fitsio.h"
#include "utilsFits.h"
#include "milosUtils.cuh"
#include "lib.cuh"
#include "readConfig.h"
#include <unistd.h>
#include <complex.h>
#include "interpolLineal.h"
// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_cuda.h"


__constant__ int d_fix_const[11];
__constant__ PRECISION d_lambda_const  [MAX_LAMBDA];
__constant__ PRECISION d_psfFunction_const  [MAX_LAMBDA];
__constant__ REAL d_weight_const [4];
__constant__ REAL d_weight_sigma_const [4];
__constant__ PRECISION d_wlines_const [2];
__constant__ Cuantic d_cuantic_const;
__constant__ Init_Model d_initModel_const;
__constant__ int d_nlambda_const;
__constant__ PRECISION d_toplim_const;
__constant__ int d_miter_const;
__constant__ REAL d_sigma_const;
__constant__ REAL d_ilambda_const;
__constant__ int d_use_convolution_const;
__constant__ REAL d_ah_const;
__constant__ int d_logclambda_const;
__constant__ cuDoubleComplex d_zdenV[7];
__constant__ cuDoubleComplex d_zdivV[7];
__constant__ int cordicPosFila[TAMANIO_SVD*TAMANIO_SVD];
__constant__ int cordicPosCol[TAMANIO_SVD*TAMANIO_SVD];
__constant__ PRECISION LIMITE_INFERIOR_PRECISION_SVD;
__constant__ PRECISION LIMITE_INFERIOR_PRECISION_TRIG;
__constant__ PRECISION LIMITE_INFERIOR_PRECISION_SINCOS;


__global__ void kernel_synthesis(Cuantic *cuantic,Init_Model *initModel,PRECISION * wlines,int nlambda,REAL *spectra,REAL * d_spectra, REAL  ah,REAL * slight, REAL * spectra_mc,REAL * spectra_slight, int  filter, int * fix){

	int i;
	
	ProfilesMemory * pM = (ProfilesMemory *) malloc(sizeof(ProfilesMemory));
	InitProfilesMemoryFromDevice(nlambda,pM,d_cuantic_const);
	REAL cosi,sinis,sina,cosa, sinda, cosda, sindi, cosdi,cosis_2;
	int uuGlobal,FGlobal,HGlobal;
	mil_sinrf(d_cuantic_const, &d_initModel_const, d_wlines_const, d_nlambda_const, spectra, d_ah_const, slight, spectra_mc, spectra_slight, filter, pM,&cosi,&sinis,&sina,&cosa,&sinda,&cosda,&sindi,&cosdi,&cosis_2,&uuGlobal,&FGlobal,&HGlobal);
	me_der(&d_cuantic_const, &d_initModel_const, d_wlines_const, d_nlambda_const, d_spectra, spectra_mc, spectra_slight, d_ah_const, slight, filter,pM, fix,cosi,sinis,sina,cosa,sinda,cosda,sindi,cosdi,cosis_2,&uuGlobal,&FGlobal,&HGlobal);
	FreeProfilesMemoryFromDevice(pM,d_cuantic_const);
	free(pM);

}

int main(int argc, char **argv)
{

	ConfigControl configCrontrolFile;
	Cuantic *cuantic,* d_Cuantic; 
	PRECISION * d_vlambda, *d_wlines;
	PRECISION * psfFunction = NULL, * d_psfFunction;
	int i,j; // for indexes
	PRECISION *wlines;
	int nlambda;
	Init_Model * h_vModels, * d_vModels,* d_initModel;
	float chisqrf, * h_vChisqrf, * d_vChisqrf;
	int * h_vNumIter, * d_vNumIter,* d_fix; // to store the number of iterations used to converge for each pixel
	int * d_displsSpectro, * d_sendCountPixels, * d_displsPixels;
	int indexLine; // index to identify central line to read it 
	REAL * h_spectra, * d_spectra,* d_spectra2, *d_weight;
	Init_Model INITIAL_MODEL;
	PRECISION * deltaLambda, * PSF;
	PRECISION initialLambda, step, finalLambda;
	int N_SAMPLES_PSF;
	int posWL=0;

	int posFila[TAMANIO_SVD * TAMANIO_SVD]={0,0,0,0,0,0,0,0,0,0,
		2,2,2,2,2,2,2,2,2,2,
		4,4,4,4,4,4,4,4,4,4,
		1,1,1,1,1,1,1,1,1,1,
		6,6,6,6,6,6,6,6,6,6,
		3,3,3,3,3,3,3,3,3,3,
		8,8,8,8,8,8,8,8,8,8,
		5,5,5,5,5,5,5,5,5,5,
		9,9,9,9,9,9,9,9,9,9,
		7,7,7,7,7,7,7,7,7,7};

	int posCol[TAMANIO_SVD * TAMANIO_SVD]={0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
		0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7,
	  0,2,4,1,6,3,8,5,9,7};


	//----------------------------------------------

	REAL * slight = NULL, * d_slight = NULL;
	int dimStrayLight;

	const char  * nameInputFileSpectra ;
	char nameOutputFilePerfiles [4096];
	const char	* nameInputFileLines;
	const char	* nameInputFilePSF ;	

    FitsImage * fitsImage;
	PRECISION  dat[7];

	/********************* Read data input from file ******************************/

	/* Read data input from file */

	loadInitialValues(&configCrontrolFile);
	readTrolFile(argv[1],&configCrontrolFile,1);
	nameInputFileSpectra = configCrontrolFile.ObservedProfiles;
	nameInputFileLines = configCrontrolFile.AtomicParametersFile;
	nameInputFilePSF = configCrontrolFile.PSFFile;
	

	cudaSetDevice(configCrontrolFile.deviceID);
	/***************** READ INIT MODEL ********************************/
	if(configCrontrolFile.InitialGuessModel[0]!='\0' && !readInitialModel(&INITIAL_MODEL,configCrontrolFile.InitialGuessModel)){
		printf("\nERROR READING GUESS MODEL 1 FILE\n");
		exit(EXIT_FAILURE);
	}
	
	/***************** READ WAVELENGHT FROM GRID OR FITS ********************************/
	PRECISION * vLambda, *vOffsetsLambda, * vLambda_wl;

	if(configCrontrolFile.useMallaGrid){ // read lambda from grid file
    	indexLine = readMallaGrid(configCrontrolFile.MallaGrid, &initialLambda, &step, &finalLambda, 1);      
    	nlambda = ((finalLambda-initialLambda)/step)+1;
		vOffsetsLambda = (PRECISION *) calloc(nlambda,sizeof(PRECISION));
		vOffsetsLambda[0] = initialLambda;
		for(i=1;i<nlambda;i++){
			vOffsetsLambda[i] = vOffsetsLambda[i-1]+step;
		}
    	// pass to armstrong 
    	initialLambda = initialLambda/1000.0;
    	step = step/1000.0;
    	finalLambda = finalLambda/1000.0;
		vLambda = (PRECISION *) calloc(nlambda,sizeof(PRECISION));
		vLambda_wl = (PRECISION *) calloc(nlambda,sizeof(PRECISION));
		configCrontrolFile.CentralWaveLenght = readFileCuanticLines(nameInputFileLines,dat,indexLine,1);
		if(configCrontrolFile.CentralWaveLenght==0){
			printf("\n QUANTUM LINE NOT FOUND, REVIEW IT. INPUT CENTRAL WAVE LENGHT: %f",configCrontrolFile.CentralWaveLenght);
			exit(1);
		}
		vLambda[0]=configCrontrolFile.CentralWaveLenght+(initialLambda);
		vLambda_wl[0] = vLambda[0] - configCrontrolFile.CentralWaveLenght;
   		for(i=1;i<nlambda;i++){
			vLambda[i]=vLambda[i-1]+step;
			vLambda_wl[i] = vLambda[i] - configCrontrolFile.CentralWaveLenght;
     	}
	}
	else{ // read lambda from fits file
		vLambda = readFitsLambdaToArray(configCrontrolFile.WavelengthFile,&indexLine,&nlambda);
		if(vLambda==NULL){
			printf("\n FILE WITH WAVELENGHT HAS NOT BEEN READ PROPERLY, please check it.\n");
			free(vLambda);
			exit(EXIT_FAILURE);
		}
		configCrontrolFile.CentralWaveLenght = readFileCuanticLines(nameInputFileLines,dat,indexLine,1);
		if(configCrontrolFile.CentralWaveLenght==0){
			printf("\n QUANTUM LINE NOT FOUND, REVIEW IT. INPUT CENTRAL WAVE LENGHT: %f",configCrontrolFile.CentralWaveLenght);
			exit(1);
		}		
	}

	/*********************************************** INITIALIZE VARIABLES  *********************************/
	
	wlines = (PRECISION *)malloc(2*sizeof(PRECISION));
	wlines[0] = 1;
	wlines[1] = configCrontrolFile.CentralWaveLenght;

	/******************* CREATE CUANTINC AND INITIALIZE DINAMYC MEMORY*******************/

	cuantic = create_cuantic(dat,1);

	cuDoubleComplex zden_h [7];
	zden_h [0] = make_cuDoubleComplex(a_fvoigt[6], 0);
	zden_h [1] = make_cuDoubleComplex(a_fvoigt[5], 0);
	zden_h [2] = make_cuDoubleComplex(a_fvoigt[4], 0);
	zden_h [3] = make_cuDoubleComplex(a_fvoigt[3], 0);
	zden_h [4] = make_cuDoubleComplex(a_fvoigt[2], 0);
	zden_h [5] = make_cuDoubleComplex(a_fvoigt[1], 0);
	zden_h [6] = make_cuDoubleComplex(a_fvoigt[0], 0);
	cudaMemcpyToSymbol(d_zdenV, zden_h, sizeof(cuDoubleComplex)*7);
	
	cuDoubleComplex zdiv_h [7];
	zdiv_h [0] = make_cuDoubleComplex(b_fvoigt[6], 0);
	zdiv_h [1]= make_cuDoubleComplex(b_fvoigt[5],0);
	zdiv_h [2]= make_cuDoubleComplex(b_fvoigt[4],0);
	zdiv_h [3]= make_cuDoubleComplex(b_fvoigt[3],0);
	zdiv_h [4]= make_cuDoubleComplex(b_fvoigt[2],0);
	zdiv_h [5]= make_cuDoubleComplex(b_fvoigt[1],0);
	zdiv_h [6] = make_cuDoubleComplex(b_fvoigt[0],0);
	cudaMemcpyToSymbol(d_zdivV, zdiv_h, sizeof(cuDoubleComplex)*7);
	
	// ********************************************* IF PSF HAS BEEN SELECTEC IN TROL READ PSF FILE OR CREATE GAUSSIAN FILTER ***********//
	if(configCrontrolFile.ConvolveWithPSF){
		
		if(configCrontrolFile.FWHM > 0){
			psfFunction = fgauss_WL(configCrontrolFile.FWHM,vLambda[1]-vLambda[0],vLambda[0],vLambda[nlambda/2],nlambda);
		}else{
			// read the number of lines 
			FILE *fp;
			char ch;
			N_SAMPLES_PSF=0;
			//open file in read more
			fp=fopen(nameInputFilePSF,"r");
			if(fp==NULL)
			{
				printf("File \"%s\" does not exist!!!\n",nameInputFilePSF);
				return 0;
			}
			//read character by character and check for new line	
			while((ch=fgetc(fp))!=EOF)
			{
				if(ch=='\n')
					N_SAMPLES_PSF++;
			}
			
			//close the file
			fclose(fp);
			if(N_SAMPLES_PSF>0){
				deltaLambda = (PRECISION * ) calloc(N_SAMPLES_PSF,sizeof(PRECISION));
				PSF = (PRECISION * ) calloc(N_SAMPLES_PSF,sizeof(PRECISION));
				readPSFFile(deltaLambda,PSF,nameInputFilePSF,configCrontrolFile.CentralWaveLenght);
				// CHECK if values of deltaLambda are in the same range of vLambda. For do that we truncate to 4 decimal places 
				if( (trunc(vOffsetsLambda[0])) < (trunc(deltaLambda[0]))  || (trunc(vOffsetsLambda[nlambda-1])) > (trunc(deltaLambda[N_SAMPLES_PSF-1])) ){
					printf("\n\n ERROR: The wavelength range given in the PSF file is smaller than the range in the mesh file [%lf,%lf] [%lf,%lf]  \n\n",deltaLambda[0],vOffsetsLambda[0],deltaLambda[N_SAMPLES_PSF-1],vOffsetsLambda[nlambda-1]);
					exit(EXIT_FAILURE);
				}
				psfFunction = (PRECISION * ) malloc(nlambda * sizeof(PRECISION));
				
				double offset=0;
				for(i=0;i<nlambda && !posWL;i++){
					if( fabs(trunc(vOffsetsLambda[i]))==0) 
						posWL = i;
				}
				if(posWL!= (nlambda/2)){ // move center to the middle of samples
					//printf("\nPOS CENTRAL WL %i",posWL);
					offset = (((nlambda/2)-posWL)*step)*1000;
					//printf ("\n OFFSET IS %f\n",offset);
				}
				interpolationLinearPSF(deltaLambda,  PSF, vOffsetsLambda ,N_SAMPLES_PSF, psfFunction, nlambda,offset);
				free(deltaLambda);
				free(PSF);
			}
			else{
				printf("\n****************** ERROR THE PSF FILE is empty or damaged.******************\n");
				exit(EXIT_FAILURE);
			}
		}
		

		cudaMemcpyToSymbol(d_psfFunction_const, psfFunction, sizeof(PRECISION)*nlambda);

	}		


	/****************************************************************************************************/
	//  IF NUMBER OF CYCLES IS LES THAN 0 THEN --> WE USE CLASSICAL ESTIMATES 
	//  IF NUMBER OF CYCLES IS 0 THEN -->  DO SYNTHESIS FROM THE INIT MODEL 
	//  IF NUMBER OF CYCLES IS GREATER THAN 0 --> READ FITS FILE OR PER FILE AND PROCESS DO INVERSION WITH N CYCLES 

	if(configCrontrolFile.NumberOfCycles<0){
		// read fits or per 
		if(strcmp(file_ext(configCrontrolFile.ObservedProfiles),PER_FILE)==0){ // invert only per file
			float * spectroPER = (float *) calloc(nlambda*NPARMS,sizeof(float));
			FILE * fReadSpectro;
			char * line = NULL;
			size_t len = 0;
			ssize_t read;
			fReadSpectro = fopen(configCrontrolFile.ObservedProfiles, "r");
			
			int contLine=0;
			if (fReadSpectro == NULL)
			{
				printf("Error opening the file of parameters, it's possible that the file doesn't exist. Please verify it. \n");
				printf("\n ******* THIS IS THE NAME OF THE FILE RECEVIED : %s \n", configCrontrolFile.ObservedProfiles);
				fclose(fReadSpectro);
				exit(EXIT_FAILURE);
			}
			float aux1, aux2,aux3,aux4,aux5,aux6;
			while ((read = getline(&line, &len, fReadSpectro)) != -1 && contLine<nlambda) {
				//sscanf(line,"%e %e %e %e %e %e",&indexLine,&dummy,&spectroPER[contLine], &spectroPER[contLine + nlambda], &spectroPER[contLine + nlambda * 2], &spectroPER[contLine + nlambda * 3]);
				sscanf(line,"%e %e %e %e %e %e",&aux1,&aux2,&aux3,&aux4,&aux5,&aux6);
				spectroPER[contLine] = aux3;
				spectroPER[contLine + nlambda] = aux4;
				spectroPER[contLine + nlambda * 2] = aux5;
				spectroPER[contLine + nlambda * 3] = aux6;
				contLine++;
			}
			fclose(fReadSpectro);
			printf("\n--------------------------------------------------------------------------------");
			printf("\nOBSERVED PROFILES FILE READ: %s ", configCrontrolFile.ObservedProfiles);
			printf("\n--------------------------------------------------------------------------------\n");
			Init_Model initModel;
			initModel.eta0 = 0;
			initModel.mac = 0;
			initModel.dopp = 0;
			initModel.aa = 0;
			initModel.alfa = 0; //0.38; //stray light factor
			initModel.S1 = 0;
			//invert with classical estimates
			estimacionesClasicas(wlines[1], vLambda, nlambda, spectroPER, &initModel,0,cuantic);
			// save model to file
			char nameAuxOutputModel [4096];
			if(configCrontrolFile.ObservedProfiles[0]!='\0')
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.ObservedProfiles));
			else
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.InitialGuessModel));
				

			strcat(nameAuxOutputModel,"_model_ce");
			strcat(nameAuxOutputModel,MOD_FILE);
			FILE * fptr = fopen(nameAuxOutputModel, "w");
			if(fptr!=NULL){
				fprintf(fptr,"eta_0               :%lf\n",initModel.eta0);
				fprintf(fptr,"magnetic field [G]  :%lf\n",initModel.B);
				fprintf(fptr,"LOS velocity[km/s]  :%lf\n",initModel.vlos);
				fprintf(fptr,"Doppler width [A]   :%lf\n",initModel.dopp);
				fprintf(fptr,"damping             :%lf\n",initModel.aa);
				fprintf(fptr,"gamma [deg]         :%lf\n",initModel.gm);
				fprintf(fptr,"phi  [deg]          :%lf\n",initModel.az);
				fprintf(fptr,"S_0                 :%lf\n",initModel.S0);
				fprintf(fptr,"S_1                 :%lf\n",initModel.S1);
				fprintf(fptr,"v_mac               :%lf\n",initModel.mac);
				fprintf(fptr,"filling factor      :%lf\n",initModel.alfa);
				fprintf(fptr,"# Iterations        :%d\n",0);
				fprintf(fptr,"chisqr              :%le\n",0.0);
				fprintf(fptr,"\n\n");
				fclose(fptr);
			}
			else{
				printf("\n ¡¡¡¡¡ ERROR: OUTPUT MODEL FILE CAN NOT BE OPENED: %s \n !!!!!",nameAuxOutputModel);
			}			
			free(spectroPER);
		}
		else if(strcmp(file_ext(configCrontrolFile.ObservedProfiles),FITS_FILE)==0){ // invert image from fits file 
			
			fitsImage = readFitsSpectroImage(configCrontrolFile.ObservedProfiles,0,nlambda,0);
			printf("\n--------------------------------------------------------------------------------");
			printf("\nOBSERVED PROFILES FILE READ: %s", configCrontrolFile.ObservedProfiles);
			printf("\n--------------------------------------------------------------------------------\n");
			// ALLOCATE MEMORY FOR STORE THE RESULTS 
			int indexPixel = 0;
			Init_Model * vModels = (Init_Model *) calloc (fitsImage->numPixels , sizeof(Init_Model));
			float * vChisqrf = (float *) calloc (fitsImage->numPixels , sizeof(float));
			int * vNumIter = (int *) calloc (fitsImage->numPixels , sizeof(int));

			for(indexPixel = 0; indexPixel < fitsImage->numPixels; indexPixel++){
				//Initial Model
				Init_Model initModel;
				initModel.eta0 = INITIAL_MODEL.eta0;
				initModel.B = INITIAL_MODEL.B; //200 700
				initModel.gm = INITIAL_MODEL.gm;
				initModel.az = INITIAL_MODEL.az;
				initModel.vlos = INITIAL_MODEL.vlos; //km/s 0
				initModel.mac = INITIAL_MODEL.mac;
				initModel.dopp = INITIAL_MODEL.dopp;
				initModel.aa = INITIAL_MODEL.aa;
				initModel.alfa = INITIAL_MODEL.alfa; //0.38; //stray light factor
				initModel.S0 = INITIAL_MODEL.S0;
				initModel.S1 = INITIAL_MODEL.S1;
				estimacionesClasicas(wlines[1],vLambda, nlambda, fitsImage->pixels[indexPixel].spectro, &initModel,0,cuantic);
				vModels[indexPixel] = initModel;

			}
			char nameAuxOutputModel [4096];
			if(configCrontrolFile.ObservedProfiles[0]!='\0')
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.ObservedProfiles));
			else
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.InitialGuessModel));
			strcat(nameAuxOutputModel,"_model_ce");
			strcat(nameAuxOutputModel,FITS_FILE);			
			if(!writeFitsImageModels(nameAuxOutputModel,fitsImage->rows,fitsImage->cols,vModels,vChisqrf,vNumIter,configCrontrolFile.saveChisqr)){
					printf("\n ERROR WRITING FILE OF MODELS: %s",nameAuxOutputModel);
			}
			free(vModels);
			free(vChisqrf);
			free(vNumIter);
		}
		else{
			printf("\n OBSERVED PROFILES DOESN'T HAVE CORRECT EXTENSION  .PER or .FITS ");
			exit(EXIT_FAILURE);
		}
	}
	else if(configCrontrolFile.NumberOfCycles==0){ // synthesis
		if(access(configCrontrolFile.StrayLightFile,F_OK)!=-1){ //  IF NOT EMPTY READ stray light file 
			slight = readPerStrayLightFile(configCrontrolFile.StrayLightFile,nlambda,vOffsetsLambda);
		}
  
		Init_Model initModel;
		initModel.eta0 = INITIAL_MODEL.eta0;
		initModel.B = INITIAL_MODEL.B; //200 700
		initModel.gm = INITIAL_MODEL.gm;
		initModel.az = INITIAL_MODEL.az;
		initModel.vlos = INITIAL_MODEL.vlos; //km/s 0
		initModel.mac = INITIAL_MODEL.mac;
		initModel.dopp = INITIAL_MODEL.dopp;
		initModel.aa = INITIAL_MODEL.aa;
		initModel.alfa = INITIAL_MODEL.alfa; //0.38; //stray light factor
		initModel.S0 = INITIAL_MODEL.S0;
		initModel.S1 = INITIAL_MODEL.S1;
		printf("\n MODEL ATMOSPHERE: \n");
		printf("\n ETA0: %lf",initModel.eta0);
		printf("\n B: %lf",initModel.B);
		printf("\n vlos: %lf",initModel.vlos);
		printf("\n dopp: %lf",initModel.dopp);
		printf("\n aa: %lf",initModel.aa);
		printf("\n gm: %lf",initModel.gm);
		printf("\n az: %lf",initModel.az);
		printf("\n S0: %lf",initModel.S0);
		printf("\n S1: %lf",initModel.S1);      
		printf("\n mac: %lf",initModel.mac);
		printf("\n alfa: %lf",initModel.alfa);
		printf("\n");          

		h_spectra = (REAL *) malloc(nlambda * NPARMS * sizeof(REAL));
		REAL * h_spectra_mac = (REAL *) malloc(nlambda * NPARMS * sizeof(REAL));
		REAL * h_d_spectra = (REAL *) malloc (nlambda * NTERMS * NPARMS * sizeof(REAL)); 
		REAL * d_spectra_mac, * d_spectra_slight, * d_d_spectra;
		REAL weight_sigma [4];
		weight_sigma[0] = configCrontrolFile.WeightForStokes[0] / configCrontrolFile.noise;
		weight_sigma[1] = configCrontrolFile.WeightForStokes[1] / configCrontrolFile.noise;
		weight_sigma[2] = configCrontrolFile.WeightForStokes[2] / configCrontrolFile.noise;
		weight_sigma[3] = configCrontrolFile.WeightForStokes[3] / configCrontrolFile.noise;

		checkCuda(cudaMalloc(&d_spectra, nlambda * NPARMS * sizeof(REAL)));
		checkCuda(cudaMalloc(&d_spectra2, nlambda * NPARMS * sizeof(REAL)));
		checkCuda(cudaMalloc(&d_spectra_mac, nlambda * NPARMS * sizeof(REAL)));
		checkCuda(cudaMalloc(&d_spectra_slight, nlambda * NPARMS * sizeof(REAL)));
		checkCuda(cudaMalloc(&d_d_spectra, nlambda * NTERMS * NPARMS * sizeof(REAL)));
		checkCuda(cudaMalloc(&d_vlambda,nlambda * sizeof(PRECISION)));
		checkCuda(cudaMalloc(&d_wlines, 2 *sizeof(PRECISION)));
		checkCuda(cudaMalloc(&d_initModel, sizeof(Init_Model)));
		checkCuda(cudaMalloc(&d_Cuantic, sizeof(Cuantic)));
		checkCuda(cudaMalloc(&d_fix, sizeof(int)* 11 ));



		cudaMemcpyToSymbol(d_lambda_const, vLambda, nlambda * sizeof(PRECISION));
		cudaMemcpyToSymbol(d_wlines_const, wlines, 2 *sizeof(PRECISION));
		cudaMemcpyToSymbol(d_weight_const, configCrontrolFile.WeightForStokes, sizeof(REAL)* 4);
		cudaMemcpyToSymbol(d_weight_sigma_const, weight_sigma, sizeof(REAL)* 4);
		
		cudaMemcpyToSymbol(d_initModel_const, &initModel, sizeof(Init_Model));
		cudaMemcpyToSymbol(d_cuantic_const, cuantic, sizeof(Cuantic));


		cudaMemcpyToSymbol(d_fix_const, configCrontrolFile.fix, 11*sizeof(int));
		cudaMemcpyToSymbol(d_nlambda_const, &nlambda, sizeof(int));
		cudaMemcpyToSymbol(d_toplim_const, &configCrontrolFile.toplim, sizeof(PRECISION));
		cudaMemcpyToSymbol(d_miter_const, &configCrontrolFile.NumberOfCycles, sizeof(int));
		cudaMemcpyToSymbol(d_sigma_const, &configCrontrolFile.noise, sizeof(REAL));
		cudaMemcpyToSymbol(d_ilambda_const, &configCrontrolFile.InitialDiagonalElement, sizeof(REAL));
		cudaMemcpyToSymbol(d_use_convolution_const, &configCrontrolFile.ConvolveWithPSF, sizeof(int));
		cudaMemcpyToSymbol(d_ah_const, &configCrontrolFile.mu, sizeof(REAL));
		cudaMemcpyToSymbol(d_logclambda_const, &configCrontrolFile.logclambda, sizeof(int));

		kernel_synthesis<<<1,1>>>(d_Cuantic,d_initModel,d_wlines,nlambda,d_spectra,d_d_spectra, configCrontrolFile.mu,NULL,d_spectra_mac,d_spectra_slight, configCrontrolFile.ConvolveWithPSF,d_fix);
		
		checkCuda( cudaMemcpy( h_spectra, d_spectra, nlambda * NPARMS * sizeof(REAL) , cudaMemcpyDeviceToHost ) );
		checkCuda( cudaMemcpy( h_spectra_mac, d_spectra_mac, nlambda * NPARMS * sizeof(REAL) , cudaMemcpyDeviceToHost ) );
		checkCuda( cudaMemcpy( h_spectra_mac, d_spectra_slight, nlambda * NPARMS * sizeof(REAL) , cudaMemcpyDeviceToHost ) );
		checkCuda( cudaMemcpy( h_d_spectra, d_d_spectra, nlambda * NTERMS * NPARMS * sizeof(REAL) , cudaMemcpyDeviceToHost ) );
		
		cudaFree(d_spectra);
		cudaFree(d_spectra_mac);
		cudaFree(d_spectra_slight);
		cudaFree(d_Cuantic);
		cudaFree(d_d_spectra);

		// in this case basenamefile is from initmodel
		char nameAux [4096];
		if(configCrontrolFile.ObservedProfiles[0]!='\0')
			strcpy(nameAux,get_basefilename(configCrontrolFile.ObservedProfiles));
		else
			strcpy(nameAux,get_basefilename(configCrontrolFile.InitialGuessModel));		
		strcat(nameAux,PER_FILE);
		FILE *fptr = fopen(nameAux, "w");
		if(fptr!=NULL){
			int kk;
			for (kk = 0; kk < nlambda; kk++)
			{
				//fprintf(fptr,"%d\t%f\t%e\t%e\t%e\t%e\n", indexLine, (vLambda[kk]-configCrontrolFile.CentralWaveLenght)*1000, spectra[kk], spectra[kk + nlambda], spectra[kk + nlambda * 2], spectra[kk + nlambda * 3]);
				fprintf(fptr,"%d\t%f\t%e\t%e\t%e\t%e\n", indexLine, (vLambda[kk]-configCrontrolFile.CentralWaveLenght)*1000, h_spectra[kk], h_spectra[kk + nlambda], h_spectra[kk + nlambda * 2], h_spectra[kk + nlambda * 3]);
			}
			fclose(fptr);
			printf("\n*******************************************************************************************");
			printf("\n******************SYNTHESIS DONE: %s",nameAux);
			printf("\n*******************************************************************************************\n");
		}
		else{
			printf("\n ERROR !!! The output file can not be open: %s",nameAux);
		}

		int number_parametros = 0;
		for (number_parametros = 0; number_parametros < NTERMS; number_parametros++)
		{
			strcpy(nameAux,get_basefilename(configCrontrolFile.InitialGuessModel));
			strcat(nameAux,"_GPU_");
			char extension[10];
			sprintf(extension, "%d%s", number_parametros,".per");
			strcat(nameAux,extension);
			FILE *fptr = fopen(nameAux, "w");
			//printf("\n FUNCION RESPUESTA: %d \n",number_parametros);
			int kk;
			for (kk = 0; kk < nlambda; kk++)
			{
				fprintf(fptr,"1\t%lf\t%le\t%le\t%le\t%le\n", vLambda[kk],
				h_d_spectra[kk + nlambda * number_parametros],
				h_d_spectra[kk + nlambda * number_parametros + nlambda * NTERMS],
				h_d_spectra[kk + nlambda * number_parametros + nlambda * NTERMS * 2],
				h_d_spectra[kk + nlambda * number_parametros + nlambda * NTERMS * 3]);
			}
			fclose(fptr);
		}
		


		free(h_spectra);
		free(h_spectra_mac);
		free(h_d_spectra);

	}
	else{ // INVERT PIXEL FROM PER FILE OR IMAGE FROM FITS FILE 
		cudaMemcpyToSymbol(cordicPosFila, posFila, TAMANIO_SVD * TAMANIO_SVD*sizeof(int));
		cudaMemcpyToSymbol(cordicPosCol, posCol, TAMANIO_SVD * TAMANIO_SVD*sizeof(int));
		PRECISION powAux = pow(2.0,-39);
		cudaMemcpyToSymbol(LIMITE_INFERIOR_PRECISION_SVD, &powAux, sizeof(PRECISION));
		cudaMemcpyToSymbol(LIMITE_INFERIOR_PRECISION_TRIG, &powAux, sizeof(PRECISION));
		cudaMemcpyToSymbol(LIMITE_INFERIOR_PRECISION_SINCOS, &powAux, sizeof(PRECISION));

		REAL weight_sigma [4];
		weight_sigma[0] = configCrontrolFile.WeightForStokes[0] / configCrontrolFile.noise;
		weight_sigma[1] = configCrontrolFile.WeightForStokes[1] / configCrontrolFile.noise;
		weight_sigma[2] = configCrontrolFile.WeightForStokes[2] / configCrontrolFile.noise;
		weight_sigma[3] = configCrontrolFile.WeightForStokes[3] / configCrontrolFile.noise;
		

		if(strcmp(file_ext(configCrontrolFile.ObservedProfiles),PER_FILE)==0){ // invert only per file
			if(configCrontrolFile.fix[10] &&  access(configCrontrolFile.StrayLightFile,F_OK)!=-1){ //  IF NOT EMPTY READ stray light file 
				slight = readPerStrayLightFile(configCrontrolFile.StrayLightFile,nlambda,vOffsetsLambda);
			}			
			float * spectroPER = (float *) calloc(nlambda*NPARMS,sizeof(float));
			float * d_spectroPER;
			FILE * fReadSpectro;
			char * line = NULL;
			size_t len = 0;
			ssize_t read;
			fReadSpectro = fopen(configCrontrolFile.ObservedProfiles, "r");
			
			int contLine=0;
			if (fReadSpectro == NULL)
			{
				printf("Error opening the file of parameters, it's possible that the file doesn't exist. Please verify it. \n");
				printf("\n ******* THIS IS THE NAME OF THE FILE RECEVIED : %s \n", configCrontrolFile.ObservedProfiles);
				fclose(fReadSpectro);
				exit(EXIT_FAILURE);
			}
			
			float aux1, aux2, aux3, aux4, aux5, aux6;
			while ((read = getline(&line, &len, fReadSpectro)) != -1 && contLine<nlambda) {
				sscanf(line,"%e %e %e %e %e %e",&aux1,&aux2,&aux3,&aux4,&aux5,&aux6);
				spectroPER[contLine] = aux3;
				spectroPER[contLine + nlambda] = aux4;
				spectroPER[contLine + nlambda * 2] = aux5;
				spectroPER[contLine + nlambda * 3] = aux6;
				contLine++;
			}
			fclose(fReadSpectro);

			if(configCrontrolFile.fix[10] &&  access(configCrontrolFile.StrayLightFile,F_OK)!=-1){ //  IF NOT EMPTY READ stray light file 
				slight = readPerStrayLightFile(configCrontrolFile.StrayLightFile,nlambda,vOffsetsLambda);
			}
      
      	
			
			Init_Model initModel;
			initModel.eta0 = INITIAL_MODEL.eta0;
			initModel.B = INITIAL_MODEL.B; //200 700
			initModel.gm = INITIAL_MODEL.gm;
			initModel.az = INITIAL_MODEL.az;
			initModel.vlos = INITIAL_MODEL.vlos; //km/s 0
			initModel.mac = INITIAL_MODEL.mac;
			initModel.dopp = INITIAL_MODEL.dopp;
			initModel.aa = INITIAL_MODEL.aa;
			initModel.alfa = INITIAL_MODEL.alfa; //0.38; //stray light factor
			initModel.S0 = INITIAL_MODEL.S0;
			initModel.S1 = INITIAL_MODEL.S1;

			h_spectra = (REAL *) malloc(nlambda * NPARMS * sizeof(REAL));

			checkCuda(cudaMalloc(&d_spectroPER, nlambda*NPARMS*sizeof(float)));
			checkCuda(cudaMalloc(&d_vModels, sizeof(Init_Model)));
			checkCuda(cudaMalloc(&d_vChisqrf, sizeof(float)));
			checkCuda(cudaMalloc(&d_vNumIter, sizeof(int)));
			checkCuda(cudaMalloc(&d_spectra, nlambda * NPARMS * sizeof(REAL)));
			
			checkCuda(cudaMalloc(&d_displsSpectro, sizeof(int)));
			checkCuda(cudaMalloc(&d_sendCountPixels, sizeof(int)));
			checkCuda(cudaMalloc(&d_displsPixels, sizeof(int)));
			

			int  displsSpectro = 0;
			int  sendCountPixels = 1;
			int  displsPixels = 0;

			cudaMemcpyToSymbol(d_lambda_const, vLambda, nlambda * sizeof(PRECISION));
			cudaMemcpyToSymbol(d_wlines_const, wlines, 2 *sizeof(PRECISION));
			cudaMemcpyToSymbol(d_weight_const, configCrontrolFile.WeightForStokes, sizeof(REAL)* 4);
			cudaMemcpyToSymbol(d_weight_sigma_const, weight_sigma, sizeof(REAL)* 4);
			

			cudaMemcpyToSymbol(d_initModel_const, &initModel, sizeof(Init_Model));
			cudaMemcpyToSymbol(d_cuantic_const, cuantic, sizeof(Cuantic));
			cudaMemcpyToSymbol(d_fix_const, configCrontrolFile.fix, 11*sizeof(int));
			checkCuda(cudaMemcpy(d_spectroPER,spectroPER,nlambda*NPARMS*sizeof(float), cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_displsSpectro,&displsSpectro,sizeof(int), cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_sendCountPixels,&sendCountPixels,sizeof(int), cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_displsPixels,&displsPixels,sizeof(int), cudaMemcpyHostToDevice));

			cudaMemcpyToSymbol(d_nlambda_const, &nlambda, sizeof(int));
			cudaMemcpyToSymbol(d_toplim_const, &configCrontrolFile.toplim, sizeof(PRECISION));
			cudaMemcpyToSymbol(d_miter_const, &configCrontrolFile.NumberOfCycles, sizeof(int));
			cudaMemcpyToSymbol(d_sigma_const, &configCrontrolFile.noise, sizeof(REAL));
			cudaMemcpyToSymbol(d_ilambda_const, &configCrontrolFile.InitialDiagonalElement, sizeof(REAL));
			cudaMemcpyToSymbol(d_use_convolution_const, &configCrontrolFile.ConvolveWithPSF, sizeof(int));
			cudaMemcpyToSymbol(d_ah_const, &configCrontrolFile.mu, sizeof(REAL));
			cudaMemcpyToSymbol(d_logclambda_const, &configCrontrolFile.logclambda, sizeof(int));
			
			lm_mils<<<1,1>>>(d_spectroPER, d_vModels, d_vChisqrf, d_slight, d_vNumIter,d_spectra, d_displsSpectro, d_sendCountPixels, d_displsPixels, 1,0);

			cudaDeviceSynchronize();

			h_spectra = (REAL *) malloc (nlambda * NPARMS * sizeof(REAL));
			h_vModels = (Init_Model *) malloc(sizeof(Init_Model));
			h_vNumIter = (int *) malloc(sizeof(int));
			h_vChisqrf = (float *) malloc(sizeof(float));

			checkCuda( cudaMemcpy( h_spectra, d_spectra, nlambda * NPARMS * sizeof(REAL) , cudaMemcpyDeviceToHost ) );
			checkCuda( cudaMemcpy( h_vModels, d_vModels, sizeof(Init_Model) , cudaMemcpyDeviceToHost ) );
			checkCuda( cudaMemcpy( h_vNumIter, d_vNumIter, sizeof(int) , cudaMemcpyDeviceToHost ) );
			checkCuda( cudaMemcpy( h_vChisqrf, d_vChisqrf, sizeof(float) , cudaMemcpyDeviceToHost ) );


			// SAVE OUTPUT MODEL 
			char nameAuxOutputModel [4096];
			if(configCrontrolFile.ObservedProfiles[0]!='\0')
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.ObservedProfiles));
			else
				strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.InitialGuessModel));
							
			strcat(nameAuxOutputModel,"_model");
			strcat(nameAuxOutputModel,MOD_FILE);

			FILE *fptr = fopen(nameAuxOutputModel, "w");
			if(fptr!=NULL){
				fprintf(fptr,"eta_0               :%lf\n",h_vModels[0].eta0);
				fprintf(fptr,"magnetic field [G]  :%lf\n",h_vModels[0].B);
				fprintf(fptr,"LOS velocity[km/s]  :%lf\n",h_vModels[0].vlos);
				fprintf(fptr,"Doppler width [A]   :%lf\n",h_vModels[0].dopp);
				fprintf(fptr,"damping             :%lf\n",h_vModels[0].aa);
				fprintf(fptr,"gamma [deg]         :%lf\n",h_vModels[0].gm);
				fprintf(fptr,"phi  [deg]          :%lf\n",h_vModels[0].az);
				fprintf(fptr,"S_0                 :%lf\n",h_vModels[0].S0);
				fprintf(fptr,"S_1                 :%lf\n",h_vModels[0].S1);
				fprintf(fptr,"v_mac [km/s]        :%lf\n",h_vModels[0].mac);
				fprintf(fptr,"filling factor      :%lf\n",h_vModels[0].alfa);
				fprintf(fptr,"# Iterations        :%d\n",h_vNumIter[0]);
				fprintf(fptr,"chisqr              :%le\n",h_vChisqrf[0]);



				fprintf(fptr,"\n\n");
				fclose(fptr);
				printf("\n*******************************************************************************************");
				printf("\n******************INVERTED MODEL SAVED IN FILE: %s",nameAuxOutputModel);
				printf("\n*******************************************************************************************\n");				
			}
			else{
				printf("\n ¡¡¡¡¡ ERROR: OUTPUT MODEL FILE CAN NOT BE OPENED\n !!!!! ");
			}


			// SAVE OUTPUT ADJUST SYNTHESIS PROFILES 
			if(configCrontrolFile.SaveSynthesisAdjusted){
				char nameAuxOutputStokes [4096];
				if(configCrontrolFile.ObservedProfiles[0]!='\0')
					strcpy(nameAuxOutputStokes,get_basefilename(configCrontrolFile.ObservedProfiles));
				else
					strcpy(nameAuxOutputStokes,get_basefilename(configCrontrolFile.InitialGuessModel));				
				strcat(nameAuxOutputStokes,STOKES_PER_EXT);
				FILE *fptr = fopen(nameAuxOutputStokes, "w");
				if(fptr!=NULL){
			      //printf("\n valores de spectro sintetizado\n");
					int kk;
					for (kk = 0; kk < nlambda; kk++)
					{
						fprintf(fptr,"%d\t%f\t%e\t%e\t%e\t%e\n", indexLine, (vLambda[kk]-configCrontrolFile.CentralWaveLenght)*1000, h_spectra[kk], h_spectra[kk + nlambda], h_spectra[kk + nlambda * 2], h_spectra[kk + nlambda * 3]);
					}
					//printf("\nVALORES DE LAS FUNCIONES RESPUESTA \n");
					fclose(fptr);
					printf("\n*******************************************************************************************");
					printf("\n******************SPECTRUM SYNTHESIS ADJUSTED SAVED IN FILE: %s",nameAuxOutputStokes);
					printf("\n*******************************************************************************************\n\n");					
				}
				else{
					printf("\n ¡¡¡¡¡ ERROR: OUTPUT SYNTHESIS PROFILE ADJUSTED FILE CAN NOT BE OPENED\n !!!!! ");
				}
			}
			cudaFree(d_spectroPER);
			cudaFree(d_vModels);
			cudaFree(d_vChisqrf);
			cudaFree(d_vNumIter);
			cudaFree(d_spectra);
			cudaFree(d_displsSpectro);
			cudaFree(d_sendCountPixels);
			cudaFree(d_displsPixels);

			free(spectroPER);	
			free(h_spectra);
			free(h_vModels);
			free(h_vNumIter);
			free(h_vChisqrf);			
		}
		else if(strcmp(file_ext(configCrontrolFile.ObservedProfiles),FITS_FILE)==0){ // invert image from fits file 
			//*****************************************************************************
			// GET DEVIDE PROPERTIES TO KNOW HOW TO DISTRIBUTE THE PIXELS IN KERNEL LM_MILS
			//*****************************************************************************
		    int deviceCount = 0;
		    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    		if (error_id != cudaSuccess)
    		{
        		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        		printf("Result = FAIL\n");
        		exit(EXIT_FAILURE);
    		}
			// This function call returns 0 if there are no CUDA capable devices.
			if (deviceCount == 0)
			{
				printf("\n**************************************************\n");
				printf("\nThere are no available device(s) that support CUDA\n");
				printf("\n**************************************************\n");
				exit(EXIT_FAILURE);
			}
			else
			{
				printf("Detected %d CUDA Capable device(s). Using device with ID %d\n", deviceCount,configCrontrolFile.deviceID);
			}

			int numberDevices = 0;
			cudaGetDeviceCount(&numberDevices);
			printf("\n Number of Devices in the system: %d. Using device with ID %d \n",numberDevices, configCrontrolFile.deviceID);
			//cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, configCrontrolFile.deviceID);
			printf("\nDevice %d: \"%s\"\n", configCrontrolFile.deviceID, deviceProp.name);
        	printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
	        char msg[256];
			sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        	printf("%s", msg);

        	printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",deviceProp.multiProcessorCount,_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        	printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);			
			printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        	printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
			//*****************************************************************************
			//*****************************************************************************
			/*activeWarps = numBlocks * blockSize / deviceProp.warpSize;
			maxWarps = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
			printf("Occupancy: %f %",(double)activeWarps / maxWarps * 100);*/			
			// READ PIXELS FROM IMAGE 
			PRECISION timeReadImage;
			clock_t t;
			t = clock();
			
			fitsImage = readFitsSpectroImage(configCrontrolFile.ObservedProfiles,1,nlambda,0);
			printf("\n--------------------------------------------------------------------------------");
			printf("\nOBSERVED PROFILES FILE READ: %s", configCrontrolFile.ObservedProfiles);
			printf("\n--------------------------------------------------------------------------------\n");
	
			t = clock() - t;
			timeReadImage = ((PRECISION)t)/CLOCKS_PER_SEC; // in seconds 
			printf("\n\n TIME TO READ FITS IMAGE:  %f seconds to execute . Número de pixeles leidos %d ", timeReadImage,fitsImage->numPixels); 
			// array to store synthesis spectra on device 
			// print first pixel 


			if(fitsImage!=NULL){

				int activeWarps;
				int maxWarps;
				int numBlocks=8; // Occupancy in terms of active blocks
				int blockSize = 2; // The launch configurator returned block size
				int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
				int gridSize; // The actual
				int threadPerBlock=32;
				int NSTREAMS = configCrontrolFile.numStreams;
				// function to get minGridSize and blockSize for function lm_mils with no limite of share memory and no limit of maximum block size
				/*cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)lm_mils,threadPerBlock,fitsImage->numPixels); 
				gridSize = (fitsImage->numPixels + blockSize - 1) / blockSize;*/
				/*printf("\n EL GRID SIZE SERÁ: %d el ",gridSize);
				printf("\n Max potential block size MINGRIDSIZE %d BLOCKSIZE %d ",minGridSize,blockSize);*/

				cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,(void*)lm_mils,threadPerBlock,0);
				//printf("\n MAX ACTIVE BLOCKS PER MULTIPROCESSOR %d ",numBlocks);


				int N_RTE_PARALLEL = numBlocks * threadPerBlock; // maximun thread to compute in parallel 
				size_t heap_size;
				cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
				//printf("\n TAMAÑO ORIGINAL DEL HEAP: %i\n",heap_size);
				cudaDeviceSetLimit(cudaLimitMallocHeapSize, 37568*N_RTE_PARALLEL*NSTREAMS);
				
				cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
				//printf("\n EL NUEVO TAMAÑO DEL HEAP ES 32KBx%d: %i\n",N_RTE_PARALLEL,heap_size);				
				// divide work in N STREAMS 
				cudaStream_t stream[NSTREAMS];
				for (i = 0; i < NSTREAMS; ++i)
					checkCuda(cudaStreamCreate(&stream[i]));
				
				int numPixelsStream = fitsImage->numPixels/NSTREAMS;
				int restoStream = fitsImage->numPixels % NSTREAMS;
				int sumStream = 0;                // Sum of counts. Used to calculate displacements
				int h_sendcountsPixelsStream [NSTREAMS] ; // array describing how many elements to send to each process				
				int h_displsPixelsStream [NSTREAMS];  // array describing the displacements where each segment begins
				for ( i = 0; i < NSTREAMS; i++) {
					h_sendcountsPixelsStream[i] = numPixelsStream;
					if (restoStream > 0) {
						h_sendcountsPixelsStream[i]++;
						restoStream--;
					}
					h_displsPixelsStream[i] = sumStream;
					sumStream += h_sendcountsPixelsStream[i];
				}				
				

				// COPY IMAGE TO DEVICE 
				float * d_spectro, * d_spectro2;
				int bytesSpectroImage = sizeof(float)* fitsImage->numPixels * fitsImage->numStokes * fitsImage->nLambdas;
				checkCuda(cudaMalloc(&d_spectro, bytesSpectroImage) );// device
				checkCuda(cudaMalloc(&d_spectro2, bytesSpectroImage) );// device

				// ***********************************

				
				int numPixelsITER[NSTREAMS];// = fitsImage->numPixels/N_RTE_PARALLEL;


				//int resto = fitsImage->numPixels % N_RTE_PARALLEL;
				int sum = 0;                // Sum of counts. Used to calculate displacements
				int sumSpectro = 0;
				int sumLambda = 0;
				int h_sendcountsPixels [NSTREAMS * N_RTE_PARALLEL] ; // array describing how many elements to send to each process
				int h_sendcountsSpectro [NSTREAMS * N_RTE_PARALLEL];
				int h_sendcountsLambda [NSTREAMS * N_RTE_PARALLEL];
				int h_displsPixels [NSTREAMS * N_RTE_PARALLEL];  // array describing the displacements where each segment begins
				int h_displsSpectro [NSTREAMS * N_RTE_PARALLEL];

				//printf("\nNUMERO DE HEBRAS %d",N_RTE_PARALLEL);
				for ( j=0; j<NSTREAMS ; j++){
					//printf("\nNUMERO DE PIXLES STREAM  %d  %d ",j, h_sendcountsPixelsStream[j] );					
					numPixelsITER[j] = h_sendcountsPixelsStream[j]/N_RTE_PARALLEL;
					//printf("\nNUMERO de pixels por hebra stream %d\n",numPixelsITER[j]);
					//printf("\n***************\n");
					int resto = h_sendcountsPixelsStream[j] % N_RTE_PARALLEL;
					for ( i = 0; i < N_RTE_PARALLEL; i++) {
						h_sendcountsPixels[ (j*N_RTE_PARALLEL) + i] = numPixelsITER[j];
						if (resto > 0) {
								h_sendcountsPixels[ (j*N_RTE_PARALLEL) + i]++;
								resto--;
						}
						h_sendcountsSpectro[ (j*N_RTE_PARALLEL) + i] = (h_sendcountsPixels[(j*N_RTE_PARALLEL) +i])*nlambda*NPARMS;
						h_sendcountsLambda[(j*N_RTE_PARALLEL) +i] = (h_sendcountsPixels[(j*N_RTE_PARALLEL) + i])*nlambda;
						h_displsPixels[(j*N_RTE_PARALLEL) +i] = sum;
						h_displsSpectro[(j*N_RTE_PARALLEL) +i] = sumSpectro;
						//displsLambda[i] = sumLambda;
						sum += h_sendcountsPixels[(j*N_RTE_PARALLEL) +i];
						sumSpectro += h_sendcountsSpectro[(j*N_RTE_PARALLEL) +i];
						sumLambda += h_sendcountsLambda[(j*N_RTE_PARALLEL) +i];
					}
				}	
				

				// create structure to store adjusted image if necessary	
				FitsImage * imageStokesAdjust = NULL;
				if(configCrontrolFile.SaveSynthesisAdjusted){
					imageStokesAdjust = (FitsImage *) malloc(sizeof(FitsImage));
					imageStokesAdjust->rows = fitsImage->rows;
					imageStokesAdjust->cols = fitsImage->cols;
					imageStokesAdjust->nLambdas = fitsImage->nLambdas;
					imageStokesAdjust->numStokes = fitsImage->numStokes;
					imageStokesAdjust->pos_col = fitsImage->pos_col;
					imageStokesAdjust->pos_row = fitsImage->pos_row;
					imageStokesAdjust->pos_lambda = fitsImage->pos_lambda;
					imageStokesAdjust->pos_stokes_parameters = fitsImage->pos_stokes_parameters;
					imageStokesAdjust->numPixels = fitsImage->numPixels;
					imageStokesAdjust->pixels = (vpixels* )calloc(imageStokesAdjust->numPixels, sizeof(vpixels));
					for( i=0;i<imageStokesAdjust->numPixels;i++){
						imageStokesAdjust->pixels[i].spectro = (float *) calloc ((imageStokesAdjust->numStokes*imageStokesAdjust->nLambdas),sizeof(float));
					}
					imageStokesAdjust->naxes = fitsImage->naxes;
					imageStokesAdjust->vCard = fitsImage->vCard;
					imageStokesAdjust->vKeyname = fitsImage->vKeyname;
					imageStokesAdjust->nkeys = fitsImage->nkeys;
					imageStokesAdjust->naxis = fitsImage->naxis;
					imageStokesAdjust->bitpix = fitsImage->bitpix;
				}

				// COPY ALL IMAGE TO MEMORY INSIDE GPU
				
				int indexPixel = 0;
				
				// ALLOCATE MEMORY FOR STORE THE RESULTS 
				// memory array of spectraAdjusted to store the arry of spectra stored 
				//printf("\n NUMERO DE SPECTRO : %d\n",fitsImage->numPixels * fitsImage->numStokes * fitsImage->nLambdas);
				float * d_spectraAdjusted;
				float * h_spectraAdjusted = (float *) malloc(bytesSpectroImage);
				h_vModels = (Init_Model *) malloc(sizeof(Init_Model)*fitsImage->numPixels);
				h_vNumIter = (int *) malloc(sizeof(int)*fitsImage->numPixels);
				h_vChisqrf = (float *) malloc(sizeof(float)*fitsImage->numPixels);

				// MALLOC MEMORY 

				
				checkCuda(cudaMalloc(&d_vModels, sizeof(Init_Model)* fitsImage->numPixels) );// device
				checkCuda(cudaMalloc(&d_vChisqrf,sizeof(float)* fitsImage->numPixels) );// device
				checkCuda(cudaMalloc(&d_vNumIter,sizeof(int)* fitsImage->numPixels) );// device
				checkCuda(cudaMalloc(&d_spectraAdjusted, bytesSpectroImage ));// device
				checkCuda(cudaMalloc(&d_displsSpectro, sizeof(int)*NSTREAMS * N_RTE_PARALLEL));
				checkCuda(cudaMalloc(&d_sendCountPixels, sizeof(int)*NSTREAMS * N_RTE_PARALLEL));
				checkCuda(cudaMalloc(&d_displsPixels, sizeof(int)*NSTREAMS * N_RTE_PARALLEL));

				cudaMemcpyToSymbol(d_lambda_const, vLambda, nlambda * sizeof(PRECISION));
				cudaMemcpyToSymbol(d_wlines_const, wlines, 2 *sizeof(PRECISION));
				cudaMemcpyToSymbol(d_weight_const, configCrontrolFile.WeightForStokes, sizeof(REAL)* 4);
				cudaMemcpyToSymbol(d_weight_sigma_const, weight_sigma, sizeof(REAL)* 4);
				cudaMemcpyToSymbol(d_initModel_const, &INITIAL_MODEL, sizeof(Init_Model));
				cudaMemcpyToSymbol(d_cuantic_const, cuantic, sizeof(Cuantic));
				cudaMemcpyToSymbol(d_fix_const, configCrontrolFile.fix, 11*sizeof(int));
				cudaMemcpyToSymbol(d_nlambda_const, &nlambda, sizeof(int));
				cudaMemcpyToSymbol(d_toplim_const, &configCrontrolFile.toplim, sizeof(PRECISION));
				cudaMemcpyToSymbol(d_miter_const, &configCrontrolFile.NumberOfCycles, sizeof(int));
				cudaMemcpyToSymbol(d_sigma_const, &configCrontrolFile.noise, sizeof(REAL));
				cudaMemcpyToSymbol(d_ilambda_const, &configCrontrolFile.InitialDiagonalElement, sizeof(REAL));
				cudaMemcpyToSymbol(d_use_convolution_const, &configCrontrolFile.ConvolveWithPSF, sizeof(int));
				cudaMemcpyToSymbol(d_ah_const, &configCrontrolFile.mu, sizeof(REAL));
				cudaMemcpyToSymbol(d_logclambda_const, &configCrontrolFile.logclambda, sizeof(int));

				checkCuda(cudaMemcpy(d_spectro,fitsImage->spectroImagen,bytesSpectroImage, cudaMemcpyHostToDevice));


				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
		
				
				checkCuda(cudaMemcpy(d_displsSpectro,h_displsSpectro,sizeof(int)*NSTREAMS * N_RTE_PARALLEL, cudaMemcpyHostToDevice));
				checkCuda(cudaMemcpy(d_sendCountPixels,h_sendcountsPixels,sizeof(int)*NSTREAMS * N_RTE_PARALLEL, cudaMemcpyHostToDevice));
				checkCuda(cudaMemcpy(d_displsPixels,h_displsPixels,sizeof(int)*NSTREAMS * N_RTE_PARALLEL, cudaMemcpyHostToDevice));
				

				cudaEventRecord(start);
				/****** LAUNCH KERNELS ******/
				for (i = 0; i < NSTREAMS; ++i){
					lm_mils<<<numBlocks,threadPerBlock,numBlocks*NTERMS*sizeof(REAL),stream[i]>>>(d_spectro,
						d_vModels, d_vChisqrf, d_slight, d_vNumIter, d_spectraAdjusted, d_displsSpectro, d_sendCountPixels, d_displsPixels, N_RTE_PARALLEL,i);
				}
				/****************************/
				cudaEventRecord(stop,0);
				cudaEventSynchronize(stop);
				//cudaDeviceSynchronize();

				checkCuda( cudaMemcpy( h_spectraAdjusted, d_spectraAdjusted, bytesSpectroImage , cudaMemcpyDeviceToHost ) );
				checkCuda( cudaMemcpy( h_vModels, d_vModels, sizeof(Init_Model) *fitsImage->numPixels, cudaMemcpyDeviceToHost ) );
				checkCuda( cudaMemcpy( h_vNumIter, d_vNumIter, sizeof(int) *fitsImage->numPixels, cudaMemcpyDeviceToHost ) );
				checkCuda( cudaMemcpy( h_vChisqrf, d_vChisqrf, sizeof(float)*fitsImage->numPixels, cudaMemcpyDeviceToHost ) );

				
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);
				printf("\n FINISH EXECUTION OF INVERSION: %f seconds to execute \n", milliseconds/1000);
				
				char nameAuxOutputModel [4096];
				if(configCrontrolFile.ObservedProfiles[0]!='\0')
					strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.ObservedProfiles));
				else
					strcpy(nameAuxOutputModel,get_basefilename(configCrontrolFile.InitialGuessModel));				

				strcat(nameAuxOutputModel,MOD_FITS);
				if(!writeFitsImageModels(nameAuxOutputModel,fitsImage->rows,fitsImage->cols,h_vModels,h_vChisqrf,h_vNumIter,configCrontrolFile.saveChisqr)){
						printf("\n ERROR WRITING FILE OF MODELS: %s",nameAuxOutputModel);
				}
				// PROCESS FILE OF SYNTETIC PROFILES

				if(configCrontrolFile.SaveSynthesisAdjusted){
					// WRITE SINTHETIC PROFILES TO FITS FILE
					char nameAuxOutputStokes [4096];
					if(configCrontrolFile.ObservedProfiles[0]!='\0')
						strcpy(nameAuxOutputStokes,get_basefilename(configCrontrolFile.ObservedProfiles));
					else
						strcpy(nameAuxOutputStokes,get_basefilename(configCrontrolFile.InitialGuessModel));					
					strcat(nameAuxOutputStokes,STOKES_FIT_EXT);

					for(indexPixel=0;indexPixel<fitsImage->numPixels;indexPixel++)
					{	
						int kk;
						for (kk = 0; kk < (nlambda * NPARMS); kk++)
						{
							imageStokesAdjust->pixels[indexPixel].spectro[kk] = h_spectraAdjusted[kk+(indexPixel*(nlambda * NPARMS))] ;
						}
					}	
					if(!writeFitsImageProfiles(nameAuxOutputStokes,nameInputFileSpectra,imageStokesAdjust)){
						printf("\n ERROR WRITING FILE OF SINTHETIC PROFILES: %s",nameOutputFilePerfiles);
					}
				}
				if(configCrontrolFile.SaveSynthesisAdjusted)
					free(imageStokesAdjust);


				for (i = 0; i < NSTREAMS; ++i)
					checkCuda( cudaStreamDestroy(stream[i]) );				
				cudaFree(d_spectro);
				cudaFree(d_vModels);
				cudaFree(d_vChisqrf);
				cudaFree(d_vNumIter);
				cudaFree(d_spectraAdjusted);
				cudaFree(d_displsSpectro);
				cudaFree(d_sendCountPixels);
				cudaFree(d_displsPixels);			

				free(h_vModels);
				free(h_vChisqrf);
				free(h_vNumIter);
				free(h_spectraAdjusted);	

			}

			else{
				printf("\n\n ***************************** FITS FILE WITH THE SPECTRO IMAGE CAN NOT BE READ IT ******************************\n");
			}			


		}
		else{
			printf("\n OBSERVED PROFILES DOESN'T HAVE CORRECT EXTENSION  .PER or .FITS ");
			exit(EXIT_FAILURE);
		}
	}


	if(configCrontrolFile.ConvolveWithPSF){
		checkCuda(cudaFree(d_psfFunction));
	}

	free(cuantic);
	free(wlines);

	if(psfFunction!=NULL) free(psfFunction);


	return 0;
}
