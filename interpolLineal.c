#include <math.h>
#include "defines.h"
#include "time.h"
#include "interpolLineal.h"
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


/**
 * Make the interpolation between deltaLambda and PSF where deltaLambda es x and PSF f(x)
 *  Return the array with the interpolation. 
 * */
int interpolationSplinePSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples){

	size_t i;
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
  	gsl_spline *spline_cubic = gsl_spline_alloc(gsl_interp_cspline, N_PSF);
	//gsl_spline *spline_akima = gsl_spline_alloc(gsl_interp_akima, NSamples);
	//gsl_spline *spline_steffen = gsl_spline_alloc(gsl_interp_steffen, NSamples);

	gsl_spline_init(spline_cubic, deltaLambda, PSF, N_PSF);
	//gsl_spline_init(spline_akima, deltaLambda, PSF, N_PSF);
	//gsl_spline_init(spline_steffen, deltaLambda, PSF, N_PSF);

	for (i = 0; i < NSamples; ++i){
   	
      //fInterpolated[i] = gsl_spline_eval(spline_cubic, xi, acc);
      //PRECISION yi_akima = gsl_spline_eval(spline_akima, xi, acc);
      //PRECISION yi_steffen = gsl_spline_eval(spline_steffen, lambdasSamples[i], acc);
		PRECISION yi = gsl_spline_eval(spline_cubic, lambdasSamples[i], acc);
		if(!gsl_isnan(yi)){
			fInterpolated[i] = yi;
		}
		else
		{
			fInterpolated[i] = 0.0f;
		}
		
   }

  	gsl_spline_free(spline_cubic);
	//gsl_spline_free(spline_akima);
	//gsl_spline_free(spline_steffen);
	gsl_interp_accel_free(acc);

	return 1;
}


/**
 * Make the interpolation between deltaLambda and PSF where deltaLambda es x and PSF f(x)
 *  Return the array with the interpolation. 
 * */
int interpolationLinearPSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples,double offset){

	size_t i;
	gsl_interp *interpolation = gsl_interp_alloc (gsl_interp_linear,N_PSF);
	gsl_interp_init(interpolation, deltaLambda, PSF, N_PSF);
	gsl_interp_accel * accelerator =  gsl_interp_accel_alloc();

	//printf("\n[");
	for (i = 0; i < NSamples; ++i){
		//printf("\n VALOR A INERPOLAR EN X %f, iteration %li",lambdasSamples[i]-offset,i);
		//printf("\t%f,",lambdasSamples[i]-offset);
		double aux;
		if(offset>=0){
			if(lambdasSamples[i]-offset>= deltaLambda[0]){
				aux = gsl_interp_eval(interpolation, deltaLambda, PSF, lambdasSamples[i]-offset, accelerator);
						// if lambdasSamples[i] is out of range from deltaLambda then aux is GSL_NAN, we put nan values to 0. 
				if(!gsl_isnan(aux)) 
					fInterpolated[i] = aux;
				else
					fInterpolated[i] = 0.0f;
			}
			else
			{
				aux = 0.0f;
			}
		}
		else{
			//printf("lamba+offset %f  deltalambda %f ",lambdasSamples[i]+offset,deltaLambda[NSamples-1] );
			if(lambdasSamples[i]-offset>= deltaLambda[NSamples-1]){
				aux = gsl_interp_eval(interpolation, deltaLambda, PSF, lambdasSamples[i]-offset, accelerator);
						// if lambdasSamples[i] is out of range from deltaLambda then aux is GSL_NAN, we put nan values to 0. 
				if(!gsl_isnan(aux)) 
					fInterpolated[i] = aux;
				else
					fInterpolated[i] = 0.0f;
			}
			else
			{
				aux = 0.0f;
			}			
		}
	}
	//printf("]\n");

  	// normalizations 
	double cte = 0;
	for(i=0; i< NSamples; i++){
		cte += fInterpolated[i];
	}
	for(i=0; i< NSamples; i++){
		fInterpolated[i] /= cte;
	}
  	gsl_interp_accel_free(accelerator);
	gsl_interp_free(interpolation);
  	
	/*for(i=0; i< NSamples; i++){
		if(fInterpolated[i]<1e-3)
			fInterpolated[i] =0;
	}*/


	return 1;
}