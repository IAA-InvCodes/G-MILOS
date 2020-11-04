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

	gsl_spline_init(spline_cubic, deltaLambda, PSF, N_PSF);

	for (i = 0; i < NSamples; ++i){
   	
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

	for (i = 0; i < NSamples; ++i){
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
  	


	return 1;
}