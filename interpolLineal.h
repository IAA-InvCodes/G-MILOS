#include "defines.h"

int interpolationSplinePSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples);
int interpolationLinearPSF(PRECISION *deltaLambda, PRECISION * PSF, PRECISION * lambdasSamples, size_t N_PSF, PRECISION * fInterpolated, size_t NSamples,double offset);