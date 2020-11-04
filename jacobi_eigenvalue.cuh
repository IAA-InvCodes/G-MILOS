#include "defines.h"  

__device__ void jacobi_eigenvalue ( int n, double a[], int it_max, double v[],double d[], int *it_num, int *rot_num );
__device__ void r8mat_diag_get_vector ( int n, double a[], double v[] );
__device__ void r8mat_identity ( int n, double a[] );
double r8mat_is_eigen_right ( int n, int k, double a[], double x[],
  double lambda[] );
double r8mat_norm_fro ( int m, int n, double a[] );
void r8mat_print ( int m, int n, double a[], char *title );
void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
  int jhi, char *title );
void r8vec_print ( int n, double a[], char *title );
void timestamp ( void );

