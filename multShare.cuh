/*
 * multShare.h
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */

#include <stdio.h>

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 
typedef struct { 
  int width; 
  int height; 
  double * elements; 
  int stride;
} MatrixD; 

// Thread block size 
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16 
#endif

__global__ void MatMulKernelD(MatrixD A, MatrixD B, MatrixD C); 
__global__ void MatMulKernelNotShare(MatrixD A, MatrixD B, MatrixD C);