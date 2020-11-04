/*
 * multShare.c
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */

 #include "multShare.cuh"

 // Matrix multiplication - Host code 
 // Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
 /*void MatMul(const Matrix A, const Matrix B, Matrix C) { 
   // Load A and B to device memory 
   Matrix d_A; 
   d_A.width = d_A.stride = A.width; 
   d_A.height = A.height; 
   size_t size = A.width * A.height * sizeof(float); 
   cudaError_t err = cudaMalloc(&d_A.elements, size); 
   printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
   err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
   printf("Copy A to device: %s\n",cudaGetErrorString(err)); 
 
   Matrix d_B; 
   d_B.width = d_B.stride = B.width; 
   d_B.height = B.height; 
   size = B.width * B.height * sizeof(float); 
   err = cudaMalloc(&d_B.elements, size); 
   printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
   err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
   printf("Copy B to device: %s\n",cudaGetErrorString(err)); 
 
   // Allocate C in device memory 
   Matrix d_C; 
   d_C.width = d_C.stride = C.width; 
   d_C.height = C.height; 
   size = C.width * C.height * sizeof(float); 
   err = cudaMalloc(&d_C.elements, size); 
   printf("CUDA malloc C: %s\n",cudaGetErrorString(err));
 
   // Invoke kernel 
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
   dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y); 
     MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
     err = cudaThreadSynchronize();
     printf("Run kernel: %s\n", cudaGetErrorString(err));
 
   // Read C from device memory 
   err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
   printf("Copy C off of device: %s\n",cudaGetErrorString(err));
 
   // Free device memory
   cudaFree(d_A.elements); 
   cudaFree(d_B.elements); 
   cudaFree(d_C.elements); 
 } */
 
 // Get a matrix element
 __device__ double GetElementD(const MatrixD A, int row, int col) { 
   return A.elements[row * A.stride + col]; 
 } 
 
 // Set a matrix element 
 __device__ void SetElementD(MatrixD A, int row, int col, double value) { 
   A.elements[row * A.stride + col] = value; 
 } 
 
 // Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is 
 // located col sub-matrices to the right and row sub-matrices down 
 // from the upper-left corner of A 
 __device__ MatrixD GetSubMatrixD(MatrixD A, int row, int col) { 
   MatrixD Asub; 
   Asub.width = BLOCK_SIZE; 
   Asub.height = BLOCK_SIZE; 
   Asub.stride = A.stride; 
   Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]; 
   return Asub; 
 }
 
 
 // Matrix multiplication kernel called by MatMul() 
 __global__ void MatMulKernelD(MatrixD A, MatrixD B, MatrixD C) { 
   // Block row and column 
   int blockRow = blockIdx.y; 
   int blockCol = blockIdx.x; 
 
   // Each thread block computes one sub-matrix Csub of C
   MatrixD Csub = GetSubMatrixD(C, blockRow, blockCol); 
 
   // Each thread computes one element of Csub 
   // by accumulating results into Cvalue 
   double Cvalue = 0.0; 
 
   // Thread row and column within Csub 
   int row = threadIdx.y; 
   int col = threadIdx.x; 
 
   // Loop over all the sub-matrices of A and B that are 
   // required to compute Csub 
   // Multiply each pair of sub-matrices together 
   // and accumulate the results 
   for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
     // Get sub-matrix Asub of A 
     MatrixD Asub = GetSubMatrixD(A, blockRow, m); 
 
     // Get sub-matrix Bsub of B 
     MatrixD Bsub = GetSubMatrixD(B, m, blockCol); 
 
     // Shared memory used to store Asub and Bsub respectively 
     __shared__ double As[BLOCK_SIZE][BLOCK_SIZE]; 
     __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE]; 
 
     // Load Asub and Bsub from device memory to shared memory 
     // Each thread loads one element of each sub-matrix 
     As[row][col] = GetElementD(Asub, row, col); 
     Bs[row][col] = GetElementD(Bsub, row, col); 
 
     // Synchronize to make sure the sub-matrices are loaded 
     // before starting the computation 
     __syncthreads(); 
 
     // Multiply Asub and Bsub together 
     for (int e = 0; e < BLOCK_SIZE; ++e) 
       Cvalue += As[row][e] * Bs[e][col];
  
     // Synchronize to make sure that the preceding 
     // computation is done before loading two new 
     // sub-matrices of A and B in the next iteration 
     __syncthreads();  
   }
 
   // Write Csub to device memory 
   // Each thread writes one element 
   SetElementD(Csub, row, col, Cvalue); 
 }



 // Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernelNotShare(MatrixD A, MatrixD B, MatrixD C) { 
    // Each thread computes one element of C 
    // by accumulating results into Cvalue 
    double Cvalue = 0.0; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if(row > A.height || col > B.width) return;
    for (int e = 0; e < A.width; ++e) 
      Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]); 
    C.elements[row * C.width + col] = Cvalue; 
  }