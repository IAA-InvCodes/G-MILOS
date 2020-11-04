#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

/*#define O_Tile_Width 30
#define Mask_width 30
#define width 30
#define Block_width (O_Tile_Width+(Mask_width-1))
#define Mask_radius (Mask_width/2)*/

#define RG          10
#define BLOCKSIZE   1024


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }
/****************/
/* CPU FUNCTION */
/****************/
void h_convolution_1D(const float * __restrict__ h_Signal, const double * __restrict__ h_ConvKernel, float * __restrict__ h_Result_CPU, const int N, const int K) {

   for (int i = 0; i < N; i++) {

      double temp = 0.f;

      int N_start_point = i - (K / 2);

      for (int j = 0; j < K; j++) 
         if (N_start_point + j >= 0 && N_start_point + j < N) {
            temp += h_Signal[N_start_point+ j] * h_ConvKernel[j];
         }

      h_Result_CPU[i] = temp;
   }
}

/********************/
/* BASIC GPU KERNEL */
/********************/
__global__ void d_convolution_1D_basic(const float * __restrict__ d_Signal, const double * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, const int N, const int K) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;

   double temp = 0.f;

   int N_start_point = i - (K / 2);

   for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {
      temp += d_Signal[N_start_point+ j] * d_ConvKernel[j];
   }

   d_Result_GPU[i] = temp;
}

/***************************/
/* GPU KERNEL WITH CACHING */
/***************************/
__global__ void d_convolution_1D_caching(const float * __restrict__ d_Signal, const double * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, 
                      const int N, const int K) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;

   __shared__ double d_Tile[BLOCKSIZE];

   d_Tile[threadIdx.x] = d_Signal[i];
   __syncthreads();

   double temp = 0.f;

   int N_start_point = i - (K / 2);

   for (int j = 0; j < K; j++) 
      if (N_start_point + j >= 0 && N_start_point + j < N) {

      if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))
         // --- The signal element is in the tile loaded in the shared memory
         temp += d_Tile[threadIdx.x + j - (K / 2)] * d_ConvKernel[j]; 
      else
         // --- The signal element is not in the tile loaded in the shared memory
         temp += d_Signal[N_start_point + j] * d_ConvKernel[j];
   }
   d_Result_GPU[i] = temp;
}

__global__ void convolution_1D_tiled(float *N,float *M,float *P){
   int index_out_x=blockIdx.x*O_Tile_Width+threadIdx.x;
   int index_in_x=index_out_x-Mask_radius;
   __shared__ float N_shared[Block_width];
   float Pvalue=0.0;

   //Load Data into shared Memory (into TILE)
   if((index_in_x>=0)&&(index_in_x<width))
   {
      N_shared[threadIdx.x]=N[index_in_x];
   }
   else
   {
      N_shared[threadIdx.x]=0.0f;
   }
   __syncthreads();

   //Calculate Convolution (Multiply TILE and Mask Arrays)
   if(threadIdx.x<O_Tile_Width)
   {
      //Pvalue=0.0f;
      for(int j=0;j<Mask_width;j++){
         Pvalue+=M[j]*N_shared[j+threadIdx.x];
      }
   P[index_out_x]=Pvalue;
   }
}

__global__ void convolution_1D(float *N,float *M,float *P,int Mask_width,int width){
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   float Pvalue=0.0;
   int N_start_point=i-(Mask_width/2);
   for(int j=0;j<Mask_width;j++){
      if(((N_start_point+j)>=0) && ((N_start_point+j)<width)){
         Pvalue+=N[N_start_point+j]*M[j];
      }
   }
   P[i]=Pvalue;
}

__global__ void convAgB(double *a, double *b, double *c, int sa, int sb)
{
    int i = (threadIdx.x + blockIdx.x * blockDim.x);
    int idT = threadIdx.x;
    int out,j;

    __shared__ double c_local [512];

    c_local[idT] = c[i];

    out = (i > sa) ? sa : i + 1;
    j   = (i > sb) ? i - sb + 1 : 1;

    for(; j < out; j++)
    {    
       if(c_local[idT] > a[j] + b[i-j])
          c_local[idT] = a[j] + b[i-j]; 
    }   

    c[i] = c_local[idT];
} 


int main()
{
   int Mask_width=30;
   int width=30;

   float * input;
   float * Mask;
   float * output;
   float * device_input;
   float * device_Mask;
   float * device_output;

   input=(float *)malloc(sizeof(float)*width);
   Mask=(float *)malloc(sizeof(float)*Mask_width);
   output=(float *)malloc(sizeof(float)*width);

   for(int i=0;i<width;i++){
      input[i]=1.0;
   }

   for(int i=0;i<Mask_width;i++){
      Mask[i]=1.0;
   }
   printf("\nInput:\n");
   for(int i=0;i<width;i++){
      printf(" %0.2f\t",*(input+i));
   }
   printf("\nMask:\n");
   for(int i=0;i<Mask_width;i++){
      printf(" %0.2f\t",*(Mask+i));
   }

   /*cudaMalloc((void **)&device_input,sizeof(float)*width);
   cudaMalloc((void **)&device_Mask,sizeof(float)*Mask_width);
   cudaMalloc((void **)&device_output,sizeof(float)*width);

   cudaMemcpy(device_input,input,sizeof(float)*width,cudaMemcpyHostToDevice);
   cudaMemcpy(device_Mask,Mask,sizeof(float)*Mask_width,cudaMemcpyHostToDevice);

   dim3 dimBlock(Block_width,1,1);
   dim3 dimGrid((((width-1)/O_Tile_Width)+1),1,1);
   convolution_1D_tiled<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output);
   cudaMemcpy(output,device_output,sizeof(float)*width,cudaMemcpyDeviceToHost);*/
   cudaMalloc((void **)&device_input,sizeof(float)*width);
   cudaMalloc((void **)&device_Mask,sizeof(float)*Mask_width);
   cudaMalloc((void **)&device_output,sizeof(float)*width);
 
   cudaMemcpy(device_input,input,sizeof(float)*width,cudaMemcpyHostToDevice);
   cudaMemcpy(device_Mask,Mask,sizeof(float)*Mask_width,cudaMemcpyHostToDevice);
 
   dim3 dimGrid(((width-1)/Mask_width)+1, 1,1);
   dim3 dimBlock(Mask_width,1, 1);
 
   convolution_1D<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output,Mask_width,width);
 
   cudaDeviceSynchronize();
 
 
   cudaMemcpy(output,device_output,sizeof(float)*width,cudaMemcpyDeviceToHost);   

   printf("\nOutput:\n");
   for(int i=0;i<width;i++){
      printf(" %0.2f\t",*(output+i));
   }

   cudaFree(device_input);
   cudaFree(device_Mask);
   cudaFree(device_output);
   free(input);
   free(Mask);
   free(output);

   printf("\n\nNumber of Blocks: %d ",dimGrid.x);
   printf("\n\nNumber of Threads Per Block: %d ",dimBlock.x);


   printf ("\n USING ANOTHER TYPE OF CONVOLUTION *************** SHARING MEMORY *****************************\n");

   const int N = 15;           // --- Signal length
   const int K = 5;            // --- Convolution kernel length

   float *h_Signal         = (float *)malloc(N * sizeof(float));
   float *h_Result_CPU     = (float *)malloc(N * sizeof(float));
   float *h_Result_GPU     = (float *)malloc(N * sizeof(float));
   double *h_ConvKernel     = (double *)malloc(K * sizeof(double));

   float *d_Signal;        cudaMalloc(&d_Signal,     N * sizeof(float));
   float *d_Result_GPU;    cudaMalloc(&d_Result_GPU, N * sizeof(float));
   double *d_ConvKernel;   cudaMalloc(&d_ConvKernel, K * sizeof(double));

   for (int i=0; i < N; i++) { h_Signal[i] = (float)(rand() % RG); }
   for (int i=0; i < K; i++) { h_ConvKernel[i] = (double)(rand() % RG); }
   
   cudaMemcpy(d_Signal,      h_Signal,       N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ConvKernel,  h_ConvKernel,   K * sizeof(double), cudaMemcpyHostToDevice);

   h_convolution_1D(h_Signal, h_ConvKernel, h_Result_CPU, N, K);

   /*dim3 dimGrid(((width-1)/Mask_width)+1, 1,1);
   dim3 dimBlock(Mask_width,1, 1);*/
   d_convolution_1D_basic<<<dimGrid,dimBlock>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);

   cudaDeviceSynchronize();
   cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost);

   for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %f, gpu %f\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

   printf("Test basic passed\n");

   d_convolution_1D_caching<<<dimGrid,dimBlock>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);
   gpuErrchk(cudaPeekAtLastError());
   gpuErrchk(cudaDeviceSynchronize());

   gpuErrchk(cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost));

   for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %f, gpu %f\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

   printf("Test caching passed\n");

   return 0;
}