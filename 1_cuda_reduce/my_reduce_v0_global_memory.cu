#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

bool check(float* d_res, float* h_res, int N)
{
    for (uint64_t i = 0; i < N; i++) {
        if (abs(d_res[i] - h_res[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

//__global__ void reduce_v0(float *d_input, float *d_output)
//{
//    // 每个block计算blockDim.x个数的和
//    float *input_begin = d_input + blockDim.x * blockIdx.x;
////    input_begin[0] += input_begin[1];
////    if (threadIdx.x == 0 or 2 or 4 or 6)
////        input_begin[threadIdx.x] += input_begin[threadIdx.x + 1];
////    if (threadIdx.x == 0 or 4)
////        input_begin[threadIdx.x] += input_begin[threadIdx.x + 2];
////    if (threadIdx.x == 0)
////        input_begin[threadIdx.x] += input_begin[threadIdx.x + 4];
//    for (int i = 0; i < blockDim.x; i *= 2) {
//        if (threadIdx.x % (i * 2) == 0) {
//            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
//            __syncthreads();
//        }
//    }
//    if (threadIdx.x == 0) {
//        d_output[blockIdx.x] = input_begin[0];
//    }
//}

__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        if(tid%(2*s) == 0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

int main()
{
    const int N = 1024;
    float* input = (float *) malloc(N * sizeof (float ));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float ));

    int block_num = N / THREAD_PER_BLOCK;
    float* output = (float *) malloc((N / THREAD_PER_BLOCK) * sizeof(float ));
    float *d_output;
    cudaMalloc((void**)&d_output, (N / THREAD_PER_BLOCK) * sizeof(float ));
    float *res = (float*) malloc((N / THREAD_PER_BLOCK) * sizeof(float ));

    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * i - 1.0;
    }
    // cpu cal
    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_input, input, N* sizeof (float ), cudaMemcpyHostToDevice);
    dim3 Grid(N / THREAD_PER_BLOCK, 1);
    dim3 block(THREAD_PER_BLOCK, 1);

    reduce0<<<Grid, block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num * sizeof(float ), cudaMemcpyDeviceToHost);
    if (check(d_output, res, block_num)) {
        printf("ans is ok\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
