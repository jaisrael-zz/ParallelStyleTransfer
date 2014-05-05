#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweep_kernel(int N, int d, int* arr)
{
    int index = (2*d-1) + 2*d*(blockIdx.x * blockDim.x + threadIdx.x);
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    if (index % (2*d) == 2*d - 1)
       arr[index] = arr[index] + arr[index - d];    

}
/*
__global__ void upsweep_kernel(int N, int d, int* start, int* result)
{
    int index = (2*d-1) + 2*d*(blockIdx.x * blockDim.x + threadIdx.x);
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    if (index % (2*d) == 2*d - 1)
    {
       result[index] = start[index] + start[index - d];    
    }
    else
       result[index] = start[index];

}
*/
__global__ void downsweep_kernel(int N, int d, int* arr)
{
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (2*d-1) + 2*d*(blockIdx.x * blockDim.x + threadIdx.x);

    

    if (index >= N) return;

    if (index % (2*d) == 2*d - 1)
    {
       if ((d*2) == N && index == N-1)
       {
           arr[index] = 0;
       }
       int tmp = arr[index];
       arr[index] = arr[index] + arr[index-d];
       arr[index-d] = tmp;
    }
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
     int ceil_length = nextPow2(length);
     
     //cudaMemcpy(device_result, device_start, ceil_length * sizeof(int), cudaMemcpyDeviceToDevice);
     const int threadsPerBlock = 512;
     int init_blocks = (ceil_length + threadsPerBlock - 1) / threadsPerBlock;
     int blocks = init_blocks;
     blocks /= 2;
     if (blocks == 0) blocks = 1;

     for (int d=1; d < ceil_length/2; d*=2)
     {
         //printf("num blocks: %d\td: %d\n", blocks, d);
         upsweep_kernel<<<blocks, threadsPerBlock>>>(ceil_length, d, device_result);
         cudaThreadSynchronize();
         blocks /= 2;
         if (blocks == 0) blocks = 1;
     }
     
     for (int d=ceil_length/2; d > 0; d/=2)
     {
         
         //printf("num blocks: %d\td: %d\n", blocks, d);
         downsweep_kernel<<<blocks, threadsPerBlock>>>(ceil_length, d, device_result);
         cudaThreadSynchronize();
         blocks *= 2;
         if (blocks > init_blocks) blocks = init_blocks;
     }

//     cudaMemcpy(device_result, device_start, ceil_length * sizeof(int), cudaMemcpyDeviceToDevice);

}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void equals_next(int length, int* arr, int* result)
{
  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)  return;

       
    if (index == length - 1)  result[index] = 0;
    if (arr[index] == arr[index+1]) result[index] = 1;
    else result[index] = 0;
    //printf("index: %d start: %d result: %d\n", index, arr[index], result[index]);

}

__global__ void repeat_indices(int length, int* prefix_array, int* binary_array, int* repeat_count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) return;
    if (index == length - 1)
    {
         *repeat_count = prefix_array[index];
         return;
    }
    //printf("index: %d prefix: %d\n", index, prefix_array[index]);
    if (index == 0) return;
    //printf("index: %d binary: %d prefix: %d\n", index, binary_array[index], prefix_array[index]);
    if (prefix_array[index-1] != prefix_array[index])
          binary_array[prefix_array[index-1]] = index-1;
   // printf("index: %d binary: %d prefix: %d\n", index, binary_array[index], prefix_array[index]);
}

__global__ void print_stuff(int length, int* arr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) return;
    printf("index: %d result: %d\n", index, arr[index]);
}
int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */

    const int threadsPerBlock = 512;
    const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
     
    equals_next<<<blocks, threadsPerBlock>>>(length, device_input, device_output);
    
    cudaThreadSynchronize();
    exclusive_scan(device_output, length, device_output);
    

    int repeat_count_h;
    int* repeat_count_d;

    cudaMalloc(&repeat_count_d, sizeof(int));
    
    repeat_indices<<<blocks, threadsPerBlock>>>(length, device_output, device_input, repeat_count_d);
    cudaThreadSynchronize();
    //print_stuff<<<blocks, threadsPerBlock>>>(length, device_input);
    //cudaThreadSynchronize();
     
    cudaMemcpy(&repeat_count_h, repeat_count_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(repeat_count_d);
    
    cudaMemcpy(device_output, device_input, repeat_count_h*sizeof(int), cudaMemcpyDeviceToDevice);
    //device_output = device_input; 

    return repeat_count_h;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
