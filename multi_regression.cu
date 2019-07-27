// CS179, 2019
// Sung Hoon Choi
#include <cassert>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdio.h>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "multi_regression.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "helper_cuda.h"


// <T>cudaMemsetType
// Sets all entries in a device buffer of floats equal to a specified value.
// This template was taken from lab5 utils.cu
template<typename T> void cudaMemsetType(T *dev_ptr, T val, int n_vals)
{
	thrust::device_ptr<T> thrust_dev_ptr(dev_ptr);
	thrust::fill(thrust_dev_ptr, thrust_dev_ptr + n_vals, val);
}

// cudaCalculateSqrtMeanCost
// This kernel computes the square mean error of the current hypothesis, which depend on the current parameters.
// The hypothesis is computed by the sum of products of parameters and predictors.
// It returns the total cost of the current hypothesis.
// It uses the sequential addressing reduction to reduce the frequency of accessing the global variable(cost) and to
// prevent the bank conflicts.
//
// GPU Optimizations:
//        sequential addressing reduction 
// Input:
//        x         : predictor data x ([NUM_SAMPLE] x [NUM_PARAM])
//        y         : response data y ([NUM_SAMPLE] x 1)
//        num_sample: number of data samples 
//        parameters: current parameters (or coefficients)
//        num_param : number of parameters
// Output:
//        none
// Return Value:
//        cost: the square mean error of the hypothesis based on current parameters
__global__
void cudaCalculateSqrtMeanCost(float *x, const float * y, int num_sample, float * parameters, int num_param, float *cost){

    float hypothesis = 0.0;
    extern __shared__ float shmem[]; // shared memory that holds the cost
   
    // Reset the shared memory before computing the cost. 
    // cudaCalculateSqrtMeanCost gets called repeatedly, and not resetting the shared memory would produce an incorrect result. 
    for (int ix = threadIdx.x; ix < 1024 ; ix += blockDim.x){
	    shmem[ix] = 0.0;
    }
    __syncthreads();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int blockId = blockIdx.x; //Uncomment the declaration of blockId if you want to activate the debug printf below.
  
    while(i < num_sample){
	for(int j = 0; j < num_param ; j++){
   	     hypothesis += parameters[j] * x[i * num_param + j]; // Compute the hypotheses
	}	    
        //printf("thread = %d hypothesis = %f\n", tid, hypothesis);	
	shmem[tid] = pow(hypothesis - y[i], 2); // compute (h_i-y_i)^2 
	i += blockDim.x * gridDim.x;
    }

    __syncthreads(); 
    
    // Compute the sum by using the sequential addressing reduction 
    for (unsigned s = blockDim.x / 2; s > 0 ; s >>= 1)
    {
        if(tid < s){
	    //printf("shmem[%d]=%f\n shmem[%d+%d]=%f\n", tid, shmem[tid], tid,s, shmem[tid+s]);
	    shmem[tid] += shmem[tid + s]; 
	}
	__syncthreads();

    }

    // The sum of cost per each blcok is stored at shmem[0]
    // To obtain the final sum in the global variable cost, we use atomicAdd
    if (threadIdx.x == 0){
	    atomicAdd(cost, shmem[0] / static_cast<float>(2 * num_sample)); // cost = (sum(h_i-y_i)^2)/(2m)
            //printf("Atomic Add -- thread %d block %d cost %f\n" , tid, blockId,  *cost);
            // If you want to activate this printf, be sure to uncomment the declaration of blockId above as well. 

    }
    
}


// cudaUpdateParamUsingGradient
// This kernel updates the parameters through the gradient descent and output the updated parameters.
// The function works in a three-step process.
// Step 1: Compute the hypotheses.
// Step 2: Compute the gradients using the hypotheses, responses, and predictors.
// Step 3: Update all parameters "simultaneuosly" by using the gradients.
// It uses sequential addressing reduction and loop unrolling to boost the performance.
// Sequential addressing is free from bank conflict, and the loop unrolling reduces loop overhead.
// Loop unrolling is available since the gradients for each parameter are independent from each other.
//
// GPU Optimizations :
//        sequential addressing reduction
//        loop unrolling
// Input:
//        x         : predictor data x ([NUM_SAMPLE] x [NUM_PARAM])
//        y         : response data y ([NUM_SAMPLE] x 1)
//        gradients : gradients used for updating the parameters
//        num_sample: number of data samples 
//        parameters: current parameters (or coefficients)
//        num_param : number of parameters
//        learning_rate: learning rate of the gradient descent
// Output:
//        parameters: updated parameters after one iteration of the gradient descent 
// Return Value:
//        none 
__global__
void cudaUpdateParamUsingGradient(float *x, const float * y,  float * gradients, int num_sample, float * parameters, int num_param, float learning_rate){
    extern __shared__ float sh_gradient[]; // Gradients in shared memory
    float hypothesis = 0.0; 
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
 
    //Reset the shared memory before computing the cost. 
    //cudaCalculateSqrtMeanCost gets called repeatedly, and not resetting the shared memory would produce an incorrect result. 
    for (int ix = threadIdx.x; ix < 512 ; ix += blockDim.x){
	    sh_gradient[ix] = 0.0; 
    }
    __syncthreads();


    while(i < num_sample){    
        for(int j = 0; j < num_param ; j++){
   	     hypothesis += parameters[j] * x[i * num_param + j]; //Compute the hypotheses
	}
	//printf("Update thread = %d hypothesis = %f\n", tid, hypothesis);
	sh_gradient[tid*NUM_PARAM+0] = ((hypothesis - y[i]) * x[i * NUM_PARAM + 0]); // Compute (h_i-y_i)*x_i0 (ith data, 0th parameter)
	sh_gradient[tid*NUM_PARAM+1] = ((hypothesis - y[i]) * x[i * NUM_PARAM + 1]); // Compute (h_i-y_i)*x_i1 (ith data, 1st parameter)
	sh_gradient[tid*NUM_PARAM+2] = ((hypothesis - y[i]) * x[i * NUM_PARAM + 2]); // Compute (h_i-y_i)*x_i2 (ith data, 2th parameter)
        sh_gradient[tid*NUM_PARAM+3] = ((hypothesis - y[i]) * x[i * NUM_PARAM + 3]); // Compute (h_i-y_i)*x_i3 (ith data, 3rd parameter)
        i += blockDim.x * gridDim.x; 
	}
    
    __syncthreads();

    // Use sequential addressing to take the sum to obtain the gradients for each parameter
    for (unsigned s = blockDim.x / 2; s > 0 ; s >>= 1)
    {
        if(tid < s){
	    sh_gradient[tid*NUM_PARAM+0] += sh_gradient[(tid + s)*NUM_PARAM+0];
	    sh_gradient[tid*NUM_PARAM+1] += sh_gradient[(tid + s)*NUM_PARAM+1];
	    sh_gradient[tid*NUM_PARAM+2] += sh_gradient[(tid + s)*NUM_PARAM+2];
	    sh_gradient[tid*NUM_PARAM+3] += sh_gradient[(tid + s)*NUM_PARAM+3];
	} 
	    __syncthreads();
    }
  
    // Finally, use atomicAdd to obtain the final gradients in the global variable (gradients)
    if (threadIdx.x == 0){
	atomicAdd(gradients + 0, sh_gradient[0*NUM_PARAM+0] / static_cast<float>(num_sample)); // gradient[0] = gradient for parameter 0
	atomicAdd(gradients + 1, sh_gradient[0*NUM_PARAM+1] / static_cast<float>(num_sample)); // gradient[1] = gradient for parameter 1
        atomicAdd(gradients + 2, sh_gradient[0*NUM_PARAM+2] / static_cast<float>(num_sample)); // gradient[2] = gradient for parameter 2
        atomicAdd(gradients + 3, sh_gradient[0*NUM_PARAM+3] / static_cast<float>(num_sample)); // gradient[3] = gradient for parameter 3
   
   	// Update allthe parameters using the gradients "simultaneously".
        parameters[0] = parameters[0] - learning_rate * gradients[0];
        parameters[1] = parameters[1] - learning_rate * gradients[1];
        parameters[2] = parameters[2] - learning_rate * gradients[2];
        parameters[3] = parameters[3] - learning_rate * gradients[3];

    }
}

// cudaMultiRegression
// This function runs the multivariate regression through graident descent calling the cudaCalculateSqrtMeanCost and cudaUpdateParamUsingGradient kernels.
// It computes the square mean error before and after each iteration of parameter update.
// When the cost doesn't change any further (|new_cost-old_cost| < convergence), it stops the gradient descent.
// If the cost exceeds GRADIENT_EXPLODE(10^10), then it halts the gradient descent and print the error message to notify the user that 
// the gradient descent has diverged.
// It prints out the regression summary when the regression is complete.   
// Input:
//        x         : predictor data x ([NUM_SAMPLE] x [NUM_PARAM])
//        y         : response data y ([NUM_SAMPLE] x 1)
//        num_sample: number of data samples 
//        initial_parameters: initial values of parameters before starting the gradient descents 
//        d_regression_parameters: final regression parameters (coefficients) 
//        num_param : number of parameters
//        learning_rate: learning rate of the gradient descent
//        convergence: preset threshold of the convergence 
//        max_iteration: maximum number of iterations allowed for the multivariate regression
// Output:
//        prints out the regression summary(string) including the initial cost, final cost, and number of iterations taken
// Return Value:
//        none
void cudaMultiRegression(float *x, const float * y, int num_sample, float * initial_parameters, float * d_regression_parameters, int num_param, float learning_rate, float convergence, int max_iteration){
    float *d_initial_cost;
    float initial_cost;
    float old_cost, new_cost; 
    float *d_old_cost, *d_new_cost;
    float *d_gradients;
    float reset_gradients[num_param] = {0.0, 0.0, 0.0, 0.0}; 
   
    // allocate the device memories for costs and gradients 
    CUDA_CALL(cudaMalloc(&d_initial_cost, sizeof(float))); 
    CUDA_CALL(cudaMalloc(&d_old_cost, sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_new_cost, sizeof(float)));    
    CUDA_CALL(cudaMalloc(&d_gradients, num_param * sizeof(float)));

    // initialize the costs and gradients in the device memory 
    cudaMemsetType<float>(d_initial_cost, 0.0, 1); 
    cudaMemsetType<float>(d_old_cost, 0.0, 1); 
    cudaMemsetType<float>(d_new_cost, 0.0, 1); 
    CUDA_CALL(cudaMemcpy(d_gradients, reset_gradients, num_param * sizeof(float), cudaMemcpyHostToDevice)); 
    
    // Copy the values of the initial parameters into the device memories of regression parameters to start the regression 
    CUDA_CALL(cudaMemcpy(d_regression_parameters, initial_parameters, num_param * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate the initial cost before starting the regression 
    cudaCalculateSqrtMeanCost<<<1, 256, 1024 * sizeof(float)>>>(x, y, num_sample, d_regression_parameters, num_param, d_initial_cost);
    // Copy the initial memory in device memory back to the host memory to observe its value 
    CUDA_CALL(cudaMemcpy(&initial_cost, d_initial_cost, sizeof(float), cudaMemcpyDeviceToHost)); 
     
    for (int iteration = 0; iteration < max_iteration; iteration++){ 
        // Reset the costs at the beginning of each iteration 
	cudaMemsetType<float>(d_old_cost, 0.0, 1); 
	cudaMemsetType<float>(d_new_cost, 0.0, 1); 
	
        //Compute the cost before running the gradient descent 
	cudaCalculateSqrtMeanCost<<<1, 256, 1024*sizeof(float)>>>(x, y, num_sample, d_regression_parameters, num_param, d_old_cost); // Calculate the cost before updating the parameters 	  
        // Copy the old cost in device memory back to the host memory	
	CUDA_CALL(cudaMemcpy(&old_cost, d_old_cost, sizeof(float), cudaMemcpyDeviceToHost)); 
	if(iteration % 1000 == 0){  
	    printf("."); 
	    //printf("GPU iteration %d -----\n", iteration); 
	    //printf("cost: %f\n", old_cost);
	    //printf("Updating parameters..........\n");
	}
 
        // Update the parameters through gradient descent	
	cudaUpdateParamUsingGradient<<<1, 256, 1024*sizeof(float)>>>(x, y, d_gradients, num_sample, d_regression_parameters, num_param, learning_rate); //Update the parameters using gradient descent
        // Reset the gradients for the next iteration	
	CUDA_CALL(cudaMemcpy(d_gradients, reset_gradients, num_param * sizeof(float), cudaMemcpyHostToDevice)); 
        // Compute the cost after running the gradient descent	
	cudaCalculateSqrtMeanCost<<<1, 256, 1024*sizeof(float)>>>(x, y, num_sample, d_regression_parameters, num_param, d_new_cost); // Calculate the cost before updating the parameters 	  
	// Copy the new cost in device memory back to the host memory	
	CUDA_CALL(cudaMemcpy(&new_cost, d_new_cost, sizeof(float), cudaMemcpyDeviceToHost)); 
	// If the gradient explodes, stop the iterations. 
	if(new_cost > GRADIENT_EXPLODE){
	    printf("Gradient exploded. Halting the gradient descent iterations"); 
	    break;
	}	       
	// If the regression has converged, print out the regression summary and exit
	if(fabs(new_cost-old_cost) < convergence){
	    //printf("Gradient Descent converged at iteration %d with final cost: %f\n\n", iteration, new_cost);
	    printf("\n------------GPU regression success! (NUM_SAMPLE = %d, NUM_PARAMETERS = %d, LEARNING_RATE = %1.5f) ---------\n", num_sample, NUM_PARAM, learning_rate);
	    printf("<< GPU Summary >>\n" ); 
	    printf("Initial cost :%7.2f  (iteration 0) ---> Final cost : %7.2f  (iteration %d)\n", initial_cost, new_cost, iteration); 
	    break;
        }
	
    }
   
    // Free the device memories 
    CUDA_CALL(cudaFree(d_initial_cost));
    CUDA_CALL(cudaFree(d_old_cost));
    CUDA_CALL(cudaFree(d_new_cost)); 
    CUDA_CALL(cudaFree(d_gradients));
}
