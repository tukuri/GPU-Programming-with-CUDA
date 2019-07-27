// CS179, 2019
// Sung Hoon Choi
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdio.h>
#include <cmath> // Included for the fabs() and pow() 
#include <cuda_runtime.h> 
#include "multi_regression.cuh" // The header file for constants (hyperparameters and parameters)
#include "helper_cuda.h"

// NOTE
// Depending on the input data, the hyperparameters (e.g. learning rate) might need to be adjusted.
// For example, if learning rate is too high for the given data, the gradient descent might explode (diverge)
// On the other hand, if learning rate is too small, then training would take too much time. 


// cpuCalculateSqrtMeanCost
// This function computes the square mean error of the current hypothesis, which depend on the current parameters.
// The hypothesis is computed by the sum of products of parameters and predictors.
// It returns the total cost of the current hypothesis.
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
float cpuCalculateSqrtMeanCost(float (*x)[NUM_PARAM], const float * y, int num_sample, float * parameters, int num_param){

    float hypothesis = 0.0; // hypothesis to be computed based on current parameters
    float cost = 0.0; // cost

    for (int sample_index = 0; sample_index < num_sample; sample_index++){
        for (int param_index = 0; param_index < num_param; param_index++){
            hypothesis += parameters[param_index] * x[sample_index][param_index]; // hypothesis = dot(parameters,x)
	                                                                                       
	}
        cost += pow((hypothesis - y[sample_index]), 2); // Accumulate square mean error. cost = sum(h_i(theta)-y_i)^2
        hypothesis = 0.0; // Reset the hypothesis to compute the hypothesis of the next data 
    }
    cost /= (2 * num_sample);  // Compute the final value of the cost. cost = (1/2m)* sum((h_i(theta)-y_i)^2) 
    return cost;
}


// cpuUpdateParamUsingGradient
// This function updates the parameters through the gradient descent and output the updated parameters.
// The function works in a three-step process.
// Step 1: Compute the hypotheses.
// Step 2: Compute the gradients using the hypotheses, responses, and predictors.
// Step 3: Update all parameters "simultaneuosly" by using the gradients.
// Input:
//        x         : predictor data x ([NUM_SAMPLE] x [NUM_PARAM])
//        y         : response data y ([NUM_SAMPLE] x 1)
//        num_sample: number of data samples 
//        parameters: current parameters (or coefficients)
//        num_param : number of parameters
//        learning_rate: learning rate of the gradient descent
// Output:
//        parameters: updated parameters after one iteration of the gradient descent 
// Return Value:
//        none 
void cpuUpdateParamUsingGradient(float (*x)[NUM_PARAM], const float * y,  int num_sample, float * parameters, int num_param, float learning_rate){
    float hypothesis = 0.0;
    float * gradient = new float[num_param]; // gradients for each parameter
    
    // We can see that computing gradients and updating parameters take a lot of for-loops and sequential processes for CPU. 
    for (int sample_index = 0; sample_index < num_sample; sample_index++){
        for (int param_index = 0; param_index < num_param; param_index++){
            hypothesis += parameters[param_index] * x[sample_index][param_index]; // Compute the hypothesis
        }
        for (int param_index = 0; param_index < num_param; param_index++){  
            gradient[param_index] += ((hypothesis - y[sample_index]) * x[sample_index][param_index]); // Compute the gradients for each parameter. Could be computed in parallel by using GPU
        }
	hypothesis = 0.0; // Reset the hypothesis h(theta) for the next datasample 
    }
    // Done computing the gradients for each parameter (except that it needs to be divided by num_sample, which will be handled below)
    // Update all parameters "simultaneously" by using the gradients.
    for (int param_index = 0; param_index < num_param; param_index++){
        gradient[param_index] = gradient[param_index] / num_sample; // gradient[param_index] = (1/m) * sum{(h(theta)-y)*x}
        //printf("gradient[%d]: %f\n", param_index, gradient[param_index]); // (Debugging) View the gradients
        parameters[param_index] = parameters[param_index] - learning_rate * gradient[param_index]; // Update the parameters using the gradients and learning rate
    }
    delete [] gradient;
}

// cpuMultiRegression
// This function runs the multivariate regression through graident descent using the cpuCalculateSqrtMeanCost and cpuUpdateParamUsingGradient
// It computes the square mean error before and after each iteration of parameter update.
// When the cost doesn't change any further (|new_cost-old_cost| < convergence), it stops the gradient descent.
// If the cost exceeds GRADIENT_EXPLODE(10^10), then it halts the gradient descent and print the error message to notify the user that 
// the gradient descent has diverged.
// It prints out the regression summary when the regression is complete.   
// Input:
//        x         : predictor data x ([NUM_SAMPLE] x [NUM_PARAM])
//        y         : response data y ([NUM_SAMPLE] x 1)
//        num_sample: number of data samples 
//        parameters: current parameters (or coefficients)
//        num_param : number of parameters
//        learning_rate: learning rate of the gradient descent
//        convergence: preset threshold of the convergence 
//        max_iteration: maximum number of iterations allowed for the multivariate regression
// Output:
//        prints out the regression summary(string) including the initial cost, final cost, and number of iterations taken
// Return Value:
//        none
void cpuMultiRegression(float (*x)[NUM_PARAM], const float * y, int num_sample, float * initial_parameters, float * regression_parameters, int num_param, float learning_rate, float convergence, int max_iteration) {
    float initial_cost; // initial cost based on initial parameter values
    float old_cost, new_cost; //costs before and after each gradient descent iteration. Needed to determine the convergence 
  
    for(int param_index = 0; param_index < num_param; param_index++){
	   regression_parameters[param_index] = initial_parameters[param_index]; // initialize the parameters with the given values
    }
    initial_cost =  cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param);
    for (int iteration = 0; iteration < max_iteration; iteration++){
        old_cost = cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param); // Calculate the cost before updating the parameters 	  
	if(iteration % 1000 == 0){  
	    printf("."); 
	    //printf("CPU iteration %d --------- \n", iteration); 
	    //printf("cost: %f\n", old_cost); 
	}
	cpuUpdateParamUsingGradient(x, y, num_sample, regression_parameters, num_param, learning_rate); //Update the parameters using gradient descent
	new_cost = cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param); // Calculate the cost before updating the parameters 	  

	// If the gradient explodes, stop the iterations. 
	if(new_cost > GRADIENT_EXPLODE){
	    printf("Gradient exploded. Halting the gradient descent iterations"); 
	    break;
	}	       
	// If the regression has converged, print out the summary and exit 
	if(fabs(new_cost-old_cost) < convergence){
       	    //printf("Gradient Descent converged at iteration %d with final cost: %f\n\n", iteration, new_cost);
	    printf("\n------------CPU regression success! (NUM_SAMPLE = %d, NUM_PARAMETERS = %d, LEARNING_RATE = %1.5f) ---------\n", num_sample, NUM_PARAM, learning_rate);
 	    printf("<< CPU Summary >>\n" ); 
	    printf("Initial cost :%5.2f  (iteration 0) ---> Final cost : %5.2f  (iteration %d)\n", initial_cost, new_cost, iteration); 
	    break;
        }
    }	

}

int main(int argc, char *argv[]){
    
    int max_iteration = 10e7;
    float learning_rate = 0.0005; // The gradient descient explodes when learning_rate is too large. 
    float convergence_threshold = 0.001;
    
    printf("CPU regression has started\n");   
    
    // The number of parameters (=dimension) include the bias term. Thus, the first columns of 'x's must be set to 1 
#if TEST == 1
    float testvector_x[NUM_SAMPLE][NUM_PARAM]={
   	{ 1.00,  1.15,  3.48,  3.79},
	{ 1.00,  3.07,  4.06, 11.51},
	{ 1.00,  5.70,  6.20, 17.61},
	{ 1.00,  4.55, 10.27, 19.27},
	{ 1.00,  9.72, 18.53, 15.03},
	{ 1.00,  9.13, 18.62, 26.74},
	{ 1.00, 12.38, 16.25, 37.06},
	{ 1.00,  8.17, 18.16, 26.79},
	{ 1.00, 11.79, 30.09, 39.72},
	{ 1.00, 18.16, 25.79, 51.99},
	{ 1.00, 18.73, 29.21, 44.04},
	{ 1.00, 23.74, 38.99, 70.21},
	{ 1.00, 22.98, 47.45, 54.86},
	{ 1.00, 20.32, 39.22, 83.80},
	{ 1.00, 17.66, 58.88, 63.87},
	{ 1.00, 22.78, 46.82, 65.94},
	{ 1.00, 24.91, 35.20, 55.30},
	{ 1.00, 31.19, 58.90, 55.51},
	{ 1.00, 24.70, 46.39, 60.14},
	{ 1.00, 30.46, 56.65, 62.89} 
    };
#elif TEST == 2 
    float testvector_x[NUM_SAMPLE][NUM_PARAM]={    
	{ 1.00,  6.14, 11.13, 15.98},
	{ 1.00,  6.35, 11.50, 17.13},
	{ 1.00,  5.41, 13.51, 16.43},
	{ 1.00,  6.29, 11.22, 16.02},
	{ 1.00,  8.27, 16.94, 18.59},
	{ 1.00, 15.70, 12.42, 16.33},
	{ 1.00, 12.39, 22.15, 18.67},
	{ 1.00, 12.99, 16.86, 23.48},
	{ 1.00, 15.33, 17.05, 16.43},
	{ 1.00, 22.35, 13.29, 31.07},
	{ 1.00, 11.33, 15.91, 30.78},
	{ 1.00, 26.15, 33.81, 26.00},
	{ 1.00, 25.41, 29.21, 27.31},
	{ 1.00, 18.58, 36.12, 36.07},
	{ 1.00,  9.78, 14.04, 27.32},
	{ 1.00, 29.21, 16.80, 25.05},
	{ 1.00, 11.84, 41.91, 32.15},
	{ 1.00, 26.73, 40.15, 36.97},
	{ 1.00, 32.95, 25.57, 33.61},
	{ 1.00, 25.36, 30.26, 39.73},
	{ 1.00, 40.05, 34.64, 15.89},
	{ 1.00, 37.37, 50.51, 48.25},
	{ 1.00, 18.10, 53.68, 53.63},
	{ 1.00, 44.12, 45.70, 58.91},
	{ 1.00, 49.81, 16.02, 33.73},
	{ 1.00, 18.71, 26.71, 52.80},
	{ 1.00, 44.85, 35.08, 56.26},
	{ 1.00, 18.54, 60.62, 47.01},
	{ 1.00, 23.17, 11.37, 45.16},
	{ 1.00, 39.82, 17.70, 61.04}
    };
#endif
#if TEST == 1
    float testvector_y[NUM_SAMPLE]={
	-33.28,
	-19.96,
	-15.86,
	-3.23,
	11.42,
	20.93,
	32.07,
	42.61,
	57.95,
	66.06,
	77.03,
	86.44,
	97.80,
	112.88,
	117.09,
	135.45,
	137.73,
	153.95,
	159.31,
	176.65,
    };
#elif TEST == 2
    float testvector_y[NUM_SAMPLE]={
	11.61,
	 9.77,
	22.46,
	28.88,
	28.63,
	41.93,
	43.92,
	47.49,
	54.89,
	61.68,
	70.22,
	75.51,
	80.09,
	84.83,
	94.22,
	101.12,
	99.25,
	113.57,
	117.47,
	119.88,
	131.81,
	132.74,
	135.75,
	145.14,
	154.75,
	159.63,
	162.77,
	164.10,
	171.65,
	179.22,
    };


#endif
 
    // The 4-dimensional testvector_x needs to be flattened into 1-dimensional array, in order to be handled by the CUDA functions 
    float * flat_testvector_x = new float[NUM_SAMPLE * NUM_PARAM];
    for (int row = 0; row < NUM_SAMPLE; row++){
	    for (int col = 0; col < NUM_PARAM; col++){
		    flat_testvector_x[row * NUM_PARAM + col] = testvector_x[row][col];
	    }
    }
   
    // Initial values of parameters  
    float * init_param = new float[NUM_PARAM]; 
    init_param[0] = 0.0; 
    // Initial values were determined by using two data points: the first data point and the last data point 
    for (int i = 1; i < NUM_PARAM; i++){
	    init_param[i] = (testvector_y[NUM_SAMPLE-1] - testvector_y[0])/(testvector_x[NUM_SAMPLE-1][i] - testvector_x[0][i]);
            //printf("initial param[%d] = %f", i, init_param[i]); // Uncomment this to observe the values of the initial parameters
    }
    
    // The final parameters after the regression is done 
    float * result_param = new float[NUM_PARAM];
    
    // Run the CPU regression 
    cpuMultiRegression(testvector_x, testvector_y, NUM_SAMPLE, init_param, result_param, NUM_PARAM, learning_rate, convergence_threshold, max_iteration);   	
    for (int param_index = 0; param_index < NUM_PARAM; param_index++){
	   printf("parameter %d: %f\n", param_index, result_param[param_index]);
    }
	
    // GPU regression starts
    printf("\nGPU regression has started\n");
    float *d_testvector_x;
    float *d_testvector_y; 
    float *d_init_param; 
    float *d_result_param; 
   
    // Allocate device memories for testvectors and parameters 
    CUDA_CALL(cudaMalloc(&d_testvector_x, sizeof(float) * NUM_SAMPLE * NUM_PARAM));
    CUDA_CALL(cudaMalloc(&d_testvector_y, NUM_SAMPLE * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_result_param, NUM_PARAM * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_init_param, NUM_PARAM * sizeof(float)));
    
    // Initialize the device memories
    CUDA_CALL(cudaMemcpy(d_testvector_x, flat_testvector_x, NUM_SAMPLE * NUM_PARAM * sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CALL(cudaMemcpy(d_testvector_y, testvector_y, NUM_SAMPLE * sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CALL(cudaMemcpy(d_init_param, init_param, NUM_PARAM * sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CALL(cudaMemcpy(d_result_param, init_param, NUM_PARAM * sizeof(float), cudaMemcpyHostToDevice)); 
  
    // Run the GPU regression
    cudaMultiRegression(d_testvector_x, d_testvector_y, NUM_SAMPLE, d_init_param, d_result_param, NUM_PARAM, learning_rate, convergence_threshold, max_iteration); 
    // Copy the result parameters in device memory back to the host memory 
    CUDA_CALL(cudaMemcpy(result_param, d_result_param, NUM_PARAM * sizeof(float), cudaMemcpyDeviceToHost)); 
    for (int param_index = 0; param_index < NUM_PARAM; param_index++){
	   printf("parameter %d: %f\n", param_index, result_param[param_index]);
    }

    // Free the device memories
    cudaFree(d_testvector_x);
    cudaFree(d_testvector_y);
    cudaFree(d_init_param); 
    cudaFree(d_result_param);

    // Free the new-generated arrays
    delete [] init_param;
    delete [] result_param;
    delete [] flat_testvector_x; 
}
