#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdio.h>
#include <cmath> // Included for the fabs() and pow() 
#include <cuda_runtime.h> 

#define NUM_PARAM 2
#define GRADIENT_EXPLODE 10e7
#define RUN_REGRESSION 0
#define VALIDATE_REGRESSION 1


// NOTE
// Depending on the input data, one needs to change the meta-data (e.g. learning rate) into appropriate values.
// For example, if learning rate is too high for the given data, the gradient descent might explode (diverge)
// On the other hand, the learning rate can be too small for the given data. In this case, learning would be very slow.

// cpuCalculateSqrtMeanCost 
// Input:
//        x: sample x  - Dimension: [num_sample] x [num_param]
//        y: sample y  - Dimension: [num_sample] x 1
//        num_sample: number of samples (m)
//        parameters: the parameters found by running the linear regression
//        num_param: number of parameters, or features (n)
// Output:
//        none
// Return value:
//        cost - the square mean error of the hypothesis based on current parameters
float cpuCalculateSqrtMeanCost(float (*x)[NUM_PARAM], const float * y, int num_sample, float * parameters, int num_param){

    float hypothesis = 0.0;
    float cost = 0.0;

    for (int sample_index = 0; sample_index < num_sample; sample_index++){
        for (int param_index = 0; param_index < num_param; param_index++){
            hypothesis += parameters[param_index] * x[sample_index][param_index]; // hypothesis = sum(parameters * x)
	                                                                                        // In case of the GPU implementation using CUDA library, we could use the matrix dot function to accelerate the computations
	}
	// h(theta) calculated
        cost += pow((hypothesis - y[sample_index]), 2); // cost += (h_i(theta) - y_i)^2
        hypothesis = 0.0; // Reset the hypothesis h(theta) for the next sample
    }
    cost /= (2 * num_sample);  // cost = (1/2m)* sum((h_i(theta)-y_i)^2) 
    return cost;
}


// cpuUpdateUsingGradient
// Input:
//        x: sample x  - Dimension: [num_sample] x [num_param]
//        y: sample y  - Dimension: [num_sample] x 1
//        num_sample: number of samples (m)
//        parameters: the parameters found by running the linear regression
//        num_param: number of parameters, or features (n)
//        learning_rate: learning rate for gradient descent
// Output:
//        updated_parameters: updated parameters to be applied simultaneously 
// Return value:
//        none 
// float (*x)[NUM_PARAM]
void cpuUpdateParamUsingGradient(float (*x)[NUM_PARAM], const float * y,  int num_sample, float * parameters, int num_param, float learning_rate){
    float hypothesis = 0.0;
    float * gradient = new float[num_param];
    
    // We can see that the CPU implementation without matrix operation libraries for computing gradients and updating the parameters takes a number of redundant For-loops. 
    // For the GPU implementation, we would be able to utilize the parallel computation along with various matrix operation libraries. The codes would become much more concise.
    for (int sample_index = 0; sample_index < num_sample; sample_index++){
        for (int param_index = 0; param_index < num_param; param_index++){
            hypothesis += parameters[param_index] * x[sample_index][param_index]; //For GPU implementation, computing the hypothesis can be simply done by using a matrix dot function
        }
        for (int param_index = 0; param_index < num_param; param_index++){  
            gradient[param_index] += ((hypothesis - y[sample_index]) * x[sample_index][param_index]); //Calculating the gradients for each features(parameters). Could be computed in parallel by using GPU
        }
	hypothesis = 0.0; // Reset the hypothesis h(theta) for the next sample 
    }
    // Done computing the gradients for each parameters(except that it needs to be scaled down by num_sample, which will be handled below)
    // Update allthe parameters using the gradients "simultaneously".
    for (int param_index = 0; param_index < num_param; param_index++){
        gradient[param_index] = gradient[param_index] / num_sample; // gradient[param_index] = (1/m) * sum{(h(theta)-y)*x}
        //printf("gradient[%d]: %f\n", param_index, gradient[param_index]); // View the gradients for debugging. You can activate/deactivate this while running the program
        parameters[param_index] = parameters[param_index] - learning_rate * gradient[param_index];  
    }
}

// cpuLinearRegression
// Input:
//        x: sample x  - Dimension: [num_sample] x [num_param]
//        y: sample y  - Dimension: [num_sample] x 1
//        num_sample: number of samples (m)
//        num_param: number of parameters, or features (n)
//        learning_rate: learning rate for gradient descent 
//        convergence: Admitted range of the convergence 
//        max_iteration: maximum number of iterations to be run for the linear regression
//        validate: whether it's running the regression or validating the regression results with parameters obtained from MATLAB
// Output:
//        regression_parameters: the parameters derived by the linear regression
// Return value:
//        none
void cpuLinearRegression(float (*x)[NUM_PARAM], const float * y, int num_sample, float * initial_parameters, float * regression_parameters, int num_param, float learning_rate, float convergence, int max_iteration, bool validate){
    float initial_cost, old_cost, new_cost;    
  
    for(int param_index = 0; param_index < num_param; param_index++){
	   regression_parameters[param_index] = initial_parameters[param_index]; // initialize the parameters with the given values
    }
    initial_cost =  cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param);
    for (int iteration = 0; iteration < max_iteration; iteration++){
        old_cost = cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param); // Calculate the cost before updating the parameters 	  
	if(iteration % 100 == 0){  
	    printf("iteration %d------------------------\n", iteration); 
	    printf("cost: %f\n", old_cost); 
	}
	cpuUpdateParamUsingGradient(x, y, num_sample, regression_parameters, num_param, learning_rate); //Update the parameters using gradient descent
	new_cost = cpuCalculateSqrtMeanCost(x, y, num_sample, regression_parameters, num_param); // Calculate the cost before updating the parameters 	  

	// If the gradient explodes, stop the iterations. 
	if(new_cost > GRADIENT_EXPLODE){
	    printf("Gradient exploded. Halting the gradient descent iterations"); 
	    break;
	}	       
	// If the regression has converged, exit the iterations
	if(fabs(new_cost-old_cost) < convergence){
            if(!validate){
	        printf("Gradient Descent converged at iteration %d with final cost: %f\n\n", iteration, new_cost);
	        printf("----------------------CPU regression success! (NUM_SAMPLE = %d, NUM_PARAMETERS = %d, LEARNING_RATE = %f) -------------------\n", num_sample, NUM_PARAM, learning_rate);
	        printf("<< Summary >>\n" ); 
	        printf("Initial cost :%f  (iteration 0)\nFinal cost : %f  (iteration %d)\n", initial_cost, new_cost, iteration); 
	    }
	    else
               printf("The final cost calculated by using the parameters from MATLAB is %f\n", new_cost);	
	    break;
        }
    }	

}


int main(int argc, char *argv[]){
    int num_sample = 12;     
    float learning_rate = 0.0001; // The gradient descient explodes when learning_rate is too large. 
    float convergence_threshold = 0.0001;
    int max_iteration = 10e6;
    
    printf("cpu regression started\n");   
    // The number of parameters (=dimension) include the bias term. Thus, the first columns of 'x's must be set to 1 
    
    float cpu_test_x[num_sample][NUM_PARAM] = 
    {
	    {1.0, 1.0},
	    {1.0, 1.1},
	    {1.0, 3.0},
	    {1.0, 5.0},
	    {1.0, 10.0},
	    {1.0, 12.0},
	    {1.0, 20.0},
	    {1.0, 34.0},
	    {1.0, 40.0},
	    {1.0, 41.3},
	    {1.0, 42.5},
	    {1.0, 42.7}
    };

    float cpu_test_y[num_sample] = 
    {
	    40.0,
	    55.0,
	    49.0,
	    51.2,
	    63.0,
	    51.0,
	    60.0,
	    67.0,
	    60.0,
	    62.5,
	    80.0,
	    70.5
    };
    
    float * init_param = new float[NUM_PARAM]; 
    init_param[0] = 0.0; 
    for (int i = 1; i < NUM_PARAM; i++){
	    init_param[i] = (cpu_test_y[num_sample-1] - cpu_test_y[0])/(cpu_test_x[num_sample-1][i] - cpu_test_x[0][i]);
            //printf("initial param[%d] = %f", i, init_param[i]); 
    }
    
    float MATLAB_param[NUM_PARAM] = 
    {
        48.8288,
        0.4882	
    };
     
    float *result_param = new float[NUM_PARAM];
    cpuLinearRegression(cpu_test_x, cpu_test_y, num_sample, init_param, result_param, NUM_PARAM, learning_rate, convergence_threshold, max_iteration, RUN_REGRESSION);   	
    for (int param_index = 0; param_index < NUM_PARAM; param_index++){
	   printf("parameter %d: %f\n", param_index, result_param[param_index]);
    }
     
    printf("\n\n -------- Now validate the linear-regression by using the parameters obtained from MATLAB regression ------- \n");
    printf("The parameters obtained from MATLAB are: \n");
    printf("parameter 0: %f\n", MATLAB_param[0]);
    printf("parameter 1: %f\n", MATLAB_param[1]);
    cpuLinearRegression(cpu_test_x, cpu_test_y, num_sample, MATLAB_param, result_param, NUM_PARAM, learning_rate, convergence_threshold, max_iteration, VALIDATE_REGRESSION);
}

