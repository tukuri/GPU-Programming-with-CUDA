// CS179, 2019
// Sung Hoon Choi
#define NUM_PARAM 4
#if TEST == 1
    #define NUM_SAMPLE 20
#elif TEST == 2
    #define NUM_SAMPLE 30
#endif
#define GRADIENT_EXPLODE 10e10
#define RUN_REGRESSION 0
#define VALIDATE_REGRESSION 1


void cudaMultiRegression(float *x, const float * y, int num_sample, float * initial_parameters, float * regression_parameters, int num_param, float learning_rate, float convergence, int max_iteration);

