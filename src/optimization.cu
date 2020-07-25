#include "optimization.h"

__host__ __device__ float fitnessFunction(float* x)
{
    float res = x[0] * pow(2.71828,-(x[0]*x[0]+x[1]*x[1]));
    return res;
}
