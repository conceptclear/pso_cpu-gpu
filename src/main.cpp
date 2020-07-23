#include "pso.h"

float fitnessFunction(float* x)
{
    float res = x[0] * pow(2.71828,-(x[0]*x[0]+x[1]*x[1]));
    return res;
}

int main()
{
    PSO problem(10000,1000,2,false);
    float max_range[2] = {15,20};
    float min_range[2] = {-10,-15};
    float input_position[2] = {0,0};
    problem.setRange(min_range,max_range);
    problem.setInitial(input_position);
    problem.getResult(fitnessFunction);
    problem.printResult();
    return 0;
}