#include "pso.h"

int main()
{
    PSO problem(10000,10,2,false);
    float max_range[2] = {15,20};
    float min_range[2] = {-10,-15};
    float input_position[2] = {0,0};
    problem.setRange(min_range,max_range);
    problem.setInitial(input_position);
    problem.getResult();
    problem.printResult();
    return 0;
}