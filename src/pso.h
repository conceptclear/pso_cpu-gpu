#pragma once

#include <math.h>
#include <ctime>
#include <cstdlib>
#include <iostream>

class PSO
{
public:
    PSO(int max_iter, int num_particles, int num_dimensions, int use_cuda) : max_iter_(max_iter), num_particles_(num_particles), num_dimensions_(num_dimensions), use_cuda_(use_cuda)
    {
        position_ = new float[num_dimensions_ * num_particles_];
        velocity_ = new float[num_dimensions_ * num_particles_];
        pbest_ = new float[num_dimensions_ * num_particles_];
        gbest_ = new float[num_dimensions_];
        pbest_value_ = new float[num_particles_];
        min_range_ = NULL;
        max_range_ = NULL;
        input_position_ = NULL;
        c1_ = 2;
        c2_ = 2;
        max_velocity_ = 0.1;
        omega_init_ = 0.9;
        omega_end_ = 0.4;
        tolerance_ = 1e-6;
        max_stall_interations_ = 20;
    }

    ~PSO()
    {
        delete position_;
        delete velocity_;
        delete pbest_;
        delete gbest_;
        delete pbest_value_;
    }

    void setRange(float *min_range, float *max_range)
    {
        min_range_ = min_range;
        max_range_ = max_range;
    }

    void setLearningFactor(float c1, float c2)
    {
        c1_ = c1;
        c2_ = c2;
    }

    void setMaxVelocity(float max_velocity)
    {
        max_velocity_ = max_velocity;
    }

    void setTolerance(float tolerance)
    {
        tolerance_ = tolerance;
    }

    void setInitial(float *position)
    {
        input_position_ = position;
    }

    void setOmega(float omega_init, float omega_end)
    {
        omega_init_ = omega_init;
        omega_end_ = omega_end;
    }

    void setMaxStallIterations(int max_stall_iterations)
    {
        max_stall_interations_ = max_stall_iterations;
    }

    void getResult(float (*fitnessFunction)(float *));
    void printPbest();
    void printPbestValue();
    void printResult();

private:
    float getRandom();
    float getRandomRange(float low, float high);
    void getResultCPU(float (*fitnessFunction)(float *));
    void getResultCUDA();
    void init();
    void initWithoutInput();
    void initWithInput();
    bool initCuda();

    int max_iter_;
    int num_particles_;
    int num_dimensions_;
    bool use_cuda_;
    float c1_;
    float c2_;
    float max_velocity_;
    float *max_range_; //max constraints
    float *min_range_; //min constraints
    float *position_;
    float *velocity_;
    float *pbest_;
    float *gbest_;
    float *pbest_value_;
    float omega_init_;
    float omega_end_;
    float tolerance_;
    int max_stall_interations_;
    float *input_position_;
};