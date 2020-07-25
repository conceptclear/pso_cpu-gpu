#include "pso.h"
#include "sys/time.h"

float PSO::getRandom()
{
    return (float)rand() / (float)RAND_MAX;
}

float PSO::getRandomRange(float low, float high)
{
    return low + (float)(high - low) * getRandom();
}

void PSO::init()
{
    srand((int)time(0));
    if (input_position_ == NULL)
    {
        initWithoutInput();
    }
    else
    {
        initWithInput();
    }
}

void PSO::initWithoutInput()
{
    if (max_range_ == NULL || min_range_ == NULL)
    {
        for (int i = 0; i < num_dimensions_; i++)
        {
            for (int j = 0; j < num_particles_; j++)
            {
                position_[j * num_dimensions_ + i] = getRandom() * RAND_MAX - RAND_MAX;
                //std::cout << "position " << position_[j * num_dimensions_ + i] << std::endl;
                pbest_[j * num_dimensions_ + i] = position_[j * num_dimensions_ + i];
                velocity_[j * num_dimensions_ + i] = 0;
            }
            gbest_[i] = pbest_[i];
        }
    }
    else
    {
        for (int i = 0; i < num_dimensions_; i++)
        {
            for (int j = 0; j < num_particles_; j++)
            {
                position_[j * num_dimensions_ + i] = getRandomRange(min_range_[i], max_range_[i]);
                pbest_[j * num_dimensions_ + i] = position_[j * num_dimensions_ + i];
                velocity_[j * num_dimensions_ + i] = 0;
            }
            gbest_[i] = pbest_[i];
        }
    }
}

void PSO::initWithInput()
{
    initWithoutInput();
    for (int i = 0; i < num_dimensions_; i++)
    {
        position_[i] = input_position_[i];
        pbest_[i] = position_[i];
        gbest_[i] = position_[i];
    }
}

void PSO::getResult()
{
    if (use_cuda_)
    {
        getResultCUDA();
    }
    else
    {
        getResultCPU();
    }
}

void PSO::getResultCPU()
{
    init();
    struct timeval t1,t2;
    double time_use;
    float omega;
    int iter_counts = 0;
    float *temp_gbest = new float[num_dimensions_];

    gettimeofday(&t1,NULL);

    for (int i = 0; i < num_particles_; i++)
    {
        pbest_value_[i] = fitnessFunction(pbest_ + i * num_dimensions_);
    }

    float temp_value = 0;
    float gbest_value = fitnessFunction(gbest_);

    for (int iter = 0; iter < max_iter_; iter++)
    {
        for (int i = 0; i < num_dimensions_; i++)
        {
            temp_gbest[i] = gbest_[i];
        }

        omega = (omega_init_ - omega_end_) * (float)(max_iter_ - iter) / (float)max_iter_ + omega_end_;

        for (int i = 0; i < num_dimensions_ * num_particles_; i++)
        {
            velocity_[i] = omega * velocity_[i] + c1_ * getRandom() * (pbest_[i] - position_[i]) + c2_ * getRandom() * (gbest_[i % num_dimensions_] - position_[i]);
            if (velocity_[i] > max_velocity_)
                velocity_[i] = max_velocity_;
            position_[i] += velocity_[i];
        }

        for (int i = 0; i < num_dimensions_ * num_particles_; i += num_dimensions_)
        {
            //reduce calling the fitnessFunciton as much as possible
            temp_value = fitnessFunction(position_+i);

            if (temp_value < pbest_value_[i/num_dimensions_])
            {
                for (int j = 0; j < num_dimensions_; j++)
                {
                    pbest_[i + j] = position_[i + j];
                }
                pbest_value_[i/num_dimensions_] = temp_value;
            }
            if (pbest_value_[i/num_dimensions_] < gbest_value)
            {
                for (int j = 0; j < num_dimensions_; j++)
                {
                    gbest_[j] = pbest_[i + j];
                }
                gbest_value = pbest_value_[i/num_dimensions_];
            }
        }
        //std::cout << "gbest:" << fitnessFunction(gbest_) << std::endl;
        if (fabs(fitnessFunction(temp_gbest) - fitnessFunction(gbest_)) <= tolerance_ * fitnessFunction(gbest_))
        {
            iter_counts++;
            if (iter_counts >= max_stall_interations_)
            {
                //std::cout << "iterations end:" << iter << std::endl;
                //std::cout << "gbest:" << fitnessFunction(gbest_) << std::endl;
                break;
            }
        }
        else
        {
            iter_counts = 0;
        }
    }
    gettimeofday(&t2,NULL);
    time_use = (t2.tv_sec-t1.tv_sec)*1000 + (double)(t2.tv_usec-t1.tv_usec)/1000.0;
    std::cout<<"[Time] PSO CPU time: "<<time_use<<"ms"<<std::endl;
    delete temp_gbest;
}

void PSO::printResult()
{
    std::cout << "Optimization Results:" << std::endl;
    for (int i = 0; i < num_dimensions_; i++)
    {
        std::cout << gbest_[i] << std::endl;
    }
}

void PSO::printPbest()
{
    for (int i=0;i<num_particles_;i++)
    {
        for(int j=0;j<num_dimensions_;j++)
        {
            std::cout<<pbest_[i*num_dimensions_+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

void PSO::printPbestValue()
{
    for (int i=0;i<num_particles_;i++)
    {
        std::cout<<pbest_value_[i]<<std::endl;
    }
}