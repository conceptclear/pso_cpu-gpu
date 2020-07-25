#pragma once

#include <math.h>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curand_kernel.h"

extern __host__ __device__ float fitnessFunction(float* x);