#include "pso.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// function used in host could not be translated to the device
// define a same function in device
__device__ __inline__ float deviceFitnessFunction(float *x)
{
	float res = x[0] * pow(2.71828, -(x[0] * x[0] + x[1] * x[1]));
	return res;
}

__global__ void setupCurandInit(curandState *state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ float generateRandom(curandState *global_state, int ind)
{
	curandState localState = global_state[ind];
	float result = curand_uniform(&localState);
	global_state[ind] = localState;
	return result;
}

//initialize the pbest_value
__global__ void initPbestValue(float *pbest_, float *pbest_value_, int num_dimensions_, int num_particles_)
{
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;
	while (thread_id < num_particles_)
	{
		pbest_value_[thread_id] = deviceFitnessFunction(pbest_ + thread_id * num_dimensions_);
		thread_id += stride;
	}
}

__global__ void updatePbest(float *position_, float *pbest_, float *pbest_value_, int num_dimensions_, int num_particles_)
{
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	while (thread_id < num_particles_)
	{
		float temp_value = deviceFitnessFunction(position_ + num_dimensions_ * thread_id);
		if (temp_value < pbest_value_[thread_id])
		{
			for (int i = 0; i < num_dimensions_; i++)
			{
				pbest_[thread_id * num_dimensions_ + i] = position_[thread_id * num_dimensions_ + i];
			}
			pbest_value_[thread_id] = temp_value;
		}
		thread_id += stride;
	}
}

__global__ void updateParticle(float *position_, float *velocity_, float *pbest_,
							   float *gbest_, float omega_init_, float omega_end_, float c1_, float c2_,
							   float max_velocity_, int num_dimensions_, int num_particles_, int iter,
							   int max_iter_, curandState *global_state)
{
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	while (thread_id < num_dimensions_ * num_particles_)
	{
		float omega = (omega_init_ - omega_end_) * (float)(max_iter_ - iter) / (float)max_iter_ + omega_end_;
		velocity_[thread_id] = omega * velocity_[thread_id] + c1_ * generateRandom(global_state, thread_id) * (pbest_[thread_id] - position_[thread_id]) + c2_ * generateRandom(global_state, thread_id + 1) * (gbest_[thread_id % num_dimensions_] - position_[thread_id]);
		if (velocity_[thread_id] > max_velocity_)
			velocity_[thread_id] = max_velocity_;
		position_[thread_id] += velocity_[thread_id];
		thread_id += stride;
	}
}

__device__ __inline__ bool checkMin(float num_a, float num_b)
{
	return num_a < num_b;
}

//function to get gbest
//refer to reduction algorithm
template <unsigned int blockSize>
__device__ void warpMin(volatile float *sdata, int blockdim, unsigned int tid, unsigned int thread_id, int num_particles_)
{
	if (blockSize >= 64 && checkMin(sdata[tid + 32], sdata[tid]) && thread_id + 32 < num_particles_)
	{
		sdata[tid] = sdata[tid + 32];
		sdata[tid + blockdim] = sdata[tid + blockdim + 32];
	}
	if (blockSize >= 32 && checkMin(sdata[tid + 16], sdata[tid]) && thread_id + 16 < num_particles_)
	{
		sdata[tid] = sdata[tid + 16];
		sdata[tid + blockdim] = sdata[tid + blockdim + 16];
	}
	if (blockSize >= 16 && checkMin(sdata[tid + 8], sdata[tid]) && thread_id + 8 < num_particles_)
	{
		sdata[tid] = sdata[tid + 8];
		sdata[tid + blockdim] = sdata[tid + blockdim + 8];
	}
	if (blockSize >= 8 && checkMin(sdata[tid + 4], sdata[tid]) && thread_id + 4 < num_particles_)
	{
		sdata[tid] = sdata[tid + 4];
		sdata[tid + blockdim] = sdata[tid + blockdim + 4];
	}
	if (blockSize >= 4 && checkMin(sdata[tid + 2], sdata[tid]) && thread_id + 2 < num_particles_)
	{
		sdata[tid] = sdata[tid + 2];
		sdata[tid + blockdim] = sdata[tid + blockdim + 2];
	}
	if (blockSize >= 2 && checkMin(sdata[tid + 1], sdata[tid]) && thread_id + 1 < num_particles_)
	{
		sdata[tid] = sdata[tid + 1];
		sdata[tid + blockdim] = sdata[tid + blockdim + 1];
	}
}

template <unsigned int blockSize>
__global__ void minGbest(float *pbest_value_, float *pbest_, float *gbest_, int num_particles_, int num_dimensions_)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	if (thread_id > num_particles_)
		return;
	sdata[tid] = pbest_value_[tid];
	sdata[tid + blockDim.x] = i;
	while (i < num_particles_)
	{
		if (checkMin(pbest_value_[i], sdata[tid]))
		{
			sdata[tid] = pbest_value_[i];
			sdata[tid + blockDim.x] = i;
		}

		if (i + blockSize > num_particles_)
			break;

		if (checkMin(pbest_value_[i + blockSize], sdata[tid]))
		{
			sdata[tid] = pbest_value_[i + blockSize];
			sdata[tid + blockDim.x] = i + blockSize;
		}
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024)
	{
		if (tid < 512 && checkMin(sdata[tid + 512], sdata[tid]) && thread_id + 512 < num_particles_)
		{
			sdata[tid] = sdata[tid + 512];
			sdata[tid + blockDim.x] = sdata[tid + blockDim.x + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (tid < 256 && checkMin(sdata[tid + 256], sdata[tid]) && thread_id + 256 < num_particles_)
		{
			sdata[tid] = sdata[tid + 256];
			sdata[tid + blockDim.x] = sdata[tid + blockDim.x + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128 && checkMin(sdata[tid + 128], sdata[tid]) && thread_id + 128 < num_particles_)
		{
			sdata[tid] = sdata[tid + 128];
			sdata[tid + blockDim.x] = sdata[tid + blockDim.x + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64 && checkMin(sdata[tid + 64], sdata[tid]) && thread_id + 64 < num_particles_)
		{
			sdata[tid] = sdata[tid + 64];
			sdata[tid + blockDim.x] = sdata[tid + blockDim.x + 64];
		}
		__syncthreads();
	}
	if (tid < 32)
		warpMin<blockSize>(sdata, blockDim.x, tid, thread_id, num_particles_);
	if (tid == 0)
	{
		for (int i = 0; i < num_dimensions_; i++)
			gbest_[i] = pbest_[i + (unsigned int)(sdata[blockDim.x]) * num_dimensions_];
	}
}

bool PSO::initCuda()
{
	int device_count = 0;
	// Check if CUDA runtime calls work at all
	cudaError t = cudaGetDeviceCount(&device_count);
	if (t != cudaSuccess)
	{
		std::cout << "[CUDA] First call to CUDA Runtime API failed. Are the drivers installed?" << std::endl;
		return false;
	}

	// Is there a CUDA device at all?
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	if (device_count < 1)
	{
		std::cout << "[CUDA] No CUDA devices found. " << std::endl;
		std::cout << "[CUDA] Make sure CUDA device is powered, connected and available. " << std::endl;
		std::cout << "[CUDA] On laptops: disable powersave/battery mode. " << std::endl;
		std::cout << "[CUDA] Exiting... " << std::endl;
		return false;
	}

	std::cout << "[CUDA] CUDA device(s) found, picking best one " << std::endl;
	std::cout << "[CUDA] " << std::endl;
	// We have at least 1 CUDA device, so now select the fastest (method from Nvidia helper library)
	int device = findCudaDevice(0, 0);

	// Print available device memory
	cudaDeviceProp properties;
	checkCudaErrors(cudaGetDeviceProperties(&properties, device));
	std::cout << "[CUDA] Best device: " << properties.name << std::endl;
	std::cout << "[CUDA] Available global device memory: " << (double)properties.totalGlobalMem / 1024 / 1024 << " MB. " << std::endl;

	// Check compute capability
	if (properties.major < 2)
	{
		std::cout << "[CUDA] Your cuda device has compute capability " << properties.major << properties.minor << ". We need at least 2.0 for atomic operations. Exiting. " << std::endl;
		return false;
	}

	return true;
}

void PSO::getResultCUDA()
{
	if (!initCuda())
		return;
	init();

	cudaEvent_t start_vox, stop_vox;
	curandState *dev_state;
	checkCudaErrors(cudaEventCreate(&start_vox));
	checkCudaErrors(cudaEventCreate(&stop_vox));
	float elapsedTime;

	float *dev_position;
	float *dev_velocity;
	float *dev_pbest;
	float *dev_gbest;
	float *dev_pbest_value;

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int block_size_particle;
	int block_size_pbest;
	int min_grid_size_particle;
	int min_grid_size_pbest;
	int grid_size_particle;
	int grid_size_pbest;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size_particle, &block_size_particle, updateParticle, 0, num_dimensions_ * num_particles_);
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size_pbest, &block_size_pbest, updatePbest, 0, num_particles_);

	grid_size_particle = (num_particles_ * num_dimensions_ + block_size_particle - 1) / block_size_particle;
	grid_size_pbest = (num_particles_ + block_size_pbest - 1) / block_size_pbest;
	int sdatasize = block_size_pbest * sizeof(float);

	//printPbest();
	//printPbestValue();
	//std::cout<<gbest_[0]<<" "<<gbest_[1]<<std::endl;
	//std::cout<<num_particles_<<" "<<num_dimensions_<<std::endl;
	//std::cout<<block_size_particle<<" "<<block_size_pbest<<" "<<min_grid_size_particle<<" "<<min_grid_size_pbest<<" "<<grid_size_particle<<" "<<grid_size_pbest<<std::endl;

	checkCudaErrors(cudaMalloc((void **)&dev_position, sizeof(float) * num_dimensions_ * num_particles_));
	checkCudaErrors(cudaMalloc((void **)&dev_velocity, sizeof(float) * num_dimensions_ * num_particles_));
	checkCudaErrors(cudaMalloc((void **)&dev_pbest, sizeof(float) * num_dimensions_ * num_particles_));
	checkCudaErrors(cudaMalloc((void **)&dev_gbest, sizeof(float) * num_dimensions_));
	checkCudaErrors(cudaMalloc((void **)&dev_pbest_value, sizeof(float) * num_particles_));
	checkCudaErrors(cudaMalloc(&dev_state, sizeof(float) * num_particles_ * num_dimensions_));

	checkCudaErrors(cudaMemcpy(dev_position, position_, sizeof(float) * num_particles_ * num_dimensions_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_velocity, velocity_, sizeof(float) * num_dimensions_ * num_particles_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_pbest, pbest_, sizeof(float) * num_dimensions_ * num_particles_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_gbest, gbest_, sizeof(float) * num_dimensions_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_pbest_value, pbest_value_, sizeof(float) * num_particles_, cudaMemcpyHostToDevice));

	//Optimization
	checkCudaErrors(cudaEventRecord(start_vox, 0));
	initPbestValue<<<grid_size_pbest, block_size_pbest>>>(dev_pbest, dev_pbest_value, num_dimensions_, num_particles_);
	cudaDeviceSynchronize();
	setupCurandInit<<<grid_size_particle, block_size_particle>>>(dev_state, unsigned(time(NULL)));
	for (int i = 0; i < max_iter_; i++)
	{
		updateParticle<<<grid_size_particle, block_size_particle>>>(dev_position, dev_velocity, dev_pbest, dev_gbest, omega_init_, omega_end_, c1_, c2_, max_velocity_, num_dimensions_, num_particles_, 0, max_iter_, dev_state);
		cudaDeviceSynchronize();
		updatePbest<<<grid_size_pbest, block_size_pbest>>>(dev_position, dev_pbest, dev_pbest_value, num_dimensions_, num_particles_);
		cudaDeviceSynchronize();
		switch (block_size_pbest)
		{
		case 1024:
			minGbest<1024><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 512:
			minGbest<512><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 256:
			minGbest<256><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 128:
			minGbest<128><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 64:
			minGbest<64><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 32:
			minGbest<32><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 16:
			minGbest<16><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 8:
			minGbest<8><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 4:
			minGbest<4><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 2:
			minGbest<2><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		case 1:
			minGbest<1><<<grid_size_pbest, block_size_pbest, sdatasize * 2>>>(dev_pbest_value, dev_pbest, dev_gbest, num_particles_, num_dimensions_);
			break;
		}
		cudaDeviceSynchronize();
	}

	checkCudaErrors(cudaEventRecord(stop_vox, 0));
	checkCudaErrors(cudaEventSynchronize(stop_vox));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	std::cout << "[Time] PSO GPU time: " << elapsedTime << "ms" << std::endl;

	checkCudaErrors(cudaMemcpy(position_, dev_position, sizeof(float) * num_particles_ * num_dimensions_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(velocity_, dev_velocity, sizeof(float) * num_particles_ * num_dimensions_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pbest_, dev_pbest, sizeof(float) * num_particles_ * num_dimensions_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gbest_, dev_gbest, sizeof(float) * num_dimensions_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pbest_value_, dev_pbest_value, sizeof(float) * num_particles_, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(dev_position));
	checkCudaErrors(cudaFree(dev_velocity));
	checkCudaErrors(cudaFree(dev_pbest));
	checkCudaErrors(cudaFree(dev_gbest));
	checkCudaErrors(cudaFree(dev_pbest_value));

	// Destroy timers
	checkCudaErrors(cudaEventDestroy(start_vox));
	checkCudaErrors(cudaEventDestroy(stop_vox));

	//printPbest();
	//printPbestValue();
	//std::cout<<gbest_[0]<<" "<<gbest_[1]<<std::endl;
}
