# pso_cpu-gpu
An implementation of pso on cpu and gpu                   

# Usage
Listing parameters in PSO must be set before this program:
- maximal iterations
- dimension of the optimization problem
- number of particles
- operation platform (CUDA or CPU)                  
parameters listed below can be changed:
- boundary
- c1,c2
- max_velocity
- omega
- max_stall_interations
- tolerance

# Fitness function
Fitness funtion can be changed in main.cpp for cpu. In fact, function conducted in cpu could not be translated to cuda. If you want to use a new fitness function in cuda, fitness function defined in pso.cu should be changed.                  
'max_stall_interations_' is not suitable for cuda because calculating the number of iterations requires memory interaction which decreases the efficiency.

# Example
Example fitness function is same as the example used in matlab.                      

for cpu with 10000 iterations and 1000 particles:
```
[Time] PSO CPU time: 1026.39ms
Optimization Results:
-0.707079
3.27442e-05
```
for cuda with 10000 iterations and 1000 particles:
```
[Time] PSO GPU time: 415.441ms
Optimization Results:
-0.70711
0.000184042
```