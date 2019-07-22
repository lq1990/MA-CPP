#include "gpu_raw_fns.h"

/**
	raw means: use float* instead of thrust
	some functions needs kernels that are writen by myself.
	
*/

__global__ void rand_kernel(float* d_out, int size, 
	float low=0.f, float high=1.f, int seed=0)
{
	curandState state;
	// 1D gridsize, 1D blocksize
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < size)
	{
		curand_init(seed, tid, 0, &state);
		float val = curand_uniform(&state);
		d_out[tid] = val;
	}
}

void gpu_fill_rand(float * d_A, 
	int n_rows, int n_cols, 
	float low, float high, int seed)
{
	// gridsize, blocksize
	int bs = 512;
	int gs = ceil( n_rows * n_cols / bs);


	// launch kernel
	rand_kernel << <gs, bs >> > (d_A, n_rows*n_cols);

}


