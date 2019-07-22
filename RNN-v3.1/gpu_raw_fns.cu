#include "gpu_raw_fns.h"

/**
	raw means: use float* instead of thrust
	some functions needs kernels that are writen by myself.
	
*/

__global__ void rand_kernel(float* d_out, int size, 
	float low=-1.f, float high=1.f, int seed = 1)
{
	int tid = get_tid();

	if (tid < size)
	{
		curandState state;
		curand_init(seed, tid, 0, &state);
		float val = curand_normal(&state); // mean=0,sigma=1 正态分布

		float val1 = (high - low) * (val+1.f)/2.f + low;
		d_out[tid] = val1;
	}
}

__global__ void fill_kernel(float* d_out, int size, float value)
{
	int tid = get_tid();
	if (tid < size)
	{
		d_out[tid] = value;
	}
}

__global__ void copy_kernel(float* d_out, float* d_in, int beginIdx, int endIdx)
{
	int tid = get_tid();

	if (tid < (endIdx - beginIdx))
	{
		float val = d_in[tid + beginIdx];
		d_out[tid] = val;
	}

}

__global__ void clear_kernel(float* d_arr, int size)
{
	int tid = get_tid();

	if (tid < size)
	{
		d_arr[tid] = 0.f;
	}
}

void gpu_fill_rand(float * d_A, 
	int n_rows, int n_cols, 
	float low, float high, int seed)
{
	// gridsize, blocksize
	int bs = 256;
	//int s = ceil(sqrt(n_rows*n_cols + bs - 1.f) / bs); // 错了。一定要注意bs gs的大小
	int s = ceil(sqrt((n_rows*n_cols + bs - 1.) / bs));
	dim3 gs = dim3(s, s);

	// launch kernel
	rand_kernel <<<gs, bs >>> (d_A, n_rows*n_cols, low, high, seed);
	
	cudaDeviceSynchronize();
}

void gpu_fill(float * d_out, int size, float value)
{
	// gridsize, blocksize
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.) / bs));
	dim3 gs = dim3(s, s);
	fill_kernel << <gs, bs >> > (d_out, size, value);

	cudaDeviceSynchronize();
}

void gpu_copy(float * d_out, float * d_in, int beginIdx, int endIdx)
{
	// gridsize, blocksize
	int bs = 256;

	int s = ceil(sqrt(((endIdx-beginIdx) + bs - 1.) / bs));
	dim3 gs = dim3(s, s);
	copy_kernel << <gs, bs >> > (d_out, d_in, beginIdx, endIdx);

	cudaDeviceSynchronize();
}

void gpu_clear_arr(float * d_arr, int size)
{
	// bs, gs
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f)/bs));
	dim3 gs = dim3(s, s);

	// launch kernel
	clear_kernel <<< gs, bs >>> (d_arr, size);

	cudaDeviceSynchronize();
}

void printToHost(float * d_A, int n_rows, int n_cols, string title)
{
	/*float* h_A;
	h_A = (float*)malloc(n_rows*n_cols * sizeof(float));
	cudaMemcpy(h_A, d_A, 
		n_rows*n_cols * sizeof(float), cudaMemcpyDeviceToHost);*/

	cout << title << endl;
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << d_A[j*n_rows + i] << "  ";
		}
		cout << endl;
	}

	cout << endl;
	//free(h_A);
}

void printToHost(int * d_A, int n_rows, int n_cols, string title)
{
	cout << title << endl;
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			cout << d_A[j*n_rows + i] << "  ";
		}
		cout << endl;
	}

	cout << endl;
}


void gpu_raw_warmup()
{
	float* d_in;
	cudaMallocManaged((void**)&d_in, sizeof(float));
	d_in[0] = 1.f;

	cudaFree(d_in);
}


