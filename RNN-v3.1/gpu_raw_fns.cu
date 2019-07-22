﻿#include "gpu_raw_fns.h"

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

/**
	d_out[outBegin, outEnd) <=copy d_in[inBegin, inEnd)
*/
__global__ void copy_kernel(float* d_out, int outBegin,
	float* d_in, int inBegin, int inEnd)
{
	int tid = get_tid();

	if (tid < (inEnd - inBegin))
	{
		float val = d_in[tid + inBegin];
		d_out[tid + outBegin] = val;
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

__global__ void add_kernel(const float * d_x,
	const float * d_y, int size,
	float * dest) 
{
	int tid = get_tid();
	if (tid < size)
	{
		dest[tid] = d_x[tid] + d_y[tid];
	}
}

__global__ void mul_elemwise_kernel(const float * d_x,
	const float * d_y, int size,
	float * dest)
{
	int tid = get_tid();
	if (tid < size)
	{
		dest[tid] = d_x[tid] * d_y[tid];
	}
}

__global__ void tanh_kernel(float * d_x, int size, float * dest)
{
	int tid = get_tid();
	if (tid < size)
	{
		dest[tid] = tanhf(d_x[tid]);
	}
}

__global__ void tanh_add_add_kernel(float * v1, float * v2, float * v3, 
	int size,
	float * dest)
{
	int tid = get_tid();
	if (tid < size)
	{
		// v1 + v2 + v3
		float s123 = v1[tid] + v2[tid] + v3[tid];
		// tanh
		dest[tid] = tanhf(s123);
	}
}

__global__ void scal_kernel(float * d_x, int size,
	float alpha, float * dest)
{
	int tid = get_tid();
	if (tid < size)
	{
		dest[tid] = alpha * d_x[tid];
	}
}

__global__ void sum_kernel(float * d_in, int size, float* cache)
{
	extern __shared__ float sdata[];

	int myId = get_tid();
	int tid = threadIdx.x; 
	// 同一个线程，在block中有位置tid，grid全局也有位置myId

	// load data from global mem
	sdata[tid] = d_in[myId];
	__syncthreads();

	// reduce
	for (int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		cache[blockIdx.x] = sdata[0];
	}
	
}


/**
	get col data of d_A(M,N),
	// dest需要提前分配好内存
*/
void gpu_get_col(float * d_A, int M, int N, int col, float* dest)
{
	int begin = col * M;
	int end = (col + 1)*M;
	gpu_copy(dest, 0, d_A, begin, end);
}

void gpu_set_col(float * d_A, int M, int N, int col, float * values)
{
	int setBegin = col * M;
	int setEnd = (col + 1)*M;

	gpu_copy(d_A, setBegin, values, 0, M);
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
	int s = ceil(sqrt( (size + bs - 1.) / bs ));
	dim3 gs = dim3(s, s);
	fill_kernel << <gs, bs >> > (d_out, size, value);

	cudaDeviceSynchronize();
}

float gpu_sum(float* d_in, int size, float* cache)
{
	// 需要确保 d_in 的size <= 512 ^ 2

	// gridsize, blocksize
	int bs = 512;
	int gs = (size + bs - 1) / bs;
	sum_kernel << <gs, bs, bs*sizeof(float) >> > (d_in, size, cache);
	cudaDeviceSynchronize();

	// 对cache中数据继续sum。cache中存储了上一步blocks中sum的结果
	if (gs == 1)
	{
		return cache[0];
	}
	else
	{
		sum_kernel << <1, bs, bs * sizeof(float) >> > (cache, gs, cache);
		cudaDeviceSynchronize(); // kernel之后一定记得加同步

		return cache[0];
	}
	
}

void gpu_softmax(float * d_in, int size, float * dest, float* cache)
{
	// softmax = exp(x) / sum( (exp(x)) )
	// gridsize, blocksize
	int bs = 256;
	int s = ceil(sqrt( (size + bs - 1.) / bs ));
	dim3 gs = dim3(s, s);

	// step 1. exp(x)

	cudaDeviceSynchronize();
	// step 2. sum(exp(x))

	cudaDeviceSynchronize();
	// step 3. exp/sum(exp)


	cudaDeviceSynchronize();
}

void gpu_copy(float * d_out, int outBegin, 
	float * d_in, int inBegin, int inEnd)
{
	// gridsize, blocksize
	int bs = 256;

	int s = ceil(sqrt(((inEnd - inBegin) + bs - 1.) / bs));
	dim3 gs = dim3(s, s);
	copy_kernel << <gs, bs >> > (d_out, outBegin, d_in, inBegin, inEnd);

	cudaDeviceSynchronize();
}

/**
	set arr to 0
*/
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

void printLast(float * d_x, int size, int numElems, string title)
{
	cout << " --------- " << title << "---------" << endl;
	int begin = size - numElems;
	int end = size - 1;
	for (int i = begin; i <= end; i++)
	{
		cout << d_x[i] << "  ";
	}
	cout << endl;
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

int gpu_find_2exp(int value)
{
	int a = ceil(log2(value));
	return pow(2, a);
}

void gpu_mmul(cublasHandle_t handle, 
	const float * d_A, 
	const float * d_B, 
	int A_M, int A_N, 
	int B_N, float * dest)
{
	// dest = alpha * A * B + beta * dest
	float alpha = 1.f;
	float beta = 0.f;
	cublasSgemm_v2(handle, 
		CUBLAS_OP_N, CUBLAS_OP_N, 
		A_M, B_N, A_N, 
		&alpha, 
		d_A, A_M, 
		d_B, A_N, 
		&beta, 
		dest, A_M);

	cudaDeviceSynchronize();
}

void gpu_mv(cublasHandle_t handle, 
	const float * d_A, 
	const float * d_x, 
	int am, int an, float * dest, bool isMT)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	if (isMT)
	{
		// y = A' * x
		cublasSgemv(handle, // y = alpha * A*x + beta*y
			CUBLAS_OP_T,
			am, an,
			&alpha,
			d_A, am,
			d_x, 1,
			&beta,
			dest, 1);
	}
	else
	{
		// y = A * x
		cublasSgemv(handle, // y = alpha * A*x + beta*y
			CUBLAS_OP_N,
			am, an,
			&alpha,
			d_A, am,
			d_x, 1,
			&beta,
			dest, 1);
	}

	cudaDeviceSynchronize();
}

void gpu_add(const float * d_x, const float * d_y, int size, 
	float * dest)
{
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f) / bs));
	dim3 gs = dim3(s, s);

	add_kernel << <gs, bs >> > (d_x, d_y, size, dest);

	cudaDeviceSynchronize();
}

void gpu_mul_elemwise(const float * d_x, const float * d_y, 
	int size, float * dest)
{
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f) / bs));
	dim3 gs = dim3(s, s);

	mul_elemwise_kernel << <gs, bs >> > (d_x, d_y, size, dest);

	cudaDeviceSynchronize();
}

void gpu_tanh(float * d_x, int size, float * dest)
{
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f) / bs));
	dim3 gs = dim3(s, s);

	tanh_kernel << <gs, bs >> > (d_x, size, dest);

	cudaDeviceSynchronize();
}

void gpu_tanh_add_add(float * v1, float * v2, float * v3, int size, 
	float * dest)
{
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f) / bs));
	dim3 gs = dim3(s, s);

	tanh_add_add_kernel << <gs, bs >> > (v1, v2, v3, size, dest);

	cudaDeviceSynchronize();
}

void gpu_scal(float * d_x, int size, float alpha, float * dest)
{
	int bs = 256;
	int s = ceil(sqrt((size + bs - 1.f) / bs));
	dim3 gs = dim3(s, s);

	scal_kernel << <gs, bs >> > (d_x, size, alpha, dest);

	cudaDeviceSynchronize();
}
