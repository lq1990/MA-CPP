#pragma once

/*
	functions run on the GPU.

	<important>
	kernel fns are in the *.cu
	</important>

	with：
		cuBLAS for op of vector, matrix 
		curand for generating random values
		thrust

	Tips about thrust:
	vector or matrix is stored in float*,
	when thrust::device_vector<float> d_vec used in proj,
		thrust::raw_pointer_cast(&d_vec[0]) => float*

	inputs of fns: thrust::device_vector<>
	ouput  of fns: thrust::device_vector<>.


	about names of Matrix and vector and scalar,
	Matrix like: A, B, C, ...
	vector like: x, y, z, ...
	scalar like: alpha, beta, ...

	d_A, d_x means: A or x lives in device memory.
*/

#include <iostream>
#include "cublas_v2.h"
#include "curand.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <math.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>


using namespace std;
using namespace thrust;


/*
	为生成随机数
	a~b
*/
struct prg
{
	float a, b;
	unsigned int N;

	__host__ __device__
		prg(float _a = 0.f, float _b = 1.f, unsigned int _N=time(NULL)) : a(_a), b(_b),N(_N) {};

	__host__ __device__
		float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng(N);
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

//struct plus
//{
//	/*! \typedef first_argument_type
//	 *  \brief The type of the function object's first argument.
//	 */
//	typedef T first_argument_type;
//
//	/*! \typedef second_argument_type
//	 *  \brief The type of the function object's second argument.
//	 */
//	typedef T second_argument_type;
//
//	/*! \typedef result_type
//	 *  \brief The type of the function object's result;
//	 */
//	typedef T result_type;
//
//	/*! Function call operator. The return value is <tt>lhs + rhs</tt>.
//	 */
//	__host__ __device__ 
//	T operator() (const T &lhs, const T &rhs) const 
//		{ return lhs + rhs; }
//
//}; // end plus

/*
	return tanh(x)
	
	向量中 each element 各自一个个计算自己的
*/
struct tanh_functor
{
	tanh_functor() {}

	__host__ __device__ 
		float operator()(const float& x) const {
			return tanhf(x);
	}
};

struct exp_functor
{
	exp_functor() {}

	__host__ __device__
		float operator()(const float& x) const {
			return expf(x);
	}
};

struct exp_div_sum_functor
{
	float sum;

	exp_div_sum_functor(float _sum) : sum(_sum) {}

	__host__ __device__
		float operator()(const float& x) const {
			return expf(x) / sum;
	}
};

struct clip_functor
{
	float low, high;

	clip_functor(float _low, float _high) : low(_low), high(_high) {}

	__host__ __device__
		float operator()(const float& x)const {
		if (x > high)
		{
			return high;
		}
		else if (x < low)
		{
			return low;
		}
		else
		{
			return x;
		}
	}
};

/*
	matrix-matrix multiplication.
	C(am,bn) = A(am,an) * B(bm,bn)
	return C
*/
void gpu_mmul(cublasHandle_t handle,
	device_vector<float>& d_A, 
	device_vector<float>& d_B,
	int A_M, int A_N, int B_N, 
	device_vector<float>& dest);


/*
	matrix-vector mulitplication.
	y(k) = A(m,k) * x(k), isMT = false

	y(m) = A'(m,k) * x(m), isMT = true
*/
void gpu_mv(cublasHandle_t handle,
	device_vector<float>& d_A,
	device_vector<float>& d_x,
	int am, int an,
	device_vector<float>& dest,
	bool isMT = false);

/*
	vector-vector or matrix-matrix add.
	z = x + y.

	Tips:
	matrix is dim2, but in mem the matrix is dim1 
		which is arranged in column-major format.

	size: num of elems of x,y,z
*/
device_vector<float> gpu_add(device_vector<float> d_x,
	device_vector<float> d_y, int size);

/*
	vector-vector or matrix-matrix multiplication element-wise.
	z = x .* y
*/
device_vector<float> gpu_mul_elemwise(device_vector<float>& d_x,
	device_vector<float>& d_y,
	int size);

/*
	y = tanh(x)
*/
device_vector<float> gpu_tanh(device_vector<float> d_x, int size);

/*
	y = alpha * x
*/
device_vector<float> gpu_scal(device_vector<float>& d_x, int size, float alpha);


device_vector<float> gpu_get_row(device_vector<float> d_A,
	thrust::device_vector<int> rowLoc,
	int an);

/*
	get col of a matrix.
	d_A: matrix(M,N)

	return device_vector

	若返回一个d_vec，涉及到gpu内存的分配、copy、释放。
	这对gpu是很慢的，gpu是memory bound.

	dest 的起始size要设置足够大
*/
void gpu_get_col(device_vector<float>& d_A, int M, int N, int col, device_vector<float>& dest);

/*
	set "d_A" with "values".
*/
void gpu_set_col(device_vector<float>& d_A, int M, int N, 
	int col,
	device_vector<float> values);

/*
	fill the array A(n_rows, n_cols) with random values (low~high) on GPU.
	when n_rows or n_cols = 1, A is a vector.

	因为是对 d_A 的引用，所以可以直接原地修改
*/
void gpu_fill_rand(device_vector<float>& d_A, 
	int n_rows, int n_cols, 
	float low=0.f, float high=1.f, int seed=0);

device_vector<float> gpu_generate_rand(int n_rows, int n_cols,
	float low = 0.f, float high = 1.f,  int seed = 10);

/*
				exp([1,2,3,...])
	softmax = ---------------------
				sum(exp([...]))
*/
device_vector<float> gpu_softmax(device_vector<float>& d_x, int size);

int gpu_max_index(device_vector<float>& d_x);

float gpu_max_value(device_vector<float>& d_x);

void gpu_clip(device_vector<float>& d_x,
	float lowMargin, float highMargin);

/*
	print the vec or mat on GPU to host.
	A is vector, when n_rows or n_cols = 1.

	the print format is A(n_rows, n_cols)
*/
void printToHost(device_vector<float> d_A, int n_rows, int n_cols, string title);

void printToHost(device_vector<int> d_A, int n_rows, int n_cols, string title);

void printToHost(float* d_A, int n_rows, int n_cols, string title);

void gpu_warmup();

