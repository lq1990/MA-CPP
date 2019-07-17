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
	ouput  of fns: thrust::device_vector<>
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
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
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

	__host__ __device__
		prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

	__host__ __device__
		float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

struct tanh_functor
{
	const float a;

	tanh_functor(float _a) : a(_a){}


};

/*
	matrix-matrix multiplication.
	C(am,bn) = A(am,an) * B(bm,bn)
	return C
*/
device_vector<float> gpu_mmul(cublasHandle_t handle,
	device_vector<float> d_A, 
	device_vector<float> d_B,
	int A_M, int A_N, int B_N);


/*
	matrix-vector mulitplication.
	y(k) = A(m,k) * x(k), isAT = false

	y(m) = A'(m,k) * x(m), isAT = true
*/
device_vector<float> gpu_mv(cublasHandle_t handle,
	device_vector<float> d_A,
	device_vector<float> d_x,
	int am, int an, bool isAT = false);

/*
	vector-vector or matrix-matrix add.
	c = a + b.

	Tips:
	matrix is dim2, but in mem the matrix is dim1 
		which is arranged in column-major format.

	size: num of elems of a,b,c
*/
device_vector<float> gpu_add(cublasHandle_t handle, 
	device_vector<float> d_a,
	device_vector<float> d_b, int size);

/*
	vector-vector or matrix-matrix multiplication element-wise.
	c = a .* b
*/
void gpu_mul_elementwise(const float* d_a, const float* d_b,
	float* d_c, int size);

/*
	y = tanh(x)
*/
device_vector<float> gpu_tanh(device_vector<float> d_x, int size);


/*
	fill the array A(n_rows, n_cols) with random values (low~high) on GPU.
	when n_rows or n_cols = 1, A is a vector.
*/
device_vector<float> gpu_fill_rand(device_vector<float> d_A, 
	int n_rows, int n_cols, 
	float low=0.f, float high=1.f);

/*
	print the vec or mat on GPU to host.
	A is vector, when n_rows or n_cols = 1.

	the print format is A(n_rows, n_cols)
*/
void printToHost(device_vector<float> d_A, int n_rows, int n_cols, string title);

void printToHost(float* d_A, int n_rows, int n_cols, string title);

