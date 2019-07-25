#pragma once

#include <iostream>
#include "cublas_v2.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "math.h"
#include <time.h>
#include "Para.h"

using namespace std;

/**
	gpu_fns 中用到的thrust 改写为 原生level2的 float*

*/


#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/*
	matrix-matrix multiplication.
	C(am,bn) = A(am,an) * B(bm,bn)
	return C
*/
void gpu_mmul(cublasHandle_t handle,
	const float* d_A,
	const float* d_B,
	int A_M, int A_N, int B_N,
	float* dest, cudaStream_t stream);


/*
	matrix-vector mulitplication.
	y(k) = A(m,k) * x(k), isMT = false

	y(m) = A'(m,k) * x(m), isMT = true

	isSync: cudaDeviceSynchronize() or not
*/
void gpu_mv(cublasHandle_t handle,
	const float* d_A,
	const float* d_x,
	int am, int an,
	float* dest,
	bool isMT = false, bool isDevSync = true);

/*
	vector-vector or matrix-matrix add.
	z = x + y.

	Tips:
	matrix is dim2, but in mem the matrix is dim1
		which is arranged in column-major format.

	size: num of elems of x,y,z
*/
void gpu_add(const float* d_x,
	const float* d_y, int size,
	float* dest, cudaStream_t stream);

/*
	vector-vector or matrix-matrix multiplication element-wise.
	z = x .* y
*/
void gpu_mul_elemwise(const float* d_x,
	const float* d_y,
	int size,
	float* dest);

/*
	y = tanh(x)
*/
void gpu_tanh(float* d_x, int size, float* dest);

/**
	tanh(v1 + v2 + v3)
*/
void gpu_tanh_add_add(float* v1, float* v2, float* v3, int size,
	float* dest);

void gpu_tanh_Mv_add_Mv_add_v(cublasHandle_t handle, 
	float* M1, int m1, int n1, float* v1, 
	float* M2, int m2, int n2, float* v2, float* v3, 
	float* dest,
	Para* para,
	cudaStream_t stream[]);

/**
	(1 - hs[t] .* hs[t]) .* dh

	in1: hs
	in2: dh
*/
void gpu_tanh_der_hs_dh(float* d_in1, float* d_in2, int size, float* dest);

/*
	y = alpha * x
*/
void gpu_scal(float* d_x, int size, float alpha, float* dest);

/*
	get col of a matrix.
	d_A: matrix(M,N)

	若返回一个d_vec，涉及到gpu内存的分配、copy、释放。
	这对gpu是很慢的，gpu是memory bound.

	dest 的起始size要设置足够大
*/
void gpu_get_col(float* d_A, 
	int M, int N, int col, float* dest);

/*
	set "d_A" with "values".
*/
void gpu_set_col(float* d_A, int M, int N,
	int col,
	float* values);

void gpu_fill_rand(float* d_A,
	int n_rows, int n_cols,
	float low = 0.f, float high = 1.f, int seed = 0);

/*
	fill d_out with value
*/
void gpu_fill(float* d_out, int size, float value);

/**
	cache[0] = sum(d_in)

	cacheSize >= size && cacheSize=pow(2,n)
*/
void gpu_sum(float* d_in, int size, float* cache);

/*
				exp([1,2,3,...])
	softmax = ---------------------
				sum(exp([...]))

	size of cache >= size of d_in
*/
void gpu_softmax(float * d_in, int size, float * dest, float* cache);

/*
	不能return，
	可从cache的[0]取得结果
*/
void gpu_max_value(float* d_in, int size, float* cache);


/**
	cache[0] saves index
*/
void gpu_max_index(float* d_in, int size, float* cache);

void gpu_clip(float * d_in_out, int size, float lowMargin, float highMargin);

/**
	d_in[beginIdx, endIdx) => copy to d_out[outBegin,)
*/
void gpu_copy(float * d_out, int outBegin,
	float * d_in, int inBegin, int inEnd);

void gpu_clear_arr(float* d_arr, int size);

/*
	print the vec or mat on GPU to host.
	A is vector, when n_rows or n_cols = 1.

	the print format is A(n_rows, n_cols)
*/
void printToHost(float* d_A, int n_rows, int n_cols, string title);

/**
	print last numElems
*/
void printLast(float* d_x, int size, int numElems, string title);

void printToHost(int* d_A, int n_rows, int n_cols, string title);

void gpu_raw_warmup();

/**
	找到 >= value 的最近的一个 以2为底的指数的值
*/
int gpu_find_2exp(int value);

/**
	dy[idx1] -= 1
*/
void gpu_update_dy(float* d_dy, int size, int idx1);

void gpu_fill_onehot(float* onehot, int size, int idx1);

