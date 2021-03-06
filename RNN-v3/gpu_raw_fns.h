﻿#pragma once

/**
	gpu_fns 中用到的thrust 改写为 原生level2的 float*
	
*/

#include <iostream>
#include "cublas_v2.h"
//#include "curand.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <math.h>
#include <time.h>

using namespace std;

/*
	matrix-matrix multiplication.
	C(am,bn) = A(am,an) * B(bm,bn)
	return C
*/
void gpu_mmul(cublasHandle_t handle,
	const float* d_A,
	const float* d_B,
	int A_M, int A_N, int B_N,
	float* dest);


/*
	matrix-vector mulitplication.
	y(k) = A(m,k) * x(k), isMT = false

	y(m) = A'(m,k) * x(m), isMT = true
*/
void gpu_mv(cublasHandle_t handle,
	const float* d_A,
	const float* d_x,
	int am, int an,
	float* dest,
	bool isMT = false);

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
	float* dest);

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

/*
	y = alpha * x
*/
void gpu_scal(float* d_x, int size, float alpha, float* dest);


void gpu_get_row(const float* d_A,
	float* rowLoc,
	int an,
	float* dest);

/*
	get col of a matrix.
	d_A: matrix(M,N)

	return device_vector

	若返回一个d_vec，涉及到gpu内存的分配、copy、释放。
	这对gpu是很慢的，gpu是memory bound.

	dest 的起始size要设置足够大
*/
void gpu_get_col(float* d_A, 
	int M, int N, int col, 
	float* dest);

/*
	set "d_A" with "values".
*/
void gpu_set_col(float* d_A, int M, int N,
	int col,
	float* values);

/*
	fill the array A(n_rows, n_cols) with random values (low~high) on GPU.
	when n_rows or n_cols = 1, A is a vector.

	因为是对 d_A 的引用，所以可以直接原地修改
*/
void gpu_fill_rand(float* d_A,
	int n_rows, int n_cols,
	float low = 0.f, float high = 1.f, int seed = 0);

void gpu_generate_rand(float* dest,int n_rows, int n_cols,
	float low = 0.f, float high = 1.f, int seed = 10);

/*
				exp([1,2,3,...])
	softmax = ---------------------
				sum(exp([...]))
*/
void gpu_softmax(float* d_x, int size, float* dest);

int gpu_max_index(float* d_x, int size);

float gpu_max_value(float* d_x, int size);

void gpu_clip(float* d_x,
	float lowMargin, float highMargin);

/*
	print the vec or mat on GPU to host.
	A is vector, when n_rows or n_cols = 1.

	the print format is A(n_rows, n_cols)
*/
void printToHost(float* d_A, int n_rows, int n_cols, string title);

void gpu_warmup();
