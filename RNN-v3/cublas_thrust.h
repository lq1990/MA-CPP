#pragma once

#include <iostream>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <time.h>

using namespace std;

/*
	fill the array A(n_rows, n_cols) with random values on GPU
*/
void gpu_fill_rand0(float *A, int n_rows, int n_cols);

/*
	C(m,n) = A(m,k) * B(k,n)
*/
void gpu_blas_mmul(cublasHandle_t handle, const float *A, const float *B, float *C,
	const int m, const int k, const int n);

void print_matrix(const float *A, int n_rows, int n_cols);


void cublas_thrust_main1();

void cublas_thrust_main2();

