#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include "gpu_raw_fns.h"
#include "MyClock.h"

using namespace std;

// 1D gs, 1D bs
#define get_tid() (blockDim.x * blockIdx.x + threadIdx.x)

/**
	basic use
*/
void stream01();


/**
	set stream to be used by each cuBLAS routine
*/
void stream_cublas01();

/**
	compare
*/
void stream_cublas02();



void print(float* h, int n_rows, int n_cols, string title);
