#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

class CudaScript
{
public:
	CudaScript();
	~CudaScript();

	static void showCudaInfo();

	static void cublasDemo();
};

