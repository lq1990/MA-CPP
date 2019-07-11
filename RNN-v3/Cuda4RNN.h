#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

using namespace std;

class Cuda4RNN
{
public:
	Cuda4RNN();
	~Cuda4RNN();

	static void showCudaInfo();
	static void cublasDemo();
};

