#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "MyStruct.h"
#include "CudaMap.h"

using namespace std;
using namespace thrust;

class Cuda4RNN
{
public:
	Cuda4RNN();
	~Cuda4RNN();

	/*
		once Forward and
		once Backward Through Time
	*/
	CudaMap lossFun(MyArray inputs, 
		float score, 
		MyArray hprev, 
		device_vector<float>& true_false,
		device_vector<float>& log_target, 
		device_vector<float>& log_prediction);




};

