#pragma once
#include <iostream>
#include <string>
#include <sstream>

#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "gpu_fns.h"

using namespace std;
using namespace thrust;

/*
	set params of Model
*/

class MyParams
{
public:
	MyParams();
	~MyParams();

	static float alpha;
	static int total_epoches;
	static int n_features; // num of columns os matData
	static int n_hidden;	// num of hidden neurons
	static int n_output_classes;	// num of predicted classes of Scenarios
	static float score_min;
	static float score_max;

};

