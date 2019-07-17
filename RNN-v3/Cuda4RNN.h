#pragma once
#include <iostream>

#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "MyArray.h"
#include "CudaMap.h"
#include "MyParams.h"

#include "gpu_fns.h"
#include "CudaUtils.h"

using namespace std;
using namespace thrust;

/*
	注意：cuda编译的文件中，不能有Armadillo

	Cuda4RNN中直接使用 gpu_fns提供的矩阵或向量计算或其他辅助函数。
	此文件中用到的fns都在 gpu_fns中找。尤其是kernel函数实现在 gpu_fns.cu

	传入 gpu_fns的参数为：thrust::device_vector, paramStruct

*/

class Cuda4RNN
{
public:
	Cuda4RNN();
	~Cuda4RNN();

	/*
		once Forward and
		once Backward Through Time.
		
		arma::mat in RNN-v2 => MyArray*
	*/
	CudaMap lossFun(MyArray* inputs, 
		float score, 
		MyArray* hprev, 
		device_vector<float>& true_false,
		device_vector<float>& log_target, 
		device_vector<float>& log_prediction);


	MyArray* score2onehot(float score);

	static void test_gpu_fns_CudaUtils();


private:
	// the params shoule be static because of using CPU Parallel Computing
	static int n_features; // num of columns os matData
	static int n_hidden;	// num of hidden neurons
	static int n_output_classes;	// num of predicted classes of Scenarios
	static float score_min;		
	static float score_max;

	static MyArray* Wxh;
	static MyArray* Whh;
	static MyArray* Why;
	static MyArray* bh;
	static MyArray* by;
	
};

