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

#include "Unified.h"

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

/*
	勿忘，在使用时，需要分配此 struct的gpu内存
*/
typedef struct d_params_struct
{
	float loss;
	device_vector<float> dWxh;
	device_vector<float> dWhh;
	device_vector<float> dWhy;
	device_vector<float> dbh;
	device_vector<float> dby;
};

typedef struct sces_struct
{
	device_vector<float> sces_id_score;
	device_vector<float> sces_data;
	device_vector<float> sces_data_mn;
	device_vector<float> sces_data_idx_begin();
};


class Cuda4RNN
{
public:
	Cuda4RNN();
	~Cuda4RNN();

	void initParams();

	/*
		sces_data: data of all scenarios
	*/
	void trainMultiThread(device_vector<float> sces_id_score,
		device_vector<float> sces_data,
		device_vector<int> sces_data_mn,
		device_vector<int> sces_data_idx_begin,
		device_vector<float>& lossAllVec,
		device_vector<float> Wxh,
		device_vector<float> Whh,
		device_vector<float> Why,
		device_vector<float> bh,
		device_vector<float> by,
		float alpha,
		int total_epoches,
		int n_features,
		int n_hidden,
		int n_output_classes,
		float score_min,
		float score_max);

	/**
		once Forward and
		once Backward Through Time.
		
		arma::mat in RNN-v2 => MyArray*, device_vector<float>()

		inputs: data of one scenario
	*/
	void lossFun(cublasHandle_t handle,
		device_vector<float> inputs, int M, int N,
		float score, 
		device_vector<float> hprev, 
		device_vector<float>& true_false,
		device_vector<int>& log_target, 
		device_vector<int>& log_prediction,
		device_vector<float> Wxh,
		device_vector<float> Whh,
		device_vector<float> Why,
		device_vector<float> bh,
		device_vector<float> by,
		float& loss,
		device_vector<float>& dWxh,
		device_vector<float>& dWhh,
		device_vector<float>& dWhy,
		device_vector<float>& dbh,
		device_vector<float>& dby,
		int n_features,
		int n_hidden,
		int n_output_classes,
		float score_min,
		float score_max);

	/*
		idx1_targets: index 1 in targets
	*/
	device_vector<float> score2onehot(float score, int& idx1_targets,
		int n_output_classes,
		float score_min,
		float score_max);

	/*
		P = P - alpha * dP
	*/
	void sgd(device_vector<float>& P, device_vector<float> dP, 
		int size, float alpha);

	void test_gpu_fns_CudaUtils();


public:
	// the params shoule be static because of using CPU Parallel Computing
	//static float alpha;
	//static int total_epoches;
	//static int n_features; // num of columns os matData
	//static int n_hidden;	// num of hidden neurons
	//static int n_output_classes;	// num of predicted classes of Scenarios
	//static float score_min;		
	//static float score_max;

	//device_vector<float>	alpha; // constant mem on GPU
	//device_vector<int>		total_epoches;
	//device_vector<int>		n_features;
	//device_vector<int>		n_hidden;
	//device_vector<int>		n_output_classes;
	//device_vector<float>	score_min;
	//device_vector<float>	score_max;

	/*float alpha;
	int total_epoches;
	int n_features;
	int n_hidden;
	int n_output_classes;
	float score_min;
	float score_max;*/

	/*device_vector<float> Wxh;
	device_vector<float> Whh;
	device_vector<float> Why;
	device_vector<float> bh;
	device_vector<float> by;*/

};

