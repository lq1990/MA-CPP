#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "gpu_raw_fns.h"

using namespace std;

/*
	注意：cuda编译的文件中，不能有Armadillo

	Cuda4RNN中直接使用 gpu_fns提供的矩阵或向量计算或其他辅助函数。
	此文件中用到的fns都在 gpu_fns中找。尤其是kernel函数实现在 gpu_fns.cu

	传入 gpu_fns的参数为：thrust::device_vector, paramStruct

*/

typedef struct rnn_params_struct
{
	int total_epoches;
	int n_features;
	int n_hidden;
	int n_output_classes;
	float alpha;
	float score_min;
	float score_max;
};

/*
	勿忘，在使用时，需要分配此 struct的gpu内存
*/
typedef struct d_params_struct
{
	float* dWxh;
	float* dWhh;
	float* dWhy;
	float* dbh;
	float* dby;
};

typedef struct params_struct
{
	float* Wxh;
	float* Whh;
	float* Why;
	float* bh;
	float* by;
};

/**
	d_vec 在动态分配内存时，出现错误，因此使用原生float*

*/
typedef struct sces_struct
{
	float* sces_id_score;// [id0,score0, id1,score1, id2,score2, ...]
	float* sces_data;// all data
	int* sces_data_mn;// [m0,n0, m1,n1, m2,n2, ...]
	int* sces_data_idx_begin;// [idx0, idx1, idx2, ...]
};

typedef struct cache_struct
{
	float* tmp_d_vec; // for get/set col
	float* W_tmp1;
	float* W_tmp2;
	float* W_tmp3;
};


void trainMultiThread(
	float* lossAllVec,
	sces_struct* sces_s,
	params_struct* p_s,
	rnn_params_struct* rnn_p_s,
	cache_struct* cache_s
);

	/**
		once Forward and
		once Backward Through Time.
		
		arma::mat in RNN-v2 => MyArray*, device_vector<float>()

		inputs: data of one scenario
	*/
	void lossFun(
		cublasHandle_t handle,
		float* inputs, int M, int N,
		float score,
		float* hprev,
		float* true_false,
		float& loss,
		params_struct* p_s,
		d_params_struct* d_p_s,
		rnn_params_struct* rnn_p,
		cache_struct* cache_s
	);

	/*
		idx1_targets: index 1 in targets
	*/
float* score2onehot(float score, int& idx1_targets,
		int n_output_classes,
		float score_min,
		float score_max);

	/*
		P = P - alpha * dP
	*/
	void sgd(float* P, float* dP, 
		int size, float alpha);

	void test_gpu_fns();

