#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "gpu_raw_fns.h"
#include "Para.h"

using namespace std;


/*
	注意：cuda编译的文件中，不能有Armadillo

	Cuda4RNN中直接使用 gpu_fns提供的矩阵或向量计算或其他辅助函数。
	此文件中用到的fns都在 gpu_fns中找。尤其是kernel函数实现在 gpu_fns.cu

	传入 gpu_fns的参数为：thrust::device_vector, paramStruct

*/



/*
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

typedef struct d_params_struct
{
	float* dWxh;
	float* dWhh;
	float* dWhy;
	float* dbh;
	float* dby;
	float* dhnext;

	float* dy;
	float* dh;
	float* dhraw;
};

typedef struct params_struct
{
	float* Wxh;
	float* Whh;
	float* Why;
	float* bh;
	float* by;
};


// d_vec 在动态分配内存时，出现错误，因此使用原生float*
typedef struct sces_struct
{
	float* sces_id_score;// [id0,score0, id1,score1, id2,score2, ...]
	float* sces_data;// all data
	float* sces_data_mn;// [m0,n0, m1,n1, m2,n2, ...]
	float* sces_data_idx_begin;// [idx0, idx1, idx2, ...]
	float* num_sces; // [0]
	float* total_size; // [0]
};

typedef struct map_state_struct
{
	// => map<int,mat> xs, hs, ys, ps;
	float* xs;
	float* hs; // n_cols of hs is 1 more than others
	float* ys;
	float* ps;
	float* Nmax;
};
*/

void trainMultiThread(
	float* lossAllVec,
	Para* para
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
		Para* para
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
	void sgd(cublasHandle_t handle, Para* para);

	void sgd0(cublasHandle_t handle, float* P, float* dP, int size, float alpha);

	void test_gpu_fns();

	void initPara(Para* para, int Nmax);

	void deInitPara(Para* para);
