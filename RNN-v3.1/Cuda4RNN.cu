#include "Cuda4RNN.h"

void trainMultiThread(
	float* lossAllVec,
	sces_struct* sces_s,
	params_struct* p_s,
	rnn_params_struct* rnn_p_s,
	cache_struct* cache_s
)
{
	cout << "train begins" << endl << "alpha: " << rnn_p_s->alpha << endl;
	cout << "total_epoches: " << rnn_p_s->total_epoches << endl << "n_features: " << rnn_p_s->n_features << endl;
	cout << "n_hidden: " << rnn_p_s->n_hidden << endl << "n_output_classes: " << rnn_p_s->n_output_classes << endl;
	cout << "score_min: " << rnn_p_s->score_min << endl << "score_max: " << rnn_p_s->score_max << endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	// 先用 标准SGD优化，使用1个cpu线程
	float* loss_one_epoch;
	float* loss_mean_each_epoch;
	float* true_false;
	float* accuracy_each_epoch;
	float loss = 0.f;
	cudaMallocManaged((void**)&loss_one_epoch, rnn_p_s->total_epoches * sizeof(float));
	cudaMallocManaged((void**)&loss_mean_each_epoch, rnn_p_s->total_epoches * sizeof(float));
	cudaMallocManaged((void**)&true_false, rnn_p_s->total_epoches * sizeof(float));
	cudaMallocManaged((void**)&accuracy_each_epoch, rnn_p_s->total_epoches * sizeof(float));

	// 先取出一个 场景数据 训练 RNN
	float id0 = sces_s->sces_id_score[0];
	float score0 = sces_s->sces_id_score[1];
	int sce0_M = sces_s->sces_data_mn[0];
	int sce0_N = sces_s->sces_data_mn[1];

	float* sce0_data;
	sce0_data = (float*)cudaMallocManaged((void**)&sce0_data, sce0_M*sce0_N*sizeof(float));
	int beginIdx = sces_s->sces_data_idx_begin[0];
	int endIdx = sces_s->sces_data_idx_begin[1];
	gpu_copy(sce0_data, sces_s->sces_data, beginIdx, endIdx);
	 
	float* hprev;
	cudaMallocManaged((void**)&hprev, rnn_p_s->n_hidden * sizeof(float));

	d_params_struct* d_p_s;
	cudaMalloc((void**)&d_p_s, sizeof(d_params_struct));
	cudaMallocManaged((void**)&d_p_s->dWxh, rnn_p_s->n_hidden*rnn_p_s->n_features * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dWhh, rnn_p_s->n_hidden*rnn_p_s->n_hidden * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dWhy, rnn_p_s->n_hidden*rnn_p_s->n_output_classes * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dbh, rnn_p_s->n_hidden * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dby, rnn_p_s->n_output_classes * sizeof(float));
	// assign values 0.f

	for (int i = 0; i < rnn_p_s->total_epoches; i++)
	{
		//true_false.clear();
		gpu_clear_arr(true_false, rnn_p_s->total_epoches);
		//loss_one_epoch.clear();
		gpu_clear_arr(loss_one_epoch, rnn_p_s->total_epoches);

		/*thrust::fill(d_p_s->dWxh.begin(), d_p_s->dWxh.end(), 0.f);
		thrust::fill(d_p_s->dWhh.begin(), d_p_s->dWhh.end(), 0.f);
		thrust::fill(d_p_s->dWhy.begin(), d_p_s->dWhy.end(), 0.f);
		thrust::fill(d_p_s->dbh.begin(),  d_p_s->dbh.end(), 0.f);
		thrust::fill(d_p_s->dby.begin(),  d_p_s->dby.end(), 0.f);*/
		gpu_fill(d_p_s->dWxh, rnn_p_s->n_hidden * rnn_p_s->n_features, 0.f);
		gpu_fill(d_p_s->dWhh, rnn_p_s->n_hidden * rnn_p_s->n_hidden, 0.f);
		gpu_fill(d_p_s->dWhy, rnn_p_s->n_hidden * rnn_p_s->n_output_classes, 0.f);
		gpu_fill(d_p_s->dbh,  rnn_p_s->n_hidden, 0.f);
		gpu_fill(d_p_s->dby,  rnn_p_s->n_output_classes, 0.f);

		//thrust::fill(hprev.begin(), hprev.end(), 0.f);
		gpu_fill(hprev, rnn_p_s->n_hidden, 0.f);
		

		cout << "epoch: " << i << endl;

	}

	// lossVec mean, accu

	// free resource
	cudaFree(hprev);
	cudaFree(sce0_data);
	cudaFree(loss_one_epoch);
	cudaFree(loss_mean_each_epoch);
	cudaFree(true_false);
	cudaFree(accuracy_each_epoch);
	cudaFree(d_p_s);
	cudaFree(d_p_s->dWxh);
	cudaFree(d_p_s->dWhh);
	cudaFree(d_p_s->dWhy);
	cudaFree(d_p_s->dbh);
	cudaFree(d_p_s->dby);

	cublasDestroy(handle);
}


/*
	arma::mat in RNN-v2 => device_vector.

	inputs: data of a scenario
	M: n_rows of orig. inputs. 目前的 M=17 即signals的数目
	N: n_cols of orig. inputs. N 是matlab中matDataZScore的行数即time步

	注意：参数中有struct，当调用这个fn时，应先 cudaMallocManaged struct
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
)
{
}


void test_gpu_fns()
{
	cudaSetDevice(0);

	// ===========================================
	const int M = 50;
	const int N = 50;
	const int size = M*N;
	float* d_in = NULL;
	cudaMallocManaged((void**)&d_in, size * sizeof(float));
	printToHost(d_in, M, N, "initial values");
	gpu_fill_rand(d_in, size, 1, -0.1f, 0.1f, 11);
	printToHost(d_in, M, N, "rand");

	// gpu_clear_arr
	gpu_clear_arr(d_in, size);
	printToHost(d_in, M, N, "clear to 0");

	// ------ fill rand :) --------

	/*gpu_fill_rand(d_in, size, 1, -0.1f, 0.1f, 11);
	printToHost(d_in, M, N, "rand");

	// ------- gpu_copy(); :) ---------------
	float* d_cp = NULL;
	cudaMallocManaged((void**)&d_cp, 3 * sizeof(float));
	gpu_copy(d_cp, d_in, 1, 4);

	printToHost(d_cp, 1, 3, "copy");

	// ---------- gpu_fill(); :)-----------
	gpu_fill(d_in, size, 2.3);
	printToHost(d_in, M, N, "fill with 2.3");

	// free
	cudaFree(d_cp);*/
	cudaFree(d_in);
	
}

