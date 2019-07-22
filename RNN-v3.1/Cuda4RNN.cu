#include "Cuda4RNN.h"

/*
	需要提前准备数据的：sces_s, rnn_p_s 
*/
void trainMultiThread(
	float* lossAllVec,
	sces_struct* sces_s,
	params_struct* p_s,
	rnn_params_struct* rnn_p_s,
	cache_struct* cache_s
)
{
	int total_epoches = rnn_p_s->total_epoches;
	int n_features = rnn_p_s->n_features;
	int n_hidden = rnn_p_s->n_hidden;
	int n_output_classes = rnn_p_s->n_output_classes;
	float alpha = rnn_p_s->alpha;
	float score_min = rnn_p_s->score_min;
	float score_max = rnn_p_s->score_max;
	cout << "train begins" << endl << "alpha: " << alpha << endl;
	cout << "total_epoches: " << total_epoches << endl 
		<< "n_features: " << n_features << endl;
	cout << "n_hidden: " << n_hidden << endl 
		<< "n_output_classes: " << n_output_classes << endl;
	cout << "score_min: " << score_min << endl 
		<< "score_max: " << score_max << endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	// 先用 标准SGD优化，使用1个cpu线程
	float* loss_one_epoch;
	float* loss_mean_each_epoch;
	float* true_false;
	float* accuracy_each_epoch;
	float loss = 0.f;
	cudaMallocManaged((void**)&loss_one_epoch, total_epoches * sizeof(float));
	cudaMallocManaged((void**)&loss_mean_each_epoch, total_epoches * sizeof(float));
	cudaMallocManaged((void**)&true_false, total_epoches * sizeof(float));
	cudaMallocManaged((void**)&accuracy_each_epoch, total_epoches * sizeof(float));

	// 先取出一个 场景数据 训练 RNN
	float id0 = sces_s->sces_id_score[0]; cout << "id0: " << id0 << endl;
	float score0 = sces_s->sces_id_score[1]; cout << "score0: " << score0 << endl;
	int sce0_M = sces_s->sces_data_mn[0]; cout << "sce0_M: " << sce0_M << endl;
	int sce0_N = sces_s->sces_data_mn[1]; cout << "sce0_N: " << sce0_N << endl;

	float* sce0_data;
	cudaMallocManaged((void**)&sce0_data, sce0_M*sce0_N*sizeof(float));
	int beginIdx = sces_s->sces_data_idx_begin[0];
	int endIdx = sces_s->sces_data_idx_begin[1];
	gpu_copy(sce0_data,0, sces_s->sces_data, beginIdx, endIdx);
	 
	float* hprev;
	cudaMallocManaged((void**)&hprev, n_hidden * sizeof(float));

	d_params_struct* d_p_s;
	cudaMallocManaged((void**)&d_p_s, sizeof(d_params_struct));
	cudaMallocManaged((void**)&d_p_s->dWxh, n_hidden*n_features * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dWhh, n_hidden*n_hidden * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dWhy, n_hidden*n_output_classes * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dbh, n_hidden * sizeof(float));
	cudaMallocManaged((void**)&d_p_s->dby, n_output_classes * sizeof(float));

	for (int i = 0; i < total_epoches; i++)
	{
		gpu_clear_arr(true_false, total_epoches);//true_false.clear();
		gpu_clear_arr(loss_one_epoch, total_epoches);//loss_one_epoch.clear();

		// set dP 0
		/*thrust::fill(d_p_s->dWxh.begin(), d_p_s->dWxh.end(), 0.f);
		thrust::fill(d_p_s->dWhh.begin(), d_p_s->dWhh.end(), 0.f);
		thrust::fill(d_p_s->dWhy.begin(), d_p_s->dWhy.end(), 0.f);
		thrust::fill(d_p_s->dbh.begin(),  d_p_s->dbh.end(), 0.f);
		thrust::fill(d_p_s->dby.begin(),  d_p_s->dby.end(), 0.f);*/
		gpu_fill(d_p_s->dWxh, n_hidden * n_features, 0.f);
		gpu_fill(d_p_s->dWhh, n_hidden * n_hidden, 0.f);
		gpu_fill(d_p_s->dWhy, n_hidden * n_output_classes, 0.f);
		gpu_fill(d_p_s->dbh,  n_hidden, 0.f);
		gpu_fill(d_p_s->dby,  n_output_classes, 0.f);

		//thrust::fill(hprev.begin(), hprev.end(), 0.f);
		gpu_fill(hprev, n_hidden, 0.f);
		
		lossFun(handle, 
			sce0_data, sce0_M, sce0_N,
			score0, 
			hprev, true_false, 
			loss,
			p_s,
			d_p_s,
			rnn_p_s, 
			cache_s);

		// use 'd_p_s' to update 'p_s' in each epoch
		/*printToHost(p_s->Wxh, n_hidden, n_features, "Wxh");
		printToHost(d_p_s->dWxh, n_hidden, n_features, "dWxh");
		printToHost(p_s->Why, n_output_classes, n_hidden, "Why");
		printToHost(d_p_s->dWhy, n_output_classes, n_hidden, "dWhy");
		printToHost(p_s->Whh, n_hidden, n_hidden, "Whh");
		printToHost(d_p_s->dWhh, n_hidden, n_hidden, "dWhh");*/

		sgd(handle, p_s, d_p_s, rnn_p_s);

		// print loss in each epoch
		if (i % 5 == 0)
		{
			cout << "epoch " << i << ", loss: " << loss << endl;
		}
	}

	// lossVec mean, accu


	// free resource
	cudaFree(hprev); 
	cudaFree(sce0_data); 
	cudaFree(loss_one_epoch);
	cudaFree(loss_mean_each_epoch);
	cudaFree(true_false);
	cudaFree(accuracy_each_epoch);
	cudaFree(d_p_s->dWxh);// 应先释放 d_p_s中成员,再释放dps。反过来的话，找不到
	cudaFree(d_p_s->dWhh);
	cudaFree(d_p_s->dWhy);
	cudaFree(d_p_s->dbh);
	cudaFree(d_p_s->dby);
	cudaFree(d_p_s);
	cout << "free over in train fn \n";
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
	rnn_params_struct* rnn_p_s,
	cache_struct* cache_s
)
{
	cout << "lossFun... \n";
	int total_epoches = rnn_p_s->total_epoches;
	int n_features = rnn_p_s->n_features;
	int n_hidden = rnn_p_s->n_hidden;
	int n_output_classes = rnn_p_s->n_output_classes;
	float alpha = rnn_p_s->alpha;
	float score_min = rnn_p_s->score_min;
	float score_max = rnn_p_s->score_max;

	int idx1_targets = -1;
	float* targets = score2onehot(score, 
		idx1_targets, n_output_classes, score_min, score_max);



}

float * score2onehot(float score, 
	int & idx1_targets, int n_output_classes, float score_min, float score_max)
{
	float part = 1.0f / n_output_classes;
	float pos = (score - score_min) / (score_max - score_min + pow(10, -8));

	idx1_targets = floor(pos / part);

	float* onehot;
	cudaMallocManaged((void**)&onehot, n_output_classes * sizeof(float));
	// init onehot with 0
	gpu_fill(onehot, n_output_classes, 0.f);
	// set 1
	onehot[idx1_targets] = 1.f;

	return onehot;
}


void sgd(cublasHandle_t handle, params_struct* P, d_params_struct* dP,
	rnn_params_struct* rnn_p_s)
{
	int n_features = rnn_p_s->n_features;
	int n_hidden = rnn_p_s->n_hidden;
	int n_output_classes = rnn_p_s->n_output_classes;
	float alpha = rnn_p_s->alpha;

	sgd0(handle, P->Wxh, dP->dWxh, n_hidden*n_features, alpha);
	sgd0(handle, P->Whh, dP->dWhh, n_hidden*n_hidden, alpha);
	sgd0(handle, P->Why, dP->dWhy, n_hidden*n_output_classes, alpha);
	sgd0(handle, P->bh, dP->dbh, n_hidden, alpha);
	sgd0(handle, P->by, dP->dby, n_output_classes, alpha);
}


void sgd0(cublasHandle_t handle, float * P, float * dP, int size, float alpha)
{
	// P = - alpha * dP + P
	// cublasSaxpy: y = a * x +  y

	//cublasStatus_t stat;
	float a = -alpha;

	cublasSaxpy_v2(handle,
		size, // num of elems in P or dP
		&a,
		dP, 1,
		P, 1); // res into P
	//cout << "stat: " << stat << endl;

	cudaDeviceSynchronize(); 
	// cublas执行后，必须跟一个同步，否则会因为数据同步问题报错。
}

void test_gpu_fns()
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	// ===========================================
	const int M = 10;
	const int K = 600;
	const int N = 70;
	const int size1 = M*K;
	const int size2 = K*N;
	const int size3 = M*N;
	float* d_in1 = NULL;
	float* d_in2 = NULL;
	float* d_out = NULL;
	float* d_x;
	float* d_x2;
	float* d_x3;
	cudaMallocManaged((void**)&d_in1, size1 * sizeof(float));
	cudaMallocManaged((void**)&d_in2, size2 * sizeof(float));
	cudaMallocManaged((void**)&d_out, size3 * sizeof(float));
	cudaMallocManaged((void**)&d_x, K * sizeof(float));
	cudaMallocManaged((void**)&d_x2, K * sizeof(float));
	cudaMallocManaged((void**)&d_x3, K * sizeof(float));
	cudaDeviceSynchronize();
	//printToHost(d_in1, M, K, "d_in1 initial");
	//printToHost(d_in2, K, N, "d_in2 initial");

	gpu_fill(d_in1, size1, 3.f);
	gpu_fill_rand(d_in2, size2, 1, -0.1f, 0.1f, 111);
	gpu_fill(d_x, K, 1.f);
	gpu_fill_rand(d_x2, 1, K);
	gpu_fill_rand(d_x3, 1, K, 0.f, 1.f, 123);
	//printToHost(d_in1, M, K, "in1");
	//printToHost(d_in2, K, N, "rand2");
	d_x[1] = 10;
	printToHost(d_x, 1, K, "x");
	//printToHost(d_x2, 1, K, "x2");
	//printToHost(d_x3, 1, K, "x3");

	// ----------- gpu_sum -------------
	float* cache;
	cudaMallocManaged((void**)&cache, K * sizeof(float));
	printToHost(cache, 1, K, "init cache");

	float s = gpu_sum(d_x, K, cache);
	cout << "sum of x: " << s << endl;

	printToHost(cache, 1, K+5, "cache");

	// --------- gpu_softmax -----------
	/*float* soft;
	cudaMallocManaged((void**)&soft, K * sizeof(float));
	gpu_softmax(d_x, K, soft, cache);
	printToHost(soft, 1, K, "softmax of x");*/

	// ------------ gpu_scal -----------
	/*float* dest;
	cudaMallocManaged((void**)&dest, K * sizeof(float));
	gpu_scal(d_x2, K, 0.1, dest);
	printToHost(dest, 1, K, "scal of x2");*/

	// -------- gpu_tanh_add_add :) --------------
	/*float* dest;
	cudaMallocManaged((void**)&dest, K * sizeof(float));
	gpu_tanh_add_add(d_x, d_x2, d_x3, K, dest);
	printToHost(dest, 1, K, "tanh(v1+v2+v3)");*/
	
	// ------------ gpu_tanh :)------------------
	/*float* res_tanh;
	cudaMallocManaged((void**)&res_tanh, K * sizeof(float));
	gpu_tanh(d_x, K, res_tanh);
	printToHost(res_tanh, 1, K, "tanh(x)");*/

	// ----------- gpu_mul_elemwise :)--------
	//float* mul;
	//cudaMallocManaged((void**)&mul, M*K * sizeof(float));
	//gpu_mul_elemwise(d_in1, d_in1, M*K, mul);
	//printToHost(mul, M, K, "mul.");

	// ----------- gpu_add :) --------------------
	/*float* add;
	cudaMallocManaged((void**)&add, M*K * sizeof(float));
	gpu_add(d_in1, d_in1, M*K, add);
	printToHost(add, M, K, "add");*/

	// -------------- gpu_mmul :)--------------
	/*gpu_mmul(handle, d_in1, d_in2, M, K, N, d_out);
	printToHost(d_out, M, N, "in1 * in2");*/
	
	// -------------- gpu_mv :)--------------
	/*float* Ax;
	cudaMallocManaged((void**)&Ax, M * sizeof(float));
	gpu_mv(handle, d_in1, d_x, M, K, Ax, false);
	printToHost(Ax, M, 1, "Ax");*/

	// ------------ get get/set col :) -----------
	/*float* col1;
	cudaMallocManaged((void**)&col1, M * sizeof(float));

	gpu_get_col(d_in, M, N, 1, col1);
	printToHost(col1, M, 1, "col1");

	float* setVal;
	cudaMallocManaged((void**)&setVal, M, 1);
	gpu_fill(setVal, M * 1, 2.3f);
	gpu_set_col(d_in, M, N, 3, setVal);
	printToHost(d_in, M, N, "set col3 to 2.3");*/

	// --------- gpu_copy :) --------------
	/*float* d_cp = NULL;
	cudaMallocManaged((void**)&d_cp, 2*M * sizeof(float));
	gpu_copy(d_cp, 5, d_in, 0, M);
	printToHost(d_cp, M, 2, "copy first col of d_in");*/

	// ----------- score2onehot :) -------
	/*int idx1;
	float* onehot = score2onehot(7.0f, idx1, 10, 6.0, 8.9);
	cout << "idx1: " << idx1 << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << onehot[i] << "  " << endl;
	}
	cout << endl;*/

	// ------ gpu_clear_arr -----------
	//gpu_clear_arr(d_in, size);
	//printToHost(d_in, M, N, "clear to 0");

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
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	cublasDestroy(handle);
}

