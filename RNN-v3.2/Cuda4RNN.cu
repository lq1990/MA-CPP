﻿#include "Cuda4RNN.h"

void trainMultiThread(
	float* lossAllVec,
	Para* para
)
{
	int num_sces			= para->h_num_sces[0];
	int total_epoches		= para->h_total_epoches[0];
	int n_features			= para->h_n_features[0];
	int n_hidden			= para->h_n_hidden[0];
	int n_output_classes	= para->h_n_output_classes[0];
	float alpha				= para->h_alpha[0];
	float score_min			= para->h_score_min[0];
	float score_max			= para->h_score_max[0];
	{
		cout << "train begins" << endl << "alpha: " << alpha << endl;
		cout << "total_epoches: " << total_epoches << endl 
			<< "n_features: " << n_features << endl;
		cout << "n_hidden: " << n_hidden << endl 
			<< "n_output_classes: " << n_output_classes << endl;
		cout << "score_min: " << score_min << endl 
			<< "score_max: " << score_max << endl;
	}

	const int n_streams = 10;
	cudaStream_t stream[n_streams];
	for (int i = 0; i < n_streams; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
	cublasHandle_t handle;
	cublasCreate(&handle);


	// 先用 标准SGD优化，使用1个cpu线程
	float* loss_one_epoch;
	float* loss_mean_each_epoch;
	float* true_false;
	float* accuracy_each_epoch;
	float loss = 0.f;
	cudaMalloc((void**)&loss_one_epoch, total_epoches * sizeof(float));
	cudaMalloc((void**)&loss_mean_each_epoch, total_epoches * sizeof(float));
	cudaMalloc((void**)&true_false, total_epoches * sizeof(float));
	cudaMalloc((void**)&accuracy_each_epoch, total_epoches * sizeof(float));

	int total_size = para->h_total_size[0];
	int Nmax = para->d_Nmax[0];
	int M = para->h_sces_data_mn[0];
	float* sce_item_data; // 给此分配和 scedata一样的大小
	cudaMalloc((void**)&sce_item_data, total_size*sizeof(float));
	float* hprev;
	cudaMalloc((void**)&hprev, n_hidden * sizeof(float));

	// total_epoches
	for (int i = 0; i < 21; i++)
	{

		/**
			loop over each scenario
		*/
		for(int item = 0; item < num_sces; item++)
		{

			// ---------- 先取出一个 场景数据 训练 RNN ------------
			float id0		= para->h_sces_id_score[item*2 + 0]; 
			float score0	= para->h_sces_id_score[item*2 + 1];
			int sce0_M		= para->h_sces_data_mn[item * 2 + 0]; 
			int sce0_N		= para->h_sces_data_mn[item * 2 + 1];

			int beginIdx	= para->h_sces_data_idx_begin[item];
			int endIdx		= para->h_sces_data_idx_begin[item + 1];
			//gpu_copy(sce_item_data, 0, para->d_sces_data, beginIdx, endIdx);
			sce_item_data = para->d_sces_data + beginIdx;
			// 使用浅拷贝

			//gpu_clear_arr(true_false, total_epoches);//true_false.clear();
			//gpu_clear_arr(loss_one_epoch, total_epoches);//loss_one_epoch.clear();

			// set dP 0
			gpu_fill(para->d_dWxh, n_hidden * n_features, 0.f);
			gpu_fill(para->d_dWhh, n_hidden * n_hidden, 0.f);
			gpu_fill(para->d_dWhy, n_hidden * n_output_classes, 0.f);
			gpu_fill(para->d_dbh,  n_hidden, 0.f);
			gpu_fill(para->d_dby,  n_output_classes, 0.f);

			gpu_fill(hprev, n_hidden, 0.f);
		
			lossFun(handle, 
				sce_item_data, sce0_M, sce0_N,
				score0, 
				hprev, true_false, 
				loss,
				para,
				stream);

			if (i % 10 == 0)
			{
				cout << "epoch: " << i << ", loss: " << loss << endl;
			}

			sgd(handle, para);
		}
		
	}

	// lossVec mean, accu

	// free resource
	;{
		deInitPara(para);
		cudaFree(hprev); 
		cudaFree(sce_item_data);
		cudaFree(loss_one_epoch);
		cudaFree(loss_mean_each_epoch);
		cudaFree(true_false);
		cudaFree(accuracy_each_epoch);
		cout << "free over in train fn \n";
	}
	cublasDestroy(handle);

	for (int i = 0; i < n_streams; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
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
	Para* para,
	cudaStream_t stream[]
)
{
	int total_epoches		= para->h_total_epoches[0];
	int n_features			= para->h_n_features[0];
	int n_hidden			= para->h_n_hidden[0];
	int n_output_classes	= para->h_n_output_classes[0];
	float alpha				= para->h_alpha[0];
	float score_min			= para->h_score_min[0];
	float score_max			= para->h_score_max[0];
	int Nmax				= para->d_Nmax[0]; // 所有场景的N中最大的数值

	int idx1_targets = -1;
	score2onehot(score,
		idx1_targets, n_output_classes, score_min, score_max);

	// hs[-1] = hprev;
	gpu_set_col(para->d_hs, n_hidden, Nmax, -1+1, hprev);
	loss = 0.f;

	// ---------------- forward pass -------------
	for (int t = 0; t < N; t++)
	{
		// ----- xs[t] = inputs.row(t).t(); ----- 因为xs中存储的都是 inputs数据，不必再copy一份
		//gpu_get_col(inputs, M, N, t, para->d_tmp_d_vec);  // tmp saves xs[t]
		//gpu_set_col(para->d_xs, n_features, Nmax, t, para->d_tmp_d_vec);

		// --- hs[t] = arma::tanh(Wxh * xs[t] + Whh*hs[t-1] + bh); ----
		// Wxh * xs[t]
		// hs[t-1]
		
		/*gpu_get_col(para->d_hs, n_hidden, Nmax, t - 1 + 1, 
			para->d_tmp_d_vec2);*/ // tmp2 saves hs[t-1]
		// Whh * hs[t-1]
		/*gpu_mv(handle, para->Whh, para->d_tmp_d_vec, n_hidden, n_hidden,
			para->d_W_tmp2);*/ // W_tmp2 saves Whh*hs[t-1]
		/*gpu_tanh_add_add(para->d_W_tmp1, para->d_W_tmp2, para->bh, n_hidden, 
			para->d_tmp_d_vec);*/ // tmp saves tanh_add_add
		gpu_tanh_Mv_add_Mv_add_v(handle,
			para->d_Wxh, n_hidden, n_features, 
			inputs + t*M, // xs[t]
			para->d_Whh, n_hidden, n_hidden, 
			para->d_hs + n_hidden*(t-1+1), // hs[t-1]
			para->d_bh,
			para->d_hs + n_hidden*(t+1), // dest => hs[t]
			para,
			stream);
		//gpu_set_col(para->d_hs, n_hidden, Nmax, t + 1, para->d_W_tmp3);

		if (t == N-1)
		{
			// ys[t] = Why * hs[t] + by;
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t + 1, 
				para->d_tmp_d_vec);*/ // tmp saves hs[t]
			gpu_mv(handle, para->d_Why, 
				para->d_hs + n_hidden*(t+1), // hs[t]
				n_output_classes, n_hidden, 
				para->d_W_tmp1); // Why * hs[t]
			gpu_add(para->d_W_tmp1, para->d_by, n_output_classes, 
				para->d_ys + n_output_classes*t, stream[0]); // tmp saves ys[t]
			cudaStreamSynchronize(stream[0]);
			//gpu_set_col(para->d_ys, n_output_classes, Nmax, t, para->d_tmp_d_vec);

			// ps[t] = softmax(ys[t])
			int sum1 = n_features + n_features + n_output_classes;
			gpu_clear_arr(para->d_W_tmp1, sum1*sum1);
			gpu_get_col(para->d_ys, n_output_classes, Nmax, t, para->d_tmp_d_vec);

			gpu_softmax(para->d_tmp_d_vec, n_output_classes, 
				para->d_W_tmp1, // dest => ps[t]
				para->d_W_tmp2); // cache
			gpu_set_col(para->d_ps, n_output_classes, Nmax, t, 
				para->d_W_tmp1); // d_W_tmp1 = softmax = ps[t]
			// loss += -log(ps[t](idx1));

			/*float val = para->d_W_tmp1[idx1_targets];
			loss += -logf(val);*/
			cudaMemcpy(para->h_cache, para->d_W_tmp1, sum1*sum1 * sizeof(float), cudaMemcpyDeviceToHost);
			float val = para->h_cache[idx1_targets];
			loss = -logf(val);

			// idx_pred
			gpu_max_index(para->d_W_tmp1, sum1*sum1, para->d_W_tmp2);

		}

	}

	// ---------------- BPTT -------------
	gpu_fill(para->d_dWxh, n_hidden*n_features, 0.f);
	gpu_fill(para->d_dWhh, n_hidden*n_hidden, 0.f);
	gpu_fill(para->d_dWhy, n_hidden*n_output_classes, 0.f);
	gpu_fill(para->d_dbh, n_hidden, 0.f);
	gpu_fill(para->d_dby, n_output_classes, 0.f);
	gpu_fill(para->d_dhnext, n_hidden, 0.f); // init dhnext = 0

	for (int t = N-1; t >= 0; t--)
	{

		if (t == N-1)
		{
			// dy = ps[t];
			gpu_get_col(para->d_ps, n_output_classes, Nmax, t, para->d_dy);

			// uvec fuvec = arma::find(targets == 1);
			// dy[fuvec(0)] -= 1;
			// ??? para->d_dy[idx1_targets] -= 1.f;
			gpu_update_dy(para->d_dy, n_output_classes, idx1_targets);

			// dWhy += dy * hs[t].t(); /// dy(10,1) * hs[t].t()(1,50) = (10,50)
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t + 1, 
				para->d_tmp_d_vec);*/ // tmp saves hs[t]'
			gpu_mmul(handle, para->d_dy, 
				para->d_hs+n_hidden*(t+1), // hs[t]'
				n_output_classes, 1, n_hidden, 
				para->d_W_tmp1, stream[0]); // Wtmp1 saves dy*hs[t]'
			gpu_add(para->d_dWhy, para->d_W_tmp1, n_output_classes*n_hidden, 
				para->d_dWhy, stream[0]);

			// dby += dy;
			gpu_add(para->d_dby, para->d_dy, n_output_classes, para->d_dby, stream[1]);

			// dh = Why.t() * dy + dhnext;
			gpu_mv(handle, para->d_Why, para->d_dy, n_output_classes, n_hidden,
				para->d_W_tmp1, true); // Wtmp1 saves Why' * dy
			gpu_add(para->d_W_tmp1, para->d_dhnext, n_hidden, para->d_dh, stream[2]);

			cudaStreamSynchronize(stream[2]);

			// dhraw = (1 - hs[t] % hs[t]) % dh; // mul elemwise
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t + 1, 
				para->d_tmp_d_vec); */// tmp saves hs[t]
			gpu_tanh_der_hs_dh(para->d_hs + n_hidden*(t+1), // hs[t]
				para->d_dh, n_hidden, 
				para->d_dhraw);

			// dbh += dhraw;
			gpu_add(para->d_dbh, para->d_dhraw, n_hidden, para->d_dbh, stream[3]);

			// dWxh += dhraw * xs[t].t(); // 惩罚项，只需要在loop中 加一次
			/*gpu_get_col(para->d_xs, n_features, Nmax, t, 
				para->d_tmp_d_vec);*/ // tmp saves xs[t]
			gpu_mmul(handle, para->d_dhraw, 
				inputs + t*M, // xs[t]
				n_hidden, 1, n_features,
				para->d_W_tmp1, stream[4]); // Wtmp1 saves dhraw*xs[t]'
			gpu_add(para->d_dWxh, para->d_W_tmp1, n_hidden*n_features, 
				para->d_dWxh, stream[4]);

			// dWhh += dhraw * hs[t - 1].t();
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t-1+1,
				para->d_tmp_d_vec);*/ // tmp saves hs[t-1]
			gpu_mmul(handle, para->d_dhraw, 
				para->d_hs + n_hidden * (t-1+1), // hs[t-1]
				n_hidden, 1, n_hidden,
				para->d_W_tmp1, stream[5]); // Wtmp1 saves dhraw*hs[t-1]'
			gpu_add(para->d_dWhh, para->d_W_tmp1, n_hidden*n_hidden,
				para->d_dWhh, stream[5]);

			// dhnext = Whh.t() * dhraw;
			gpu_mv(handle, para->d_Whh, para->d_dhraw, n_hidden, n_hidden, 
				para->d_dhnext, true);

			for (int i = 0; i <= 5; i++)
			{
				cudaStreamSynchronize(stream[i]);
			}

		}
		else
		{
			// dh = dhnext;
			para->d_dh = para->d_dhnext;

			// dhraw = (1 - hs[t] % hs[t]) % dh; // mul elemwise
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t + 1,
				para->d_tmp_d_vec); */// tmp saves hs[t]

			gpu_tanh_der_hs_dh(para->d_hs + n_hidden*(t+1), // hs[t]
				para->d_dh, n_hidden,
				para->d_dhraw);

			// 可并行的有： dbh, dWxh, dWhh, dhnext
			// dbh += dhraw;
			gpu_add(para->d_dbh, para->d_dhraw, n_hidden, para->d_dbh, stream[0]);

			// dWxh += dhraw * xs[t].t(); // 惩罚项，只需要在loop中 加一次
			/*gpu_get_col(para->d_xs, n_features, Nmax, t,
				para->d_tmp_d_vec);*/ // tmp saves xs[t]
			gpu_mmul(handle, para->d_dhraw, 
				inputs + t*M, // xs[t]
				n_hidden, 1, n_features,
				para->d_W_tmp1, stream[1]); // Wtmp1 saves dhraw*xs[t]'
			gpu_add(para->d_dWxh, para->d_W_tmp1, n_hidden*n_features,
				para->d_dWxh, stream[1]);

			// dWhh += dhraw * hs[t - 1].t();
			/*gpu_get_col(para->d_hs, n_hidden, Nmax, t - 1 + 1,
				para->d_tmp_d_vec);*/ // tmp saves hs[t-1]
			gpu_mmul(handle, para->d_dhraw, 
				para->d_hs + n_hidden*(t-1+1), // hs[t-1]
				n_hidden, 1, n_hidden,
				para->d_W_tmp2, stream[2]); // Wtmp1 saves dhraw*hs[t-1]'
			gpu_add(para->d_dWhh, para->d_W_tmp2, n_hidden*n_hidden,
				para->d_dWhh, stream[2]);

			// dhnext = Whh.t() * dhraw;
			gpu_mv(handle, para->d_Whh, para->d_dhraw, n_hidden, n_hidden,
				para->d_dhnext, true);

			cudaStreamSynchronize(stream[0]);
			cudaStreamSynchronize(stream[1]);
			cudaStreamSynchronize(stream[2]);

		}

	}

	// clip
	gpu_clip(para->d_dWxh, n_hidden*n_features,-5.f, 5.f);
	gpu_clip(para->d_dWhh, n_hidden*n_hidden,-5.f, 5.f);
	gpu_clip(para->d_dWhy, n_hidden*n_output_classes,-5.f, 5.f);
	gpu_clip(para->d_dbh, n_hidden,-5.f, 5.f);
	gpu_clip(para->d_dby, n_output_classes,-5.f, 5.f);

}


void score2onehot(float score, 
	int & idx1_targets, int n_output_classes, float score_min, float score_max)
{
	float part = 1.0f / n_output_classes;
	float pos = (score - score_min) / (score_max - score_min + pow(10, -4));

	idx1_targets = floor(pos / part);

	//// init onehot with 0
	//gpu_fill(onehot, n_output_classes, 0.f);
	//// set 1
	//onehot[idx1_targets] = 1.f;

	//gpu_fill_onehot(onehot, n_output_classes, idx1_targets);

	return;
}


void sgd(cublasHandle_t handle, Para* para)
{
	int n_features			= para->h_n_features[0];
	int n_hidden			= para->h_n_hidden[0];
	int n_output_classes	= para->h_n_output_classes[0];
	float alpha				= para->h_alpha[0];

	sgd0(handle, para->d_Wxh, para->d_dWxh, n_hidden*n_features, alpha);
	sgd0(handle, para->d_Whh, para->d_dWhh, n_hidden*n_hidden, alpha);
	sgd0(handle, para->d_Why, para->d_dWhy, n_hidden*n_output_classes, alpha);
	sgd0(handle, para->d_bh,  para->d_dbh, n_hidden, alpha);
	sgd0(handle, para->d_by,  para->d_dby, n_output_classes, alpha);
}


void sgd0(cublasHandle_t handle, float * P, float * dP,
	int size, float alpha)
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
	const int K = 11;
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
	gpu_fill(d_x, K, 0.f); // d_x
	gpu_fill_rand(d_x2, 1, K, -4.f, 4.f, 43);
	gpu_fill_rand(d_x3, 1, K, 0.f, 1.f, 123);
	//printToHost(d_in1, M, K, "in1");
	//printToHost(d_in2, K, N, "rand2");
	d_x[1] = 0;
	//printToHost(d_x, 1, K, "x");
	printToHost(d_x2, 1, K, "x2");
	//printToHost(d_x3, 1, K, "x3");

	// --------- gpu_clip :)-------------
	/*gpu_clip(d_x2, K, -1.f, 1.f);
	printToHost(d_x2, 1, K, "clip x2");*/

	// ------ gpu_max_value -----------
	/*float* cache;
	cudaMallocManaged((void**)&cache, K * sizeof(float));
	printToHost(cache, 1, K, "init cache");

	float x2_max = gpu_max_value(d_x2, K, cache);
	cout << "max val of x2: " << x2_max << endl;

	int idx = gpu_max_index(d_x2, K, cache);
	cout << "index of max val of x2: " << idx << endl;*/

	// ----------- gpu_sum -------------
	/*float* cache;
	cudaMallocManaged((void**)&cache, K * sizeof(float));
	printToHost(cache, 1, K, "init cache");

	float s = gpu_sum(d_x, K, cache);
	cout << "sum of x: " << s << endl;

	printToHost(cache, 1, K+5, "cache");*/

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

void initPara(Para * para, int Nmax)
{
	// malloc 
	cudaMallocHost((void**)&para->h_total_epoches, sizeof(float));
	cudaMallocHost((void**)&para->h_n_features, sizeof(float));
	cudaMallocHost((void**)&para->h_n_hidden, sizeof(float));
	cudaMallocHost((void**)&para->h_n_output_classes, sizeof(float));
	cudaMallocHost((void**)&para->h_alpha, sizeof(float));
	cudaMallocHost((void**)&para->h_score_min, sizeof(float));
	cudaMallocHost((void**)&para->h_score_max, sizeof(float));

	cudaMalloc((void**)&para->d_total_epoches, sizeof(float));
	cudaMalloc((void**)&para->d_n_features, sizeof(float));
	cudaMalloc((void**)&para->d_n_hidden, sizeof(float));
	cudaMalloc((void**)&para->d_n_output_classes, sizeof(float));
	cudaMalloc((void**)&para->d_alpha, sizeof(float));
	cudaMalloc((void**)&para->d_score_min, sizeof(float));
	cudaMalloc((void**)&para->d_score_max, sizeof(float));

	// malloc dP, only in GPU needed
	cudaMalloc((void**)&para->d_dWxh, n_hidden*n_features * sizeof(float));
	cudaMalloc((void**)&para->d_dWhh, n_hidden*n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_dWhy, n_hidden*n_output_classes * sizeof(float));
	cudaMalloc((void**)&para->d_dbh, n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_dby, n_output_classes * sizeof(float));
	cudaMalloc((void**)&para->d_dhnext, n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_dy, n_output_classes * sizeof(float));
	cudaMalloc((void**)&para->d_dh, n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_dhraw, n_hidden * sizeof(float));

	// malloc params of RNN
	cudaMallocHost((void**)&para->h_Wxh, n_hidden*n_features * sizeof(float));
	cudaMallocHost((void**)&para->h_Whh, n_hidden*n_hidden * sizeof(float));
	cudaMallocHost((void**)&para->h_Why, n_hidden*n_output_classes * sizeof(float));
	cudaMallocHost((void**)&para->h_bh, n_hidden * sizeof(float));
	cudaMallocHost((void**)&para->h_by, n_output_classes * sizeof(float));

	cudaMalloc((void**)&para->d_Wxh, n_hidden*n_features * sizeof(float));
	cudaMalloc((void**)&para->d_Whh, n_hidden*n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_Why, n_hidden*n_output_classes * sizeof(float));
	cudaMalloc((void**)&para->d_bh, n_hidden * sizeof(float));
	cudaMalloc((void**)&para->d_by, n_output_classes * sizeof(float));

	// malloc state, only in GPU needed
	cudaMalloc((void**)&para->d_xs, n_features*Nmax * sizeof(float));
	cudaMalloc((void**)&para->d_hs, n_hidden*(Nmax + 1) * sizeof(float));
	cudaMalloc((void**)&para->d_ys, n_output_classes*Nmax * sizeof(float));
	cudaMalloc((void**)&para->d_ps, n_output_classes*Nmax * sizeof(float));
	cudaMallocManaged((void**)&para->d_Nmax, sizeof(float));
	para->d_Nmax[0] = Nmax;
	
	// malloc cache
	int sum1 = n_features + n_features + n_output_classes;
	cudaMalloc((void**)&para->d_tmp_d_vec, sum1 * sizeof(float));
	cudaMalloc((void**)&para->d_tmp_d_vec2, sum1 * sizeof(float));
	cudaMalloc((void**)&para->d_W_tmp1, sum1*sum1 * sizeof(float));
	cudaMalloc((void**)&para->d_W_tmp2, sum1*sum1 * sizeof(float));
	cudaMalloc((void**)&para->d_W_tmp3, sum1*sum1 * sizeof(float));

	cudaMallocHost((void**)&para->h_cache, sum1*sum1 * sizeof(float));
	

}

void deInitPara(Para * para)
{
	// free 
	cudaFreeHost(para->h_total_epoches);
	cudaFreeHost(para->h_n_features);
	cudaFreeHost(para->h_n_hidden);
	cudaFreeHost(para->h_n_output_classes);
	cudaFreeHost(para->h_alpha);
	cudaFreeHost(para->h_score_min);
	cudaFreeHost(para->h_score_max);

	cudaFree(para->d_total_epoches);
	cudaFree(para->d_n_features);
	cudaFree(para->d_n_hidden);
	cudaFree(para->d_n_output_classes);
	cudaFree(para->d_alpha);
	cudaFree(para->d_score_min);
	cudaFree(para->d_score_max);

	// free dP, only GPU
	cudaFree(para->d_dWxh);
	cudaFree(para->d_dWhh);
	cudaFree(para->d_dWhy);
	cudaFree(para->d_dbh);
	cudaFree(para->d_dby);
	cudaFree(para->d_dhnext);
	cudaFree(para->d_dy);
	cudaFree(para->d_dh);
	cudaFree(para->d_dhraw);

	// free params of RNN
	cudaFreeHost(para->h_Wxh);
	cudaFreeHost(para->h_Whh);
	cudaFreeHost(para->h_Why);
	cudaFreeHost(para->h_bh);
	cudaFreeHost(para->h_by);

	cudaFree(para->d_Wxh);
	cudaFree(para->d_Whh);
	cudaFree(para->d_Why);
	cudaFree(para->d_bh);
	cudaFree(para->d_by);

	// free sces
	cudaFreeHost(para->h_sces_id_score);
	cudaFreeHost(para->h_sces_data);
	cudaFreeHost(para->h_sces_data_mn);
	cudaFreeHost(para->h_sces_data_idx_begin);
	cudaFreeHost(para->h_num_sces);
	cudaFreeHost(para->h_total_epoches);

	cudaFree(para->d_sces_id_score);
	cudaFree(para->d_sces_data);
	cudaFree(para->d_sces_data_mn);
	cudaFree(para->d_sces_data_idx_begin);
	cudaFree(para->d_num_sces);
	cudaFree(para->d_total_epoches);

	// free state, inly GPU
	cudaFree(para->d_xs);
	cudaFree(para->d_hs);
	cudaFree(para->d_ys);
	cudaFree(para->d_ps);
	cudaFree(para->d_Nmax);

	// free cache, only  GPU
	int sum1 = n_features + n_features + n_output_classes;
	cudaFree(para->d_tmp_d_vec);
	cudaFree(para->d_tmp_d_vec2);
	cudaFree(para->d_W_tmp1);
	cudaFree(para->d_W_tmp2);
	cudaFree(para->d_W_tmp3);
}

