#include "Cuda4RNN.h"

//device_vector<float> Cuda4RNN::Wxh = 
//	gpu_generate_rand(MyParams::n_hidden, MyParams::n_features, 
//		-0.01, 0.01f, 1);
//
//device_vector<float> Cuda4RNN::Whh =
//	gpu_generate_rand(MyParams::n_hidden, MyParams::n_hidden, 
//		-0.01, 0.01f, 11);
//
//device_vector<float> Cuda4RNN::Why =
//	gpu_generate_rand(MyParams::n_output_classes, MyParams::n_hidden, 
//		-0.01, 0.01f, 111);
//
//device_vector<float> Cuda4RNN::bh =
//	gpu_generate_rand(MyParams::n_hidden, 1, -0.01, 0.01f, 22);
//
//device_vector<float> Cuda4RNN::by =
//	gpu_generate_rand(MyParams::n_output_classes, 1, -0.01, 0.01f, 222);



//Cuda4RNN::Cuda4RNN()
//{
//}
//
//
//Cuda4RNN::~Cuda4RNN()
//{
//}
//
//void Cuda4RNN::initParams()
//{
//	//cout << "init alpha epoch n score" << endl;
//	/*alpha[0] = 0.1f;
//	total_epoches[0] = 501;
//	n_features[0] = 17;
//	n_hidden[0] = 50;
//	n_output_classes[0] = 10;
//	score_min[0] = 6.0f;
//	score_max[0] = 8.9f;*/
//
//	/*alpha = 0.1f;
//	total_epoches = 501;
//	n_features = 17;
//	n_hidden = 50;
//	n_output_classes = 10;
//	score_min = 6.0f;
//	score_max = 8.9f;*/
//
//	//cout << "init W b" << endl;
//	//this->Wxh = device_vector<float>(n_hidden * n_features);
//	//this->Whh = device_vector<float>(n_hidden * n_hidden);
//	//this->Why = device_vector<float>(n_output_classes * n_hidden);
//	//this->bh = device_vector<float>(n_hidden);
//	//this->by = device_vector<float>(n_output_classes);
//
//	///*Wxh = gpu_generate_rand(n_hidden, n_features, -0.01, 0.01f, 1);
//	//Whh = gpu_generate_rand(n_hidden, n_hidden, -0.01, 0.01f, 11);
//	//Why = gpu_generate_rand(n_output_classes, n_hidden, -0.01, 0.01f, 111);
//	//bh = gpu_generate_rand(n_hidden, 1, -0.01, 0.01f, 22);
//	//by = gpu_generate_rand(n_output_classes, 1, -0.01, 0.01f, 222);*/
//	//gpu_fill_rand(Wxh, n_hidden, n_features, -0.01f, 0.01f, 1);
//	//gpu_fill_rand(Whh, n_hidden, n_hidden, -0.01f, 0.01f, 11);
//	//gpu_fill_rand(Why, n_output_classes, n_hidden, -0.01f, 0.01f, 111);
//	//gpu_fill_rand(bh, n_hidden, 1, -0.01f, 0.01f, 22);
//	//gpu_fill_rand(by, n_output_classes, 1, -0.01f, 0.01f, 222);
//}


void trainMultiThread(
	/*device_vector<float>& sces_id_score,
	device_vector<float>& sces_data,
	device_vector<int>& sces_data_mn,
	device_vector<int>& sces_data_idx_begin,*/
	sces_struct* sces_s,
	device_vector<float>& lossAllVec,
	/*device_vector<float>& Wxh,
	device_vector<float>& Whh,
	device_vector<float>& Why,
	device_vector<float>& bh,
	device_vector<float>& by,*/
	params_struct* p_s,
	float alpha,
	int total_epoches,
	int n_features,
	int n_hidden,
	int n_output_classes,
	float score_min,
	float score_max,
	/*device_vector<float>& tmp_d_vec,
	device_vector<float>& d_cache1,
	device_vector<float>& d_cache2,
	device_vector<float>& d_cache3*/
	cache_struct* cache_s
)
{
	cout << "train begins, \ntotal_epoches: " << total_epoches << endl;
	cout << "alpha: " << alpha << endl;
	cout << total_epoches << endl;
	cout << n_features << endl;
	cout << n_hidden << endl;
	cout << n_output_classes << endl;
	cout << score_min << endl;
	cout << score_max << endl;
	cout << "Wxh size: " << p_s->Wxh.size() << endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	// init 一个tmp d_vec，用于存储get col的结果
	//device_vector<float> tmp_d_vec(n_hidden); // size = max(n_features, n_hidden, n_output_classes)

	// 先用 标准SGD优化，使用1个cpu线程
	//device_vector<float> lossAllVec; // log all loss
	device_vector<float> loss_one_epoch;
	device_vector<float> loss_mean_each_epoch;
	device_vector<float> true_false;
	device_vector<float> accuracy_each_epoch;
	device_vector<int> log_target;
	device_vector<int> log_prediction;

	float loss = 0.f;
	// 先取出一个 场景数据 训练 RNN
	float id0 = sces_s->sces_id_score[0];
	float score0 = sces_s->sces_id_score[1];
	int sce0_M = sces_s->sces_data_mn[0];
	int sce0_N = sces_s->sces_data_mn[1];

	//device_vector<float> sce0_data(sce0_M* sce0_N);
	device_vector<float> sce0_data(sces_s->sces_data, 
		sces_s->sces_data + sce0_M*sce0_N);

	//thrust::copy(sces_d.begin(),
	//	sces_d.end(),
	//	sce0_data.begin());

	/*device_vector<float> dWxh(n_hidden * n_features), 
		dWhh(n_hidden * n_hidden), 
		dWhy(n_output_classes * n_hidden), 
		dbh(n_hidden), 
		dby(n_output_classes),*/ 
	device_vector<float> hprev(n_hidden);
	d_params_struct* d_p_s;
	cudaMalloc((void**)&d_p_s, sizeof(d_params_struct));
	d_p_s->dWxh = device_vector<float>(n_hidden*n_features);
	d_p_s->dWhh = device_vector<float>(n_hidden*n_hidden);
	d_p_s->dWhy = device_vector<float>(n_hidden*n_output_classes);
	d_p_s->dbh = device_vector<float>(n_hidden);
	d_p_s->dby = device_vector<float>(n_output_classes);

	for (int i = 0; i < total_epoches; i++)
	{
		true_false.clear();
		loss_one_epoch.clear();

		/*dWxh = device_vector<float>(n_hidden * n_features, 0.f);
		dWhh = device_vector<float>(n_hidden * n_hidden, 0.f);
		dWhy = device_vector<float>(n_output_classes * n_hidden, 0.f);
		dbh = device_vector<float>(n_hidden, 0.f);
		dby = device_vector<float>(n_output_classes, 0.f);*/
		thrust::fill(d_p_s->dWxh.begin(), d_p_s->dWxh.end(), 0.f);
		thrust::fill(d_p_s->dWhh.begin(), d_p_s->dWhh.end(), 0.f);
		thrust::fill(d_p_s->dWhy.begin(), d_p_s->dWhy.end(), 0.f);
		thrust::fill(d_p_s->dbh.begin(),  d_p_s->dbh.end(), 0.f);
		thrust::fill(d_p_s->dby.begin(),  d_p_s->dby.end(), 0.f);

		//hprev = device_vector<float>(n_hidden, 0.f); // init hprev
		thrust::fill(hprev.begin(), hprev.end(), 0.f);

		lossFun(handle,
			sce0_data, sce0_M, sce0_N,
			score0,
			hprev,
			true_false,
			log_target,
			log_prediction,
			p_s,
			loss,
			d_p_s,
			n_features,
			n_hidden,
			n_output_classes,
			score_min,
			score_max,
			cache_s);

		lossAllVec.push_back(loss);
		loss_one_epoch.push_back(loss);
		if (i % 2 == 0)
		{
			// cout 属于cpu的范畴，gpu在运行时能 打印吗
			cout << "epoch " << i << ", loss: " << loss << endl;
		}

		// sgd, W = W - alpha * dW
		sgd(p_s->Wxh, d_p_s->dWxh, n_hidden *n_features, alpha);
		sgd(p_s->Whh, d_p_s->dWhh, n_hidden *n_hidden, alpha);
		sgd(p_s->Why, d_p_s->dWhy, n_hidden *n_output_classes, alpha);
		sgd(p_s->bh,  d_p_s->dbh, n_hidden, alpha);
		sgd(p_s->by,  d_p_s->dby, n_output_classes, alpha);

	}

	// lossVec mean, accu

	cudaFree(d_p_s);
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
	device_vector<float>& inputs, int M, int N,
	float score,
	device_vector<float>& hprev,
	device_vector<float>& true_false,
	device_vector<int>& log_target,
	device_vector<int>& log_prediction,
	/*device_vector<float>& Wxh,
	device_vector<float>& Whh,
	device_vector<float>& Why,
	device_vector<float>& bh,
	device_vector<float>& by,*/
	params_struct* p_s,
	float& loss,
	/*device_vector<float>& dWxh,
	device_vector<float>& dWhh,
	device_vector<float>& dWhy,
	device_vector<float>& dbh,
	device_vector<float>& dby,*/
	d_params_struct* d_p_s,
	int n_features,
	int n_hidden,
	int n_output_classes,
	float score_min,
	float score_max,
	/*device_vector<float>& tmp_d_vec,
	device_vector<float>& W_tmp1,
	device_vector<float>& W_tmp2,
	device_vector<float>& W_tmp3*/
	cache_struct* cache_s
)
{

	//float alpha  = this->alpha[0]; // constant mem on GPU
	//int total_epoches = this->total_epoches[0];
	//int n_features = this->n_features[0];
	//int n_hidden  =this->n_hidden[0];
	//int n_output_classes= this->n_output_classes[0];
	//float score_min  =this->score_min[0];
	//float score_max = this->score_max[0];

	// --------------------------------------
	int idx1_targets = -1; // index 1 in targets
	device_vector<float> targets = score2onehot(score, idx1_targets, 
		n_output_classes,
		score_min,
		score_max);

	// map<int, mat> xs, hs, ys, ps。 2维转1维vec
	device_vector<float> xs(M * N);
	device_vector<float> hs(n_hidden * (N+1));// hs比xs ys ps 多了一行，保存hprev
	device_vector<float> ys(n_output_classes * N);
	device_vector<float> ps(ys.size());
	
	// hs[-1] = hprev;
	gpu_set_col(hs, n_hidden, N + 1, 0, hprev); // vec没有-1，所以用0位置

	//device_vector<float> W_tmp1(n_hidden*n_hidden); // 需要验证下，大小设置是否有问题
	//device_vector<float> W_tmp2(n_hidden*n_hidden);
	// -------------- Forward Pass ---------------------

	// N 是time步数
	for (int t = 0; t < N; t++)
	{
		// ---------- xs[t] = inputs.row(t).t();
		gpu_get_col(inputs, M, N, t, cache_s->tmp_d_vec); // inputs.row(t)
		//device_vector<float> colValues = gpu_get_col(inputs, M, N, t); // getCol ==> inputs.row()
		gpu_set_col(xs, M, N, t, cache_s->tmp_d_vec);

		// --------- hs[t] = arma::tanh(Wxh * xs[t] + Whh*hs[t-1] + bh);
		gpu_get_col(xs, M, N, t, cache_s->tmp_d_vec); // xs[t] => tmp_d_vec
		//auto Wxh_xst = gpu_mv(handle, Wxh, tmp_d_vec, n_hidden, n_features);
		gpu_mv(handle, p_s->Wxh, cache_s->tmp_d_vec, n_hidden, n_features, cache_s->W_tmp1); // tmp1 => Wxh_xst
		gpu_get_col(hs, n_hidden, N + 1, t + 1 - 1, cache_s->tmp_d_vec); // hs[t-1] => tmp_d_vec
		//auto Whh_hs_t_1 = gpu_mv(handle, Whh, tmp_d_vec, n_hidden, n_hidden);
		gpu_mv(handle, p_s->Whh, cache_s->tmp_d_vec, n_hidden, n_hidden, cache_s->W_tmp2); // tmp2 => Whh_hs_t_1
		//auto W_W = gpu_add(Wxh_xst, Whh_hs_t_1, n_hidden);
		//auto W_W_bh = gpu_add(W_W, bh, n_hidden);
		//auto tanh_W_W_bh = gpu_tanh(W_W_bh, n_hidden);
		auto W_W = gpu_add(
			cache_s->W_tmp1, // Wxh_xst
			cache_s->W_tmp2, // Whh_hs_t_1
			n_hidden);
		auto W_W_bh = gpu_add(
			W_W, // W_W
			p_s->bh, n_hidden); // W_W_bh => tmp1
		gpu_set_col(hs, n_hidden, N + 1, t + 1, 
			gpu_tanh(
				W_W_bh, // W_W_bh
				n_hidden)); // tanh_W_W_bh

		if (t == N-1) // 当t到达最后一步
		{
			// ys[t] = Why * hs[t] + by;
			gpu_get_col(hs, n_hidden, N + 1, t + 1, cache_s->tmp_d_vec); // hs[t] => tmp
			/*auto Why_hst = 
				gpu_mv(handle, Why, tmp_d_vec,
					n_output_classes, n_hidden);*/
			//auto W_h_by = gpu_add(Why_hst, by, n_output_classes);
			gpu_mv(handle, p_s->Why, cache_s->tmp_d_vec, n_output_classes, n_hidden,
				cache_s->W_tmp1); // Why_hst => tmp1
			auto W_h_by = gpu_add(
				cache_s->W_tmp1, // Why_hst
				p_s->by, n_output_classes); // W_h_by => tmp1
			gpu_set_col(ys, n_output_classes, N, t, 
				W_h_by); // W_h_by

			// ps[t] = softmax(ys[t])
			gpu_get_col(ys, n_output_classes, N, t, cache_s->tmp_d_vec); // ys[t]
			auto soft = gpu_softmax(cache_s->tmp_d_vec, n_output_classes);
			gpu_set_col(ps, n_output_classes, N, t, 
				soft); // soft

			// loss += -log( ps[t](idx1_targets) )
			loss += -logf(soft[idx1_targets]);
			
			// find index_max(soft), index_prediction
			int idx_max_ps = gpu_max_index(soft);

			// push_back 此处多线程锁？？？

			//mtx.lock();
			log_target.push_back(idx1_targets);
			log_prediction.push_back(idx_max_ps);
			if (idx1_targets == idx_max_ps)
			{
				true_false.push_back(1);
			}
			else
			{
				true_false.push_back(0);
			}
			// mtx.unlock();
		}
	}

	// --------------- BPTT ---------------------------
	device_vector<float> dhnext(n_hidden, 0.f); // hs[0]的大小
	device_vector<float> dy, dh, dhraw;

	for (int t = N-1; t >= 0; t--)
	{
		if (t == N-1)
		{
			// dy = ps[t]
			gpu_get_col(ps, n_output_classes, N, t, cache_s->tmp_d_vec); // ps[t] => tmp
			dy = cache_s->tmp_d_vec;
			dy[idx1_targets] -= 1;
			// dWhy += dy * hs[t].t();
			gpu_get_col(hs, n_hidden, N + 1, t + 1, cache_s->tmp_d_vec); // hs[t] => tmp

			//auto dWhy_right = gpu_mmul(handle,
			//	dy, // dy
			//	tmp_d_vec, // hs[t]
			//	n_output_classes, 1, // dy_M, dy_N
			//	n_hidden); // hst'_N
			gpu_mmul(handle, dy, cache_s->tmp_d_vec, n_output_classes, 1, n_hidden,
				cache_s->W_tmp1); // dWhy_right => tmp1
			d_p_s->dWhy = gpu_add(d_p_s->dWhy,
				cache_s->W_tmp1,
				n_output_classes * n_hidden);
			// dby += dy;
			d_p_s->dby = gpu_add(d_p_s->dby, dy, n_output_classes);

			// dh = Why.t() * dy + dhnext;
			//auto Why_t_dy = 
				gpu_mv(handle, 
					p_s->Why,
				dy, 
				n_output_classes, n_hidden, 
					cache_s->W_tmp1,
				true);
			dh = gpu_add(dh, dhnext, n_hidden);

			// dhraw = (1 - hs[t] % hs[t]) % dh; // mul elemwise
			gpu_get_col(hs, n_hidden, N + 1, t + 1, cache_s->tmp_d_vec); // hs[t] => tmp
			//auto hst = gpu_get_col(hs, n_hidden, N + 1, t + 1);
			auto hst_hst = gpu_mul_elemwise(cache_s->tmp_d_vec, cache_s->tmp_d_vec, n_hidden);
			device_vector<float> ones_hs_hs(n_hidden);
			device_vector<float> ones_vec(n_hidden, 1.f);
			thrust::transform(ones_vec.begin(), ones_vec.end(), 
				hst_hst.begin(), 
				ones_hs_hs.begin(),
				thrust::minus<float>());
			dhraw = gpu_mul_elemwise(ones_hs_hs, dh, n_hidden);

			// dbh += dhraw;
			d_p_s->dbh = gpu_add(d_p_s->dbh, dhraw, n_hidden);

			// 目前的 W 都没加 惩罚
			// dWxh += dhraw * xs[t].t(); // (50, 17)
			gpu_get_col(xs, M, N, t, cache_s->tmp_d_vec); // xs[t] => tmp
			//auto dWxh_right = 
				gpu_mmul(handle, 
				dhraw, // (50, 1)
				cache_s->tmp_d_vec, // xs[t]: (17)
				n_hidden, 1, 
				n_features, cache_s->W_tmp1); // set xs[t]: (1, 17)
				d_p_s->dWxh = gpu_add(d_p_s->dWxh, cache_s->W_tmp1, n_hidden*n_features);

			// dWhh += dhraw * hs[t - 1].t(); // (50, 50)
			gpu_get_col(hs, n_hidden, N + 1, t + 1 - 1, cache_s->tmp_d_vec); // hs[t-1] => tmp
			//auto dWhh_right = 
				gpu_mmul(handle, 
				dhraw, // (50, 1)
				cache_s->tmp_d_vec, // (50)
				n_hidden, 1,
				n_hidden, cache_s->W_tmp1); // set (1,50)
				d_p_s->dWhh =
				gpu_add(d_p_s->dWhh, cache_s->W_tmp1, n_hidden*n_hidden);

			// dhnext = Whh.t() * dhraw;
			//dhnext = 
				gpu_mv(handle, p_s->Whh, dhraw, n_hidden, n_hidden, cache_s->W_tmp1, true);
				dhnext = cache_s->W_tmp1;

		}
		else
		{
			// dh = dhnext;
			dh = dhnext;

			// dhraw = (1 - hs[t] % hs[t]) % dh; // mul elemwise
			gpu_get_col(hs, n_hidden, N + 1, t + 1, cache_s->tmp_d_vec); // hs[t] => tmp
			//auto hst = gpu_get_col(hs, n_hidden, N + 1, t + 1);
			auto hst_hst = gpu_mul_elemwise(cache_s->tmp_d_vec, cache_s->tmp_d_vec, n_hidden);
			device_vector<float> ones_hs_hs(n_hidden);
			device_vector<float> ones_vec(n_hidden, 1.f);
			thrust::transform(ones_vec.begin(), ones_vec.end(),
				hst_hst.begin(),
				ones_hs_hs.begin(),
				thrust::minus<float>());
			dhraw = gpu_mul_elemwise(ones_hs_hs, dh, n_hidden);

			// dbh += dhraw;
			d_p_s->dbh =
				gpu_add(d_p_s->dbh, dhraw, n_hidden);

			// dWxh += dhraw * xs[t].t(); // (50, 17)
			gpu_get_col(xs, M, N, t, cache_s->tmp_d_vec); // xs[t] => tmp
			//auto dWxh_right = 
				gpu_mmul(handle,
				dhraw, // (50, 1)
				cache_s->tmp_d_vec, // xs[t]: (17)
				n_hidden, 1,
				n_features, cache_s->W_tmp1); // set xs[t]: (1, 17)
				d_p_s->dWxh =
				gpu_add(d_p_s->dWxh, cache_s->W_tmp1, n_hidden*n_features);

			// dWhh += dhraw * hs[t - 1].t(); // (50, 50)
			gpu_get_col(hs, n_hidden, N + 1, t + 1 - 1, cache_s->tmp_d_vec); // hs[t-1] => tmp
			//auto dWhh_right = 
				gpu_mmul(handle,
				dhraw, // (50, 1)
					cache_s->tmp_d_vec, // (50)
				n_hidden, 1,
				n_hidden, cache_s->W_tmp1); // set (1,50)
				d_p_s->dWhh =
				//W_tmp2 = dWhh;
				gpu_add(d_p_s->dWhh, cache_s->W_tmp1, n_hidden*n_hidden);

			// dhnext = Whh.t() * dhraw;
			//dhnext = 
				gpu_mv(handle, p_s->Whh, dhraw, n_hidden, n_hidden, cache_s->W_tmp1,true);
				dhnext = cache_s->W_tmp1;
		}
	}

	// clip: dWxh, dWhh, dWhy, dbh, dby
	gpu_clip(d_p_s->dWxh, -5.f, 5.f); 
	gpu_clip(d_p_s->dWhh, -5.f, 5.f);
	gpu_clip(d_p_s->dWhy, -5.f, 5.f);
	gpu_clip(d_p_s->dbh, -5.f, 5.f); // 思考下，是否需要 对 b 进行clip
	gpu_clip(d_p_s->dby, -5.f, 5.f);

	// --------------------------------------
	return;
}

device_vector<float> score2onehot(float score, int& idx1_targets, 
	int n_output_classes,
	float score_min,
	float score_max)
{
	float part = 1.0f / n_output_classes;

	float pos = (score - score_min) /
		(score_max - score_min + powf(10, -8));

	idx1_targets = floorf(pos / part);

	// d_vec
	device_vector<float> d_vec(n_output_classes, 0.0f);
	d_vec[idx1_targets] = 1.0f;

	// create array with 0 or 1
	//float* arr;
	//arr = (float*)malloc(n_output_classes * sizeof(float));
	//for (int i = 0; i < n_output_classes; i++)
	//{
	//	arr[i] = 0.0f;
	//}
	//arr[pos_idx] = 1.0f;

	//// stores info of arr into marr
	//MyArray* marr = new MyArray();
	//marr->arr = arr;
	//marr->n_rows_origin = n_output_classes;
	//marr->size = n_output_classes;

	return d_vec;
}

void sgd(device_vector<float>& P, device_vector<float> dP, 
	int size, float alpha)
{
	// P = P - alpha * dP
	//auto P_right = gpu_scal(dP, size, -alpha);
	P = gpu_add(P, 
			gpu_scal(dP, size, -alpha), // right
			size);
}

void test_gpu_fns_CudaUtils()
{
	// ------------------------------------
	cublasHandle_t handle;
	cublasCreate(&handle);

	/*
	cout << "test gpu_fns" << endl;
	const int A_M = 4;
	const int A_N = 2;
	const int B_M = 2;
	const int B_N = 3;

	device_vector<float> d_A(A_M * A_N); // device_vector会自动释放资源
	device_vector<float> d_B(B_M * B_N);
	device_vector<float> d_C; // d_C 是动态可变的

	cout << "A.size: " << d_A.size()
		<< "\tB.size: " << d_B.size()
		<< "\tC.size: " << d_C.size() << endl;

	// init values
	for (int i = 0; i < A_M*A_N; i++)
	{
		d_A[i] = float(i);
	}
	for (int i = 0; i < B_M*B_N; i++)
	{
		d_B[i] = i/2.0f;
	}

	d_C = gpu_mmul(handle, d_A, d_B, A_M, A_N, B_N);
	cout << "after mmul, C.size: " << d_C.size() << endl;
	printToHost(d_A, A_M, A_N, "A");
	printToHost(d_B, B_M, B_N, "B");
	printToHost(d_C, A_M, B_N, "C = A*B");

	// G = D*E*F ，将gpu_fns串起来使用
	device_vector<float> D(4 * 2);
	device_vector<float> E(2 * 3);
	device_vector<float> F(3 * 5);
	device_vector<float> G;
	for (int i = 0; i < 8; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(i/2);
	}
	for (int i = 0; i < 15; i++)
	{
		F[i] = float(1);
	}

	G = gpu_mmul(handle,
			gpu_mmul(handle, D, E, 4, 2, 3), 
			F, 4, 3, 5);

	printToHost(D, 4, 2, "D");
	printToHost(E, 2, 3, "E");
	printToHost(F, 3, 5, "F");
	printToHost(G, 4, 5, "G = D*E*F");
	*/

	// ============== Cuda4RNN test =============

	// ------------- W b :)----------------
	/*
	cout << "init val" << endl;
	printToHost(Cuda4RNN::Wxh, 50, 17, "Wxh");
	printToHost(Cuda4RNN::Whh, 50, 50, "Whh");
	printToHost(Cuda4RNN::Why, 10, 50, "Why");
	printToHost(Cuda4RNN::bh, 50, 1, "bh");
	printToHost(Cuda4RNN::by, 10, 1, "by");*/



	// ========== 使用4个一维dvec将场景数据存储 ============
	/*
	// 测试 vec嵌套vec，以方便对Scanarios数据存储 ！！！！！
	// 但thrust vec不支持多重嵌套vec。若欲处理高维数据，应该将 高维 => 一维

	// "sce_id_score", "sce_data", "sce_data_mn", "sce_data_idx_begin"

	// "sce_id_score"
	const int num_sce = 3; // num of scenarios
	// sce0: 5*3, sce1: 4*3, sce2: 2*3
	const int sce0_rows = 5, sce0_cols = 3;
	const int sce1_rows = 4, sce1_cols = 3;
	const int sce2_rows = 2, sce2_cols = 3;
	
	thrust::device_vector<float> sce_id_score(num_sce*2); // 有些id是浮点数
	for (int i = 0; i < num_sce*2; i+=2)
	{
		sce_id_score[i] = 100.0f + i;
		sce_id_score[i+1] = rand() % 40 / 10.f + 6.f;
	}

	// "sce_data" saves all data of sceanrios, not including id,score
	const int total_size = sce0_rows * sce0_cols + sce1_rows * sce1_cols + sce2_rows * sce2_cols;
	thrust::device_vector<float> sce_data(total_size);

	// M,N are saved in "sce_data_mn"
	thrust::device_vector<float> sce_data_mn;
	sce_data_mn.push_back(5);
	sce_data_mn.push_back(3);
	sce_data_mn.push_back(4);
	sce_data_mn.push_back(3);
	sce_data_mn.push_back(2);
	sce_data_mn.push_back(3);

	// use M,N to generate "sce_data_idx_begin" 
	//		that saves index of beginning point in "sce"
	thrust::device_vector<float> sce_data_idx_begin(num_sce, 0.f);
	int idx_cumsum = 0;
	for (int i = 0; i < num_sce-1; i++)
	{
		 idx_cumsum += sce_data_mn[i * 2] * sce_data_mn[i * 2 + 1];
		 sce_data_idx_begin[i+1] = idx_cumsum;
	}

	// how to use? 
	// from "sce_data_mn" => "sce_data_idx_begin" to get data of a sce
	printToHost(sce_id_score, 2, num_sce, "sce_id_score");
	printToHost(sce_data_mn, 2, num_sce, "sce_data_mn, row0=m, row1=n");
	printToHost(sce_data_idx_begin, 1, num_sce, "sce_data_idx_begin");

	// init sce_data
	for (int i = 0; i < sce_data_idx_begin.size(); i++)
	{
		// 第 i 个场景
		int idx_begin = sce_data_idx_begin[i];
		int size = sce_data_mn[i*2 + 0] * sce_data_mn[i*2 + 1];

		for (int j = idx_begin; j < idx_begin+size; j++)
		{
			sce_data[j] = rand() % 10 / 10.f + i;
		}
	}

	printToHost(sce_data, total_size, 1, "sce_data");

	// 按照场景 print data
	device_vector<float> sce0_data(sce0_rows*sce0_cols);
	device_vector<float> sce1_data(sce1_rows*sce1_cols);
	device_vector<float> sce2_data(sce2_rows*sce2_cols);

	thrust::copy(sce_data.begin() + sce_data_idx_begin[0], sce_data.begin() + sce_data_idx_begin[1], sce0_data.begin()); // copy中 左闭右开
	thrust::copy(sce_data.begin() + sce_data_idx_begin[1], sce_data.begin() + sce_data_idx_begin[2], sce1_data.begin());
	thrust::copy(sce_data.begin() + sce_data_idx_begin[2], sce_data.begin() + total_size,			 sce2_data.begin());

	printToHost(sce0_data, sce0_rows, sce0_cols, "sce0 data");
	printToHost(sce1_data, sce1_rows, sce1_cols, "sce1 data");
	printToHost(sce2_data, sce2_rows, sce2_cols, "sce2 data");

	*/
	// ============ 使用 封装的类 CudaUtils ===============
	// G = D*E*F ，将gpu_fns串起来使用
	/*int dm = 1, dn = 2;
	int em = 2, en = 3;
	int fm = 3, fn = 1;

	device_vector<float> D(dm * dn);
	device_vector<float> E(em * en);
	device_vector<float> F(fm * fn);
	device_vector<float> G;
	for (int i = 0; i < 2; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(1);
	}
	for (int i = 0; i < 3; i++)
	{
		F[i] = float(1);
	}
	printToHost(D, dm, dn, "D");
	printToHost(E, em, en, "E");
	printToHost(F, fm, fn, "F");*/

	// ----------- mul ----------------
	/*
	cout << "by gpu_fns: " << endl;
	G = gpu_mmul(handle, 
			gpu_mmul(handle, 
				D, E, 1, 2, 3), 
			F, 1, 3, 1); // G = D*E*F
	//G = gpu_mmul(handle, D, E, 4, 2, 3);
	printToHost(D, 1, 2, "D");
	printToHost(E, 2, 3, "E");
	printToHost(F, 3, 1, "F");
	printToHost(G, 1, 1, "G = D*E");

	cout << "by CudaUtils \n";
	CudaUtils* cu = new CudaUtils(handle); 
	// 思考下，有问题，不能只在cpu上分配内存，否则结果是错的 
	// 因为此类不会真正在gpu上执行，仅仅在cpu上调用gpu fn，所以没问题

	// 注意链式计算时，并不会区分 +*优先级，只会顺序往后
	cu->box(D, 1, 2)->mmul(E, 2, 3)->mmul(F, 3, 1);
	auto res = cu->getResDevVec();
	int resM = cu->getResM(); cout << "resM: " << resM << endl;
	int resN = cu->getResN(); cout << "resN: " << resN << endl;
	printToHost(res, resM, resN, "G, by CudaUtils");
	*/
	// ----------- add ------------------
	/*
	cout << "add by gpu_add" << endl;
	G = gpu_add(handle, E, E, em*en);
	printToHost(G, em, en, "E+E");

	cout << "add by CudaUtils\n";
	CudaUtils* cuE = new CudaUtils(handle, E, em, en); // 思考下handle的作用
	// 注意此处在cpu上new内存，因为传入gpu_fns的参数都是gpu上分配的，所以没问题
	auto res = cuE->add(E, em, en)->getResDevVec();
	//auto res = cu->getResDevVec();
	printToHost(res, em, en, "add by CudaUtils");
	*/
	// ------------ z = (A*x + B*y) 混合 ------------
	/*
	int am = 3, an = 2;
	int bm = 3, bn = 2;
	device_vector<float> A(am*an);
	device_vector<float> x(an);
	device_vector<float> B(bm*bn);
	device_vector<float> y(bn);
	device_vector<float> z;
	//gpu_fill_rand(raw_pointer_cast(&A[0]), am, an);
	A = gpu_fill_rand(A, am, an, 1.f, 3.f); // 必须return A，否则不改变
	thrust::fill(x.begin(), x.end(), 1.0f);
	B = gpu_fill_rand(B, bm, bn);
	thrust::fill(y.begin(), y.end(), 1.0f);

	printToHost(A, am, an, "A");
	printToHost(x, an, 1, "x");
	printToHost(B, bm, bn, "B");
	printToHost(y, bn, 1, "y");

	// use gpu_fns
	z = gpu_add(handle, gpu_mv(handle, A, x, am, an),
		gpu_mv(handle, B, y, bm, bn), am);
	printToHost(z, am, 1, "gpu_fns, A*x + B*y");

	// use CudaUtils
	CudaUtils* cuA = new CudaUtils(handle, A, am, an);
	CudaUtils* cuB = new CudaUtils(handle, B, bm, bn);
	
	cuA->mv(x) // A*x  => (am, 1)
		->add( // +
	cuB->mv(y)); // B*y => (bm, 1)

	z = cuA->getResDevVec();
	printToHost(z, am, 1, "CudaUtils, A*x + B*y");

	*/

	// --- Mv, M'v ----
	/*
	device_vector<float> M(3 * 2, 2.f);
	device_vector<float> v(2, 1.1f);
	device_vector<float> v2(3, 1.1f);
	device_vector<float> y;
	device_vector<float> y2;
	
	printToHost(M, 3, 2, "M");
	printToHost(v, 2, 1, "v");

	y = new CudaUtils(handle, M, 3, 2)->mv(v)->getResDevVec();
	printToHost(y, 3, 1, "Mv");

	y2 = new CudaUtils(handle, M, 3, 2)->mv(v2, true)->getResDevVec(); // 注意：cuM已经不是最开始的值了，因为你上面已经把cuM更改了
	printToHost(y2, 2, 1, "M'v");
	*/

	// ---------- mul_elemwise, tanh, scal --------
	/*
	device_vector<float> A(5* 3);
	device_vector<float> B(5* 3);
	gpu_fill_rand(A, 5, 3, -1.f, 1.f, 1); // last arg is seed
	gpu_fill_rand(B, 5, 3, -1.f, 1.f, 11);
	printToHost(A, 5, 3, "A");
	printToHost(B, 5, 3, "B");

	auto C = new CudaUtils(handle, A, 5, 3)->mul_elemwise(B)->getResDevVec();
	printToHost(C, 5, 3, "A .* B");

	auto D = new CudaUtils(handle, A, 5, 3)->tanh()->getResDevVec();
	printToHost(D, 5, 3, "tanh(A)");

	auto E = new CudaUtils(handle, A, 5, 3)->scal(0.5)->getResDevVec();
	printToHost(E, 5, 3, "0.5 * A");*/
	

	// ---------- getRow() demo ------------
	/*
	device_vector<float> A(5*3); // 使用device_vector初始化时，传入size，而非dim
	A = gpu_fill_rand(A, 5, 3, -1.f, 2.f);
	printToHost(A, 5, 3, "A");

	for (int i = 0; i < 5; i++)
	{
		device_vector<float> Arow_i =
			new CudaUtils(handle, A, 5, 3)->getRow(i)->getResDevVec();

		string title = "A row";
		stringstream ss;
		ss << title << i;
		printToHost(Arow_i, 1, 3, ss.str());
	}*/


	// ----------- gpu_tanh --------------
	/*device_vector<float> A(10);
	A = gpu_fill_rand(A, 10, 1, 0.f, 10.f);
	printToHost(A, 1, 10, "A");

	device_vector<float> R = gpu_tanh(A, 10);
	printToHost(R, 1, 10, "tanh(A)");*/

	// ----------- gpu_mul_elemwise ------------
	/*device_vector<float> x(10, 1.1f);
	device_vector<float> y(10, 2.f);
	device_vector<float> z;
	z = gpu_mul_elemwise(x, y, 10);
	printToHost(z, 1, 10, "x .* y");*/

	// ---------- gpu_set_row -------------
	/*
	int M = 5;
	int N = 3;
	device_vector<float> A(M*N);
	device_vector<float> values(1 * 3, 1.0f);
	gpu_fill_rand(A, M, N);
	printToHost(A, M, N, "A");

	gpu_set_row(A, M, N, 1, values, false);
	printToHost(A, M, N, "fill A.row(1) with 1.0");*/

	// ---------- gpu_max_index, gpu_max_value :) ----------
	/*device_vector<float> vec(10);
	gpu_fill_rand(vec, 1, 10);
	printToHost(vec, 1, 10, "vec");

	float mval = gpu_max_value(vec);
	int midx = gpu_max_index(vec);
	
	cout << "max val: " << mval << ", max idx: " << midx << endl;*/
	

	// --------- gpu_get_col, gpu_set_col  :) -----------------
	/*
	device_vector<float> A(4 * 3);
	for (int i = 0; i < 12; i++)
	{
		A[i] = float(i);
	}
	printToHost(A, 4, 3, "A");

	// get col
	for (int col = 0; col < 3; col++)
	{
		auto Acol = gpu_get_col(A, 4, 3, col);
		stringstream ss;
		ss << "Acol" << col;
		printToHost(Acol, 4, 1, ss.str());
	}

	// set col
	int setColIdx = 1;
	device_vector<float> values(4 * 1, 1.2f);
	gpu_set_col(A, 4, 3, setColIdx, values);
	printToHost(A, 4, 3, "set Acol1=1.2");
	*/
	
	// -------- gpu_softmax :) -------------
	/*
	device_vector<float> d_x(10);
	gpu_fill_rand(d_x, 10, 1, -10.f, 10.f);
	printToHost(d_x, 1, 10, "x");

	device_vector<float> soft = gpu_softmax(d_x, 10);
	printToHost(soft, 1, 10, "softmax(x)");*/
	
	// ------- gpu_clip :) ------------
/*
	device_vector<float> vec(10);
	gpu_fill_rand(vec, 1, 10, -2.f, 2.f);
	printToHost(vec, 1, 10, "vec");

	gpu_clip(vec, -1, 1);
	printToHost(vec, 1, 10, "clip -1 1");
*/

	// -------- 尝试用 类 封装gpu_fns，有了类后可以做, e.g. gpu.mmul(A, B).add(gpu.mmul(C, D)); => A*B + C*D---------
	/*
	// G = D*E*F ，将gpu_fns串起来使用

	cout << "test CudaUtils" << endl;
	CudaUtils cu = CudaUtils(handle);

	device_vector<float> D(4 * 2);
	device_vector<float> E(2 * 3);
	device_vector<float> F(3 * 5);
	device_vector<float> G;

	for (int i = 0; i < 8; i++)
	{
		D[i] = float(i);
	}
	for (int i = 0; i < 6; i++)
	{
		E[i] = float(i / 2.0f);
	}
	for (int i = 0; i < 15; i++)
	{
		F[i] = float(1);
	}
	printToHost(D, 4, 2, "D pre");

	cu.mmul(D, E, 4, 2, 3);
	*/
	

	// ------------- device_vector<T> 测试泛型 ---------------
	/*
	cout << "test vec<int> 保存vec的地址" << endl; 
	device_vector<float*> d_pool;
	cout << "d_pool.size(): " << d_pool.size() << endl;

	d_pool.push_back(raw_pointer_cast(&D[0]));
	cout << "d_pool.size(): " << d_pool.size() << endl;

	float* d = d_pool[0];
	//device_vector<float> d_vec(d, d + 8); // 必须用device_vector还原
	printToHost(d, 4, 2, "d_vec");
	*/

	// ---------- permutation_iterator -------------
	/*thrust::device_vector<int> rowLoc(2);
	rowLoc[0] = 0;
	rowLoc[1] = 3;

	thrust::device_vector<float> d_vec(6);
	d_vec = gpu_fill_rand(d_vec, 6, 1);
	printToHost(d_vec, 3, 2, "d_vec:");

	thrust::device_vector<float> targets(2);
	thrust::transform(thrust::make_permutation_iterator(d_vec.begin(),
		rowLoc.begin()),
			thrust::make_permutation_iterator(d_vec.begin(),
		rowLoc.end()),
		targets.begin(), thrust::identity<float>());
	printToHost(targets, 1, 2, "row0");*/

	cublasDestroy(handle);
}

