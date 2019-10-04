#include "RNN.h"

std::mutex RNN::mtx;

int		RNN::total_epoches = 501;
double	RNN::alpha = 0.1;
double	RNN::score_max = 9.4; // start: 8.9, gearShiftUp: 9.4
double	RNN::score_min = 4.9; // start: 6.0, gearShiftUp: 4.9
int		RNN::n_features = 20;
int		RNN::n_hidden = 30;
int		RNN::n_output_classes = 10;

int RNN::tmp = 11; // 试验是否可以 实例操作静态对象

const int D = RNN::n_features;
const int H = RNN::n_hidden;
const int Z = RNN::n_features + RNN::n_hidden;
const int Classes = RNN::n_output_classes;

mat RNN::Wf = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden forget
mat RNN::Wi = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden input
mat RNN::Wc = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden candidate cell
mat RNN::Wo = arma::randn(Z, H) / sqrt(Z / 2.); // W 包含h x, hidden output
mat RNN::Wy = arma::randn(H, Classes) / sqrt(Classes / 2.); //  y

mat RNN::bf = arma::zeros(1, H); // 可看出来，默认 行向量
mat RNN::bi = arma::zeros(1, H);
mat RNN::bc = arma::zeros(1, H);
mat RNN::bo = arma::zeros(1, H);
mat RNN::by = arma::zeros(1, Classes);


RNN::RNN()
{

}


RNN::~RNN()
{
}


void RNN::trainMultiThread(vector<SceStruct> listStructTrain, 
	AOptimizer* opt, int n_threads, double lambda)
{
	// init memory，用于在 Adagrad中，计算L2范式，平方 求和 开根
	mat mdWf = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
	mat mdWi = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
	mat mdWc = arma::zeros<mat>(Wc.n_rows, Wc.n_cols);
	mat mdWo = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
	mat mdWy = arma::zeros<mat>(Wy.n_rows, Wy.n_cols);
	mat mdbf = arma::zeros<mat>(bf.n_rows, bf.n_cols);
	mat mdbi = arma::zeros<mat>(bi.n_rows, bi.n_cols);
	mat mdbc = arma::zeros<mat>(bc.n_rows, bc.n_cols);
	mat mdbo = arma::zeros<mat>(bo.n_rows, bo.n_cols);
	mat mdby = arma::zeros<mat>(by.n_rows, by.n_cols);

	//mat hprev = zeros<mat>(this->n_hidden, 1); // init hprev
	mat hprev, cprev;
	vector<double> lossAllVec; // 记录所有loss
	vector<double> loss_one_epoch;
	vector<double> loss_mean_each_epoch; // 只记录每个epoch中 loss mean
	vector<double> true_false; // 存储一个epoch中，所有场景的正确率
	vector<double> accuracy_each_epoch;
	vector<double> log_target; // 记录所有epoches和所有场景的 target
	vector<double> log_prediction; // 与log_target对应，记录所有的模型 prediction

	vector<future<map<string, mat>>> vec_future; // 存储多线程的vector
	mat dWfSum, dbfSum;
	mat dWiSum, dbiSum;
	mat dWcSum, dbcSum;
	mat dWoSum, dboSum;
	mat dWySum, dbySum;
	mat loss;

	/*
		外层循环: 遍历所有场景为一个 epoch。
		total_steps: 是 epoch 数量
	*/
	for (int i = 0; i < this->total_epoches; i++)
	{

		vector<SceStruct>::iterator ascenario;
		true_false.clear(); // 清空
		loss_one_epoch.clear();

		/*
			内层循环，
			遍历 allScenarios 中所有场景，拿到 matData & score 进行训练模型
		*/
		ascenario = listStructTrain.begin(); // 每个epoch的开始，都要重置 ascenario
		while (ascenario != listStructTrain.end())
		{
			vec_future.clear(); // 每次多线程计算后，clear
			for (int t = 0; t < n_threads && ascenario != listStructTrain.end(); t++, ascenario++)
			{
				arma::mat matData = ascenario->matDataZScore; // 使用 zscore norm 的matData
				double score = ascenario->score;

				// lstm中默认 行向量
				hprev = arma::zeros<mat>(1, this->n_hidden); // for each scenario, hprev is 0
				cprev = arma::zeros<mat>(1, this->n_hidden); // for each scenario, hprev is 0
				vec_future.push_back(async(RNN::lossFun, 
					matData, score, lambda, hprev, cprev, std::ref(true_false), std::ref(log_target), std::ref(log_prediction)));
			}

			// 清零。在多个线程计算之后，get结果 dW db
			dWfSum = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
			dWiSum = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
			dWcSum = arma::zeros<mat>(Wc.n_rows, Wc.n_cols);
			dWoSum = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
			dWySum = arma::zeros<mat>(Wy.n_rows, Wy.n_cols);
			dbfSum = arma::zeros<mat>(bf.n_rows, bf.n_cols);
			dbiSum = arma::zeros<mat>(bi.n_rows, bi.n_cols);
			dbcSum = arma::zeros<mat>(bc.n_rows, bc.n_cols);
			dboSum = arma::zeros<mat>(bo.n_rows, bo.n_cols);
			dbySum = arma::zeros<mat>(by.n_rows, by.n_cols);

			for (int s = 0; s < vec_future.size(); s++)
			{
				map<string, mat> mymap = vec_future[s].get(); // get 是阻塞型的

				loss = mymap["loss"];
				dWfSum += mymap["dWf"];
				dWiSum += mymap["dWi"];
				dWcSum += mymap["dWc"];
				dWoSum += mymap["dWo"];
				dWySum += mymap["dWy"];
				dbfSum += mymap["dbf"];
				dbiSum += mymap["dbi"];
				dbcSum += mymap["dbc"];
				dboSum += mymap["dbo"];
				dbySum += mymap["dby"];

				// store in vec
				lossAllVec.push_back(loss(0, 0));
				loss_one_epoch.push_back(loss(0, 0));

				// print loss
				if (i % 100 == 0)
				{
					// 把此epoch中的每个场景的 loss 打印
					cout << "				loss: " << loss(0, 0) << endl;
				}
			}

			// clip
			double maxVal, minVal;
			maxVal = 5.0;
			minVal = -5.0;
			clip(dWfSum, maxVal, minVal);
			clip(dWiSum, maxVal, minVal);
			clip(dWcSum, maxVal, minVal);
			clip(dWoSum, maxVal, minVal);
			clip(dWySum, maxVal, minVal);
			clip(dbfSum, maxVal, minVal);
			clip(dbiSum, maxVal, minVal);
			clip(dbcSum, maxVal, minVal);
			clip(dboSum, maxVal, minVal);
			clip(dbySum, maxVal, minVal);
			

			// update params。把每个场景看做一个样本的话，则是sgd。
			opt->optimize(this->Wf, this->alpha, dWfSum, mdWf, i);
			opt->optimize(this->Wi, this->alpha, dWiSum, mdWi, i);
			opt->optimize(this->Wc, this->alpha, dWcSum, mdWc, i);
			opt->optimize(this->Wo, this->alpha, dWoSum, mdWo, i);
			opt->optimize(this->Wy, this->alpha, dWySum, mdWy, i);
			opt->optimize(this->bf, this->alpha, dbfSum, mdbf, i);
			opt->optimize(this->bi, this->alpha, dbiSum, mdbi, i);
			opt->optimize(this->bc, this->alpha, dbcSum, mdbc, i);
			opt->optimize(this->bo, this->alpha, dboSum, mdbo, i);
			opt->optimize(this->by, this->alpha, dbySum, mdby, i);
		}

		// lossVec mean, accuracy
		double loss_this_epoch = MyLib<double>::mean_vector(loss_one_epoch); // 记录每个epoch的 loss
		double accu_this_epoch = MyLib<double>::mean_vector(true_false); // 记录每个epoch的 accu
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 10 == 0)
		{
			cout << "lambda: " << lambda << ", epoch: " << i 
				<< ", loss_mean_this_epoch: " << loss_this_epoch
				<< ", accu_this_epoch: " << accu_this_epoch << "\n";
		}

	}

	// vector to mat
	this->log_target_prediction = Mat<short>(log_target.size(), 3);
	for (int i = 0; i < log_target_prediction.n_rows; i++)
	{
		this->log_target_prediction(i, 0) = (short)log_target[i];
		this->log_target_prediction(i, 1) = (short)log_prediction[i];
		if (log_target[i] == log_prediction[i])
		{
			this->log_target_prediction(i, 2) = (short)1;
		}
		else
		{
			this->log_target_prediction(i, 2) = (short)0;
		}
	}

	this->lossAllVec = lossAllVec;
	this->loss_mean_each_epoch = loss_mean_each_epoch;
	this->accuracy_each_epoch = accuracy_each_epoch;
}

/*
	once FORWARD & once BACKWARD

	to get dW db from training of one scenario
*/
map<string, mat> RNN::lossFun(mat inputs, 
	double score, double lambda, mat hprev, mat cprev,
	vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction)
{
	// 注：参数要在这个函数体外部提前初始化。
	int idx1 = -1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs, ys, ps;
	/*
		h_fs: hidden forget state
		h_is: hidden input state
		h_os: hidden output state
		h_cs: hidden candidate cell state

		cs: cell
		hs: 
		ys: 
		ps: softmax(ys)
	*/
	hs[-1] = hprev; // 默认是 深拷贝
	cs[-1] = cprev; // LSTM有2个隐藏states
	mat loss = arma::zeros<mat>(1, 1);

	// ===================== forward pass =========================

	// inputs.n_rows 是 samples的数量，inputs.n_cols 是 features数量
	for (int t = 0; t < inputs.n_rows; t++)
	{
		xs[t] = inputs.row(t); // (1,d)
		X[t] = arma::join_horiz(hs[t - 1], xs[t]); 
		// X: concat [h_old, curx] (1,h+d)
		
		//hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);

		h_fs[t] = RNN::sigmoid(X[t] * Wf + bf); // (1,h+d)(h+d,h) = (1,h)
		h_is[t] = RNN::sigmoid(X[t] * Wi + bi);
		h_os[t] = RNN::sigmoid(X[t] * Wo + bo );
		h_cs[t] = arma::tanh(X[t] * Wc  + bc);

		cs[t] = h_fs[t] % cs[t-1] + h_is[t] % h_cs[t]; // (1,h)
		hs[t] = h_os[t] % arma::tanh(cs[t]); // (1,h)

		if (t == inputs.n_rows - 1)
		{
			ys[t] = hs[t] * Wy  + by; // (1,h)(h,y) = (1,y)
			ps[t] = RNN::softmax(ys[t]);

			loss += -log( ps[t](idx1) );

			//cout << "loss: " << loss << endl;

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 1); // index_prediction

			mtx.lock();
			log_target.push_back(idx1);
			log_prediction.push_back(idx_max_ps(0));
			if (idx1 == idx_max_ps(0))
			{
				true_false.push_back(1);
			}
			else
			{
				true_false.push_back(0);
			}
			mtx.unlock();
		}
	}

	// =============== BPTT ============================
	mat dWf, dbf;
	mat dWi, dbi;
	mat dWc, dbc;
	mat dWo, dbo;
	mat dWy, dby; // dhnext

	dWf = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
	dWi = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
	dWc = arma::zeros<mat>(Wc.n_rows, Wc.n_cols);
	dWo = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
	dWy = arma::zeros<mat>(Wy.n_rows, Wy.n_cols);
	dbf = arma::zeros<mat>(bf.n_rows, bf.n_cols);
	dbi = arma::zeros<mat>(bi.n_rows, bi.n_cols);
	dbc = arma::zeros<mat>(bc.n_rows, bc.n_cols);
	dbo = arma::zeros<mat>(bo.n_rows, bo.n_cols);
	dby = arma::zeros<mat>(by.n_rows, by.n_cols);

	// dnext init
	mat dhnext = arma::zeros<mat>(hs[0].n_rows, hs[0].n_cols);
	mat dcnext = arma::zeros<mat>(cs[0].n_rows, cs[0].n_cols);

	mat dy, dh, dho, dc, dhf, dhi, dhc;
	mat dXf, dXi, dXo, dXc, dX;
	for (int t = (int)inputs.n_rows - 1; t >= 0; t--)
	{
		if (t == inputs.n_rows - 1)
		{
			// softmax loss gradient
			dy = ps[t];
			dy[idx1] -= 1;

			// hidden to output gradient
			dWy += hs[t].t() * dy + lambda * Wy;
			dby += dy;
			


			dh = dy * Wy.t() + dhnext; // adding dhnext

			// gradient for ho in h = ho * tanh(c)
			dho = arma::tanh(cs[t]) % dh;
			dho = MyLib<mat>::dsigmoid(h_os[t]) % dho;

			// gradient for c in h = ho * tanh(c), adding dcnext
			dc = h_os[t] % dh % MyLib<mat>::dtanh(cs[t]);
			dc = dc + dcnext;

			// gradient for hf in c = hf * c_old + hi * hc
			dhf = cs[t-1] % dc; // c_old => cs[t-1]
			dhf = MyLib<mat>::dsigmoid(h_fs[t]) % dhf;

			// gradient for hi in c = hf * c_old + hi * hc
			dhi = h_cs[t] % dc;
			dhi = MyLib<mat>::dsigmoid(h_is[t]) % dhi;

			// gradient for hc in c = hf * c_old + hi * hc
			dhc = h_is[t] % dc;
			dhc = MyLib<mat>::dtanh(h_cs[t]) % dhc;

			// gate gradinets
			dWf += X[t].t() * dhf +  lambda * Wf;
			dbf += dhf;
			dXf = dhf * Wf.t();

			dWi += X[t].t() * dhi + lambda * Wi;
			dbi += dhi;
			dXi = dhi * Wi.t();

			dWo += X[t].t() * dho + lambda * Wo;
			dbo += dho;
			dXo = dho * Wo.t();

			dWc += X[t].t() * dhc + lambda * Wc;
			dbc += dhc;
			dXc = dhc * Wc.t();

			// as X was used in multiple gates
			dX = dXo + dXc + dXi + dXf;

			// update dhnext dcnext
			dhnext = dX.cols(0, H-1);
			dcnext = h_fs[t] % dc;

		}
		else
		{
			dh = dhnext; // adding dhnext

			// gradient for ho in h = ho * tanh(c)
			dho = arma::tanh(cs[t]) % dh;
			dho = MyLib<mat>::dsigmoid(h_os[t]) % dho;

			// gradient for c in h = ho * tanh(c), adding dcnext
			dc = h_os[t] % dh % MyLib<mat>::dtanh(cs[t]);
			dc = dc + dcnext;

			// gradient for hf in c = hf * c_old + hi * hc
			dhf = cs[t-1] % dc;
			dhf = MyLib<mat>::dsigmoid(h_fs[t]) % dhf;

			// gradient for hi in c = hf * c_old + hi * hc
			dhi = h_cs[t] % dc;
			dhi = MyLib<mat>::dsigmoid(h_is[t]) % dhi;

			// gradient for hc in c = hf * c_old + hi * hc
			dhc = h_is[t] % dc;
			dhc = MyLib<mat>::dtanh(h_cs[t]) % dhc;

			// gate gradinets
			dWf += X[t].t() * dhf;
			dbf += dhf;
			dXf = dhf * Wf.t();

			dWi += X[t].t() * dhi;
			dbi += dhi;
			dXi = dhi * Wi.t();

			dWo += X[t].t() * dho;
			dbo += dho;
			dXo = dho * Wo.t();

			dWc += X[t].t() * dhc;
			dbc += dhc;
			dXc = dhc * Wc.t();

			// as X was used in multiple gates
			dX = dXo + dXc + dXi + dXf;

			// update dhnext dcnext
			dhnext = dX.cols(0, H-1);
			dcnext = h_fs[t] % dc;
		}
	}


	// clip
	/*
	double maxVal, minVal;
	maxVal = 5.0;
	minVal = -5.0;
	clip(dWf, maxVal, minVal);
	clip(dWi, maxVal, minVal);
	clip(dWc, maxVal, minVal);
	clip(dWo, maxVal, minVal);
	clip(dWy, maxVal, minVal);
	clip(dbf, maxVal, minVal);
	clip(dbi, maxVal, minVal);
	clip(dbc, maxVal, minVal);
	clip(dbo, maxVal, minVal);
	clip(dby, maxVal, minVal);
	*/
	
	map<string, mat> mymap;
	mymap["loss"] = loss;
	mymap["dWf"] = dWf;
	mymap["dWi"] = dWi;
	mymap["dWc"] = dWc;
	mymap["dWo"] = dWo;
	mymap["dWy"] = dWy;
	mymap["dbf"] = dbf;
	mymap["dbi"] = dbi;
	mymap["dbc"] = dbc;
	mymap["dbo"] = dbo;
	mymap["dby"] = dby;
	return mymap;
}




void RNN::predictOneScenario(mat Wf, mat Wi, mat Wc, mat Wo, mat Wy,
	mat bf, mat bi, mat bc, mat bo, mat by,
	mat inputs,
	double score, double& loss, int& idx_target, int& idx_prediction)
{
	n_hidden = bf.n_cols; // predict 时修改
	n_output_classes = by.n_cols; // predict时修改

	int idx1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs, ys, ps;

	mat hprev = arma::zeros<mat>(1, RNN::n_hidden);
	mat cprev = arma::zeros<mat>(1, RNN::n_hidden);
	hs[-1] = hprev; // 默认是 深拷贝
	cs[-1] = cprev; // 默认是 深拷贝
	mat loss_tmp = arma::zeros<mat>(1, 1);

	for (int t = 0; t < inputs.n_rows; t++)
	{
		xs[t] = inputs.row(t); // (1,d)
		X[t] = arma::join_horiz(hs[t - 1], xs[t]); // X: concat [h_old, curx] (1,h+d)

		//hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);

		h_fs[t] = RNN::sigmoid(X[t] * Wf + bf); // (1,h+d)(h+d,h) = (1,h)
		h_is[t] = RNN::sigmoid(X[t] * Wi + bi);
		h_os[t] = RNN::sigmoid(X[t] * Wo + bo);
		h_cs[t] = arma::tanh(X[t] * Wc + bc);

		cs[t] = h_fs[t] % cs[t - 1] + h_is[t] % h_cs[t]; // (1,h)
		hs[t] = h_os[t] % arma::tanh(cs[t]); // (1,h)

		if (t == inputs.n_rows - 1)
		{
			ys[t] = hs[t] * Wy + by; // (1,h)(h,y) = (1,y)
			ps[t] = RNN::softmax(ys[t]);

			loss_tmp += -log(ps[t](idx1));

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 1); // index_prediction

			loss = loss_tmp(0, 0);
			idx_target = idx1;
			idx_prediction = idx_max_ps(0);
		}
	}
}

void RNN::saveParams()
{
	this->Wf.save("Wf.txt", file_type::raw_ascii);
	this->Wi.save("Wi.txt", file_type::raw_ascii);
	this->Wc.save("Wc.txt", file_type::raw_ascii);
	this->Wo.save("Wo.txt", file_type::raw_ascii);
	this->Wy.save("Wy.txt", file_type::raw_ascii);
	this->bf.save("bf.txt", file_type::raw_ascii);
	this->bi.save("bi.txt", file_type::raw_ascii);
	this->bc.save("bc.txt", file_type::raw_ascii);
	this->bo.save("bo.txt", file_type::raw_ascii);
	this->by.save("by.txt", file_type::raw_ascii);

	mat loss_all = MyLib<double>::vector2mat(this->lossAllVec);
	mat loss_mean_each_epoch = MyLib<double>::vector2mat(this->loss_mean_each_epoch);
	mat accuracy_each_epoch = MyLib<double>::vector2mat(this->accuracy_each_epoch);

	loss_all.save("loss_all.txt", file_type::raw_ascii);
	loss_mean_each_epoch.save("loss_mean_each_epoch.txt", file_type::raw_ascii);
	accuracy_each_epoch.save("accuracy_each_epoch.txt", file_type::raw_ascii);
	log_target_prediction.save("log_target_prediction.txt", file_type::raw_ascii);

}

mat RNN::sigmoid(arma::mat mx)
{
	// sigmoid = 1 / (1+exp(-x))
	return 1 / (1 + arma::exp(-mx)); 
	// arma 
	/*
		+ - % / elemwise
		* mul
	*/
}

mat RNN::softmax(arma::mat mx)
{
	// softmax = exp(x) / sum(exp(x))
	mat sum_exp = arma::sum(arma::exp(mx), 1); // sum(,0) 列方向sum
	return arma::exp(mx) / (sum_exp(0, 0) /* + pow(10, -8) */ );
}

void RNN::clip(mat& matrix, double maxVal, double minVal)
{
	// transform 遍历每一个元素
	matrix.transform([maxVal, minVal](double val) {
		if (val >= maxVal)
		{
			//cout << "clip, val >= " << maxVal << " ";
			return maxVal;
		}
		else if (val <= minVal)
		{
			//cout << "clip, val <= " << minVal << " ";
			return minVal;
		}
		else
		{
			//cout << "no clip ";
			return val;
		}
	});
}

mat RNN::score2onehot(double score, int& idx1)
{
	if (score > score_max || score < score_min)
	{
		cout << "score > score_max || score < score_min" << endl;
	}


	double part = 1.0 / n_output_classes;

	double pos = (score - score_min)
		/ (score_max - score_min + pow(10, -4)); // 分母加上 1e-8, 避免score是 max时的index越界

	idx1 = std::floor(pos / part);

	mat zs = arma::zeros<mat>(1, n_output_classes); // 默认行向量
	zs(0, idx1) = 1;

	return zs;
}

void RNN::forward(mat inputs, double score)
{
	vector<double> log_target;
	vector<double> log_prediction;
	vector<double> true_false;

	// 注：参数要在这个函数体外部提前初始化。
	int idx1 = -1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs, ys, ps;
	
	mat hprev(1, this->n_hidden, fill::zeros), 
		cprev(1, this->n_hidden, fill::zeros);
	hs[-1] = hprev; // 默认是 深拷贝
	cs[-1] = cprev; // LSTM有2个隐藏states
	mat loss = arma::zeros<mat>(1, 1);
	// inputs.n_rows 是 samples的数量，inputs.n_cols 是 features数量
	for (int t = 0; t < inputs.n_rows; t++)
	{
		xs[t] = inputs.row(t); // (1,d)
		X[t] = arma::join_horiz(hs[t - 1], xs[t]); // X: concat [h_old, curx] (1,h+d)

		//hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);

		h_fs[t] = RNN::sigmoid(X[t] * Wf + bf); // (1,h+d)(h+d,h) = (1,h)
		h_is[t] = RNN::sigmoid(X[t] * Wi + bi);
		h_os[t] = RNN::sigmoid(X[t] * Wo + bo);
		h_cs[t] = arma::tanh(X[t] * Wc + bc);

		cs[t] = h_fs[t] % cs[t - 1] + h_is[t] % h_cs[t]; // (1,h)
		hs[t] = h_os[t] % arma::tanh(cs[t]); // (1,h)

		if (t == inputs.n_rows - 1)
		{
			ys[t] = hs[t] * Wy + by; // (1,h)(h,y) = (1,y)
			ps[t] = RNN::softmax(ys[t]);

			loss += -log(ps[t](idx1));

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 1); // index_prediction

			mtx.lock();
			log_target.push_back(idx1);
			log_prediction.push_back(idx_max_ps(0));
			if (idx1 == idx_max_ps(0))
			{
				true_false.push_back(1);
			}
			else
			{
				true_false.push_back(0);
			}
			mtx.unlock();

		}
	}

}

map<string, arma::mat> RNN::getParams()
{
	map<string, arma::mat> mymap;

	mymap["Wf"] = this->Wf;
	mymap["Wi"] = this->Wi;
	mymap["Wc"] = this->Wc;
	mymap["Wo"] = this->Wo;
	mymap["Wy"] = this->Wy;

	mymap["bf"] = this->bf;
	mymap["bi"] = this->bi;
	mymap["bc"] = this->bc;
	mymap["bo"] = this->bo;
	mymap["by"] = this->by;

	mymap["lossAllVec"] = this->lossAllVec;

	return mymap;
}

void RNN::loadParams()
{
	/*this->Wxh = Wxh;
	this->Whh = Whh;
	this->Why = Why;
	this->bh = bh;
	this->by = by;*/

	this->Wf.load("Wf.txt", file_type::raw_ascii);
	this->Wi.load("Wi.txt", file_type::raw_ascii);
	this->Wc.load("Wc.txt", file_type::raw_ascii);
	this->Wo.load("Wo.txt", file_type::raw_ascii);
	this->Wy.load("Wy.txt", file_type::raw_ascii);
	this->bf.load("bf.txt", file_type::raw_ascii);
	this->bi.load("bi.txt", file_type::raw_ascii);
	this->bc.load("bc.txt", file_type::raw_ascii);
	this->bo.load("bo.txt", file_type::raw_ascii);
	this->by.load("by.txt", file_type::raw_ascii);

	this->tmp = 33; // 可以由实例操作 静态变量

}
