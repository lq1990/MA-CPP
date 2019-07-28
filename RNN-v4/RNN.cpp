#include "RNN.h"

std::mutex RNN::mtx;

double RNN::alpha = 0.1;
int RNN::total_epoches = 201;
double RNN::score_max = 8.9;
double RNN::score_min = 6.0;
int RNN::n_features = 17;
int RNN::n_hidden = 50;
int RNN::n_output_classes = 10;

mat RNN::Wf = arma::randn(n_hidden, n_hidden) * 0.01; // Whf (h,h), ft,it,at,ot,ht(h,1)
mat RNN::Uf = arma::randn(n_hidden, n_features) * 0.01; // Wxf (h,d), xt(d,1) 默认列向量
mat RNN::bf = arma::randn(n_hidden, 1) * 0.01; //     (h,1)

mat RNN::Wi = arma::randn(n_hidden, n_hidden) * 0.01; // Whi (h,h)
mat RNN::Ui = arma::randn(n_hidden, n_features) * 0.01; // Wxi (h,d)
mat RNN::bi = arma::randn(n_hidden, 1) * 0.01; //     (h,1)

mat RNN::Wa = arma::randn(n_hidden, n_hidden) * 0.01; // Wha (h,h)
mat RNN::Ua = arma::randn(n_hidden, n_features) * 0.01; // Wxa (h,d)
mat RNN::ba = arma::randn(n_hidden, 1) * 0.01; //     (h,1)

mat RNN::Wo = arma::randn(n_hidden, n_hidden) * 0.01; // Who (h,h)
mat RNN::Uo = arma::randn(n_hidden, n_features) * 0.01; // Wxo (h,d)
mat RNN::bo = arma::randn(n_hidden, 1) * 0.01; //     (h,1)

mat RNN::Why = arma::randn(n_output_classes, n_hidden) * 0.01; // Why, V (y,h)
mat RNN::by = arma::randn(n_output_classes, 1) * 0.01; // by,   c (y,1)


RNN::RNN()
{

}


RNN::~RNN()
{
}

/*
RNN::RNN(int n_hidden, int n_output_classes = 3,
	double alpha = 0.01, int total_epoches = 501,
	double score_max = 8.9, double score_min = 6.0)
{
	this->n_hidden = n_hidden;
	this->n_output_classes = n_output_classes;
	this->alpha = alpha;
	this->total_epoches = total_epoches;
	this->score_max = score_max;
	this->score_min = score_min;
}*/


void RNN::trainMultiThread(vector<SceStruct> listStructTrain, AOptimizer* opt, int n_threads, double lambda)
{
	// init memory，用于在 Adagrad中，计算L2范式，平方 求和 开根
	mat mdWf = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
	mat mdUf = arma::zeros<mat>(Uf.n_rows, Uf.n_cols);
	mat mdbf = arma::zeros<mat>(bf.n_rows, bf.n_cols);
	mat mdWi = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
	mat mdUi = arma::zeros<mat>(Ui.n_rows, Ui.n_cols);
	mat mdbi = arma::zeros<mat>(bi.n_rows, bi.n_cols);
	mat mdWa = arma::zeros<mat>(Wa.n_rows, Wa.n_cols);
	mat mdUa = arma::zeros<mat>(Ua.n_rows, Ua.n_cols);
	mat mdba = arma::zeros<mat>(ba.n_rows, ba.n_cols);
	mat mdWo = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
	mat mdUo = arma::zeros<mat>(Uo.n_rows, Uo.n_cols);
	mat mdbo = arma::zeros<mat>(bo.n_rows, bo.n_cols);
	mat mdWhy = arma::zeros<mat>(Why.n_rows, Why.n_cols);
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
	mat dWfSum, dUfSum, dbfSum;
	mat dWiSum, dUiSum, dbiSum;
	mat dWaSum, dUaSum, dbaSum;
	mat dWoSum, dUoSum, dboSum;
	mat dWhySum, dbySum;
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

				hprev = arma::zeros<mat>(this->n_hidden, 1); // for each scenario, hprev is 0
				cprev = arma::zeros<mat>(this->n_hidden, 1); // for each scenario, hprev is 0
				vec_future.push_back(async(RNN::lossFun, 
					matData, score, lambda, hprev, cprev, std::ref(true_false), std::ref(log_target), std::ref(log_prediction)));
			}

			// 清零。在多个线程计算之后，get结果 dW db
			dWfSum = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
			dUfSum = arma::zeros<mat>(Uf.n_rows, Uf.n_cols);
			dbfSum = arma::zeros<mat>(bf.n_rows, bf.n_cols);
			dWiSum = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
			dUiSum = arma::zeros<mat>(Ui.n_rows, Ui.n_cols);
			dbiSum = arma::zeros<mat>(bi.n_rows, bi.n_cols);
			dWaSum = arma::zeros<mat>(Wa.n_rows, Wa.n_cols);
			dUaSum = arma::zeros<mat>(Ua.n_rows, Ua.n_cols);
			dbaSum = arma::zeros<mat>(ba.n_rows, ba.n_cols);
			dWoSum = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
			dUoSum = arma::zeros<mat>(Uo.n_rows, Uo.n_cols);
			dboSum = arma::zeros<mat>(bo.n_rows, bo.n_cols);

			for (int s = 0; s < vec_future.size(); s++)
			{
				map<string, mat> mymap = vec_future[s].get(); // get 是阻塞型的

				loss = mymap["loss"];
				dWfSum += mymap["dWf"];
				dUfSum += mymap["dUf"];
				dbfSum += mymap["dbf"];
				dWiSum += mymap["dWi"];
				dUiSum += mymap["dUi"];
				dbiSum += mymap["dbi"];
				dWaSum += mymap["dWa"];
				dUaSum += mymap["dUa"];
				dbaSum += mymap["dba"];
				dWoSum += mymap["dWo"];
				dUoSum += mymap["dUo"];
				dboSum += mymap["dbo"];
				dWhySum += mymap["dWhy"];
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

			// update params。把每个场景看做一个样本的话，则是sgd。
			opt->optimize(this->Wf, this->alpha, dWfSum, mdWf, i);
			opt->optimize(this->Uf, this->alpha, dUfSum, mdUf, i);
			opt->optimize(this->bf, this->alpha, dbfSum, mdbf, i);
			opt->optimize(this->Wi, this->alpha, dWiSum, mdWi, i);
			opt->optimize(this->Ui, this->alpha, dUiSum, mdUi, i);
			opt->optimize(this->bi, this->alpha, dbiSum, mdbi, i);
			opt->optimize(this->Wa, this->alpha, dWaSum, mdWa, i);
			opt->optimize(this->Ua, this->alpha, dUaSum, mdUa, i);
			opt->optimize(this->ba, this->alpha, dbaSum, mdba, i);
			opt->optimize(this->Wo, this->alpha, dWoSum, mdWo, i);
			opt->optimize(this->Uo, this->alpha, dUoSum, mdUo, i);
			opt->optimize(this->bo, this->alpha, dboSum, mdbo, i);
			opt->optimize(this->Why, this->alpha, dWhySum, mdWhy, i);
			opt->optimize(this->by, this->alpha, dbySum, mdby, i);

		}

		// lossVec mean, accuracy
		double loss_this_epoch = MyLib<double>::mean_vector(loss_one_epoch); // 记录每个epoch的 loss
		double accu_this_epoch = MyLib<double>::mean_vector(true_false); // 记录每个epoch的 accu
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 10 == 0)
		{
			cout << "lambda: " << lambda << ", epoch: " << i << ", loss_mean_this_epoch: " << loss_this_epoch
				<< ", accu_this_epoch: " << accu_this_epoch << endl;
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

void RNN::predictOneScenario(mat Wxh, mat Whh, mat Why, mat bh, mat by, mat inputs, double score, double& loss, int& idx_target, int& idx_prediction)
{
	n_hidden = Whh.n_cols; // predict 时修改
	n_output_classes = by.n_rows; // predict时修改

	int idx1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	map<int, mat> xs, hs, ys, ps;

	mat hprev = arma::zeros<mat>(RNN::n_hidden, 1);
	hs[-1] = hprev; // 默认是 深拷贝
	mat loss_tmp = arma::zeros<mat>(1, 1);

	for (int t = 0; t < inputs.n_rows; t++)
	{
		// 对inputs一行一行进行，到最后一行时，有一个label
		xs[t] = inputs.row(t).t(); // 把行转置为列向量, (#features 20,1)
		hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);
		// (100, 20)(20,1) + (100,100)(100,1) + (100,1) = (100,1)
		if (t == inputs.n_rows - 1)
		{
			ys[t] = Why * hs[t] + by; // (#classes 3, 100)(100,1) + (3,1) = (3,1)

			mat sum_exp = arma::sum(arma::exp(ys[t]), 0);
			ps[t] = arma::exp(ys[t]) / sum_exp(0, 0); // (3,1) softmax

			uvec fuvec = arma::find(targets == 1); // index_targets
			loss_tmp += -log(ps[t](fuvec(0)));

			uvec idx_max_ps = arma::index_max(ps[t], 0); // index_prediction
			
			loss = loss_tmp(0, 0);
			idx_target = fuvec(0);
			idx_prediction = idx_max_ps(0);
		}
	}
}

void RNN::saveParams()
{
	this->Wf.save("Wf.txt", file_type::raw_ascii);
	this->Uf.save("Uf.txt", file_type::raw_ascii);
	this->bf.save("bf.txt", file_type::raw_ascii);
	this->Wi.save("Wi.txt", file_type::raw_ascii);
	this->Ui.save("Ui.txt", file_type::raw_ascii);
	this->bi.save("bi.txt", file_type::raw_ascii);
	this->Wa.save("Wa.txt", file_type::raw_ascii);
	this->Ua.save("Ua.txt", file_type::raw_ascii);
	this->ba.save("ba.txt", file_type::raw_ascii);
	this->Wo.save("Wo.txt", file_type::raw_ascii);
	this->Uo.save("Uo.txt", file_type::raw_ascii);
	this->bo.save("bo.txt", file_type::raw_ascii);
	this->Why.save("Why.txt", file_type::raw_ascii);
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
	mat sum_exp = arma::sum(arma::exp(mx), 0); // sum(,0) 列方向sum
	return arma::exp(mx) / sum_exp(0, 0);
}

map<string, mat> RNN::lossFun(mat inputs, 
	double score, double lambda, mat hprev, mat cprev,
	vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction)
{
	// 注：参数要在这个函数体外部提前初始化。
	int idx1 = -1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	map<int, mat> xs, os, cs, fs, is, as, hs, ys, ps;
	/*
		xs: one row of inputs
		os: output gate
		cs: cell
		fs: forget gate
		is: input gate
		as: candidate cell~
		hs: hidden
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
		xs[t] = inputs.row(t).t(); 
		
		//hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);
		fs[t] = RNN::sigmoid(Wf * hs[t-1] + Uf * xs[t] + bf);
		is[t] = RNN::sigmoid(Wi * hs[t-1] + Ui * xs[t] + bi);
		as[t] = arma::tanh(  Wa * hs[t-1] + Ua * xs[t] + ba);
		cs[t] = cs[t-1] % fs[t] + is[t] % as[t];
		os[t] = RNN::sigmoid(Wo * hs[t-1] + Uo * xs[t] + bo );
		hs[t] = os[t] % arma::tanh(cs[t]);

		if (t == inputs.n_rows - 1)
		{
			ys[t] = Why * hs[t] + by;
			
			ps[t] = RNN::softmax(ys[t]);

			loss += -log( ps[t](idx1) );

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 0); // index_prediction

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
	mat dWf, dUf, dbf; // dWhf, dWxf, dbf
	mat dWi, dUi, dbi;
	mat dWa, dUa, dba;
	mat dWo, dUo, dbo;
	mat dWhy, dby; // dhnext
	dWf = arma::zeros<mat>(Wf.n_rows, Wf.n_cols);
	dUf = arma::zeros<mat>(Uf.n_rows, Uf.n_cols);
	dbf = arma::zeros<mat>(bf.n_rows, bf.n_cols);
	dWi = arma::zeros<mat>(Wi.n_rows, Wi.n_cols);
	dUi = arma::zeros<mat>(Ui.n_rows, Ui.n_cols);
	dbi = arma::zeros<mat>(bi.n_rows, bi.n_cols);
	dWa = arma::zeros<mat>(Wa.n_rows, Wa.n_cols);
	dUa = arma::zeros<mat>(Ua.n_rows, Ua.n_cols);
	dba = arma::zeros<mat>(ba.n_rows, ba.n_cols);
	dWo = arma::zeros<mat>(Wo.n_rows, Wo.n_cols);
	dUo = arma::zeros<mat>(Uo.n_rows, Uo.n_cols);
	dbo = arma::zeros<mat>(bo.n_rows, bo.n_cols);
	dWhy = arma::zeros<mat>(Why.n_rows, Why.n_cols);
	dby = arma::zeros<mat>(by.n_rows, by.n_cols);
	//mat dhnext = arma::zeros<mat>(hs[0].n_rows, hs[0].n_cols);

	mat dy, dhh, deltaC;
	map<int, mat> dh, dc;
	for (int t = (int)inputs.n_rows - 1; t >= 0; t--)
	{
		if (t == inputs.n_rows - 1)
		{
			dy = ps[t];
			dy[idx1] -= 1;
			dWhy += dy * hs[t].t();
			dby += dy;


			dh[t] = Why.t() * dy; // dh,dc 只存储某个时刻 的值。如果不行的话，dh dc也用map结构
			dc[t] = dh[t] % os[t] % (1 - arma::tanh(cs[t]) % arma::tanh(cs[t]));

			dWf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])) * hs[t - 1].t(); // (h,1)(1,h)
			dUf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])) * xs[t].t(); // (h,1)(1,d)
			dbf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])); // (h,1)
			dWa += (dc[t] % is[t] % (1 - as[t] % as[t])) * hs[t - 1].t(); // (h,1)(1,h)
			dUa += (dc[t] % is[t] % (1 - as[t] % as[t])) * xs[t].t();
			dba += (dc[t] % is[t] % (1 - as[t] % as[t]));
			dWi += (dc[t] % as[t] % is[t] % (1 - is[t])) * hs[t - 1].t();
			dUi += (dc[t] % as[t] % is[t] % (1 - is[t])) * xs[t].t();
			dbi += (dc[t] % as[t] % is[t] % (1 - is[t]));
			dWo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t])) * hs[t - 1].t();
			dUo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t])) * xs[t].t();
			dbo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t]));

		}
		else
		{
			deltaC = os[t + 1] % ( 1- arma::tanh(cs[t+1]) % arma::tanh(cs[t+1]) );
			dhh = Wo.t() * ( os[t + 1] % (1 - os[t + 1]) % arma::tanh(cs[t+1]) )
				+ Wf.t() * ( deltaC % fs[t + 1] % (1 - fs[t+1]) % cs[t] ) 
				+ Wa.t() * (deltaC % is[t+1] % (1-( as[t+1]%as[t+1] )))
				+ Wi.t() * ( deltaC % as[t+1] % is[t+1] % (1-is[t+1])  );

			dh[t] = 0 + dhh.t() * dh[t + 1];
			dc[t] = dc[t + 1] % fs[t + 1] 
				+ dh[t] % os[t] % (1 - arma::tanh(cs[t]) % arma::tanh(cs[t]));
			
			dWf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])) * hs[t - 1].t(); // (h,1)(1,h)
			dUf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])) * xs[t].t(); // (h,1)(1,d)
			dbf += (dc[t] % cs[t - 1] % fs[t] % (1 - fs[t])); // (h,1)
			dWa += (dc[t] % is[t] % (1 - as[t] % as[t])) * hs[t - 1].t(); // (h,1)(1,h)
			dUa += (dc[t] % is[t] % (1 - as[t] % as[t])) * xs[t].t();
			dba += (dc[t] % is[t] % (1 - as[t] % as[t]));
			dWi += (dc[t] % as[t] % is[t] % (1 - is[t])) * hs[t - 1].t();
			dUi += (dc[t] % as[t] % is[t] % (1 - is[t])) * xs[t].t();
			dbi += (dc[t] % as[t] % is[t] % (1 - is[t]));
			dWo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t])) * hs[t - 1].t();
			dUo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t])) * xs[t].t();
			dbo += (dh[t] % arma::tanh(cs[t]) % os[t] % (1 - os[t]));

		}
	}


	// clip
	clip(dWf, 5.0, -5.0);
	clip(dUf, 5.0, -5.0);
	clip(dbf, 5.0, -5.0);
	clip(dWi, 5.0, -5.0);
	clip(dUi, 5.0, -5.0);
	clip(dbi, 5.0, -5.0);
	clip(dWa, 5.0, -5.0);
	clip(dUa, 5.0, -5.0);
	clip(dba, 5.0, -5.0);
	clip(dWo, 5.0, -5.0);
	clip(dUo, 5.0, -5.0);
	clip(dbo, 5.0, -5.0);
	clip(dWhy, 5.0, -5.0);
	clip(dby, 5.0, -5.0);


	map<string, mat> mymap;
	mymap["loss"] = loss;
	mymap["dWf"] = dWf;
	mymap["dUf"] = dUf;
	mymap["dbf"] = dbf;
	mymap["dWi"] = dWi;
	mymap["dUi"] = dUi;
	mymap["dbi"] = dbi;
	mymap["dWa"] = dWa;
	mymap["dUa"] = dUa;
	mymap["dba"] = dba;
	mymap["dWo"] = dWo;
	mymap["dUo"] = dUo;
	mymap["dbo"] = dbo;
	mymap["dWhy"] = dWhy;
	mymap["dby"] = dby;

	return mymap;
}

void RNN::clip(mat& matrix, double maxVal, double minVal)
{
	// transform 遍历每一个元素
	matrix.transform([maxVal, minVal](double val) {
		if (val >= maxVal)
		{
			return maxVal;
		}
		else if (val <= minVal)
		{
			return minVal;
		}
		else
		{
			return val;
		}
	});
}


mat RNN::score2onehot(double score, int& idx1)
{
	double part = 1.0 / n_output_classes;

	double pos = (score - score_min)
		/ (score_max - score_min + pow(10, -4)); // 分母加上 1e-8, 避免score是 max时的index越界

	idx1 = std::floor(pos / part);

	mat zs = arma::zeros<mat>(n_output_classes, 1);
	zs(idx1, 0) = 1;

	return zs;
}

map<string, arma::mat> RNN::getParams()
{
	map<string, arma::mat> mymap;
	/*mymap["Wxh"] = this->Wxh;
	mymap["Whh"] = this->Whh;
	mymap["Why"] = this->Why;
	mymap["bh"] = this->bh;
	mymap["by"] = this->by;*/

	mymap["lossAllVec"] = this->lossAllVec;

	return mymap;
}

void RNN::setParams(mat Wxh, mat Whh, mat Why, mat bh, mat by)
{
	/*this->Wxh = Wxh;
	this->Whh = Whh;
	this->Why = Why;
	this->bh = bh;
	this->by = by;*/
}
