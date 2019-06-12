#include "RNN.h"

std::mutex RNN::mtx;

/*
	静态属性的初始化如下。之后可用 MyParams类来保存，以便全局使用和修改
*/
double RNN::score_max = MyParams::score_max;
double RNN::score_min = MyParams::score_min;
int RNN::n_features = MyParams::n_features;
int RNN::n_hidden = MyParams::n_hidden;
int RNN::n_output_classes = MyParams::n_output_classes;

mat RNN::Wxh1 = MyParams::Wxh1;
mat RNN::Wh1h1 = MyParams::Wh1h1;
mat RNN::Wh1h2 = MyParams::Wh1h2;
mat RNN::Wh2h2 = MyParams::Wh2h2;
mat RNN::Wh2y = MyParams::Wh2y;
mat RNN::bh1 = MyParams::bh1;
mat RNN::bh2 = MyParams::bh2;
mat RNN::by = MyParams::by;


RNN::RNN()
{
	// init params
	this->alpha = MyParams::alpha;
	this->total_epoches = MyParams::total_epoches;
	this->score_max = MyParams::score_max;
	this->score_min = MyParams::score_min;
	this->n_features = MyParams::n_features;
	this->n_hidden = MyParams::n_hidden;
	this->n_output_classes = MyParams::n_output_classes;  // 回归转为分类，默认输出类别数目
	this->Wxh1 = MyParams::Wxh1;
	this->Wh1h1 = MyParams::Wh1h1;
	this->Wh1h2 = MyParams::Wh1h2;
	this->Wh2h2 = MyParams::Wh2h2;
	this->Wh2y = MyParams::Wh2y;
	this->bh1 = MyParams::bh1;
	this->bh2 = MyParams::bh2;
	this->by = MyParams::by;
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
	mat mdWxh1 = arma::zeros<mat>(Wxh1.n_rows, Wxh1.n_cols);
	mat mdWh1h1 = arma::zeros<mat>(Wh1h1.n_rows, Wh1h1.n_cols);
	mat mdWh1h2 = arma::zeros<mat>(Wh1h2.n_rows, Wh1h2.n_cols);
	mat mdWh2h2 = arma::zeros<mat>(Wh2h2.n_rows, Wh2h2.n_cols);
	mat mdWh2y = arma::zeros<mat>(Wh2y.n_rows, Wh2y.n_cols);
	mat mdbh1 = arma::zeros<mat>(bh1.n_rows, bh1.n_cols);
	mat mdbh2 = arma::zeros<mat>(bh2.n_rows, bh2.n_cols);
	mat mdby = arma::zeros<mat>(by.n_rows, by.n_cols);

	mat h1prev, h2prev;
	vector<double> lossAllVec; // 记录所有loss
	vector<double> loss_one_epoch;
	vector<double> loss_mean_each_epoch; // 只记录每个epoch中 loss mean
	vector<double> true_false; // 存储一个epoch中，所有场景的正确率
	vector<double> accuracy_each_epoch;
	vector<double> log_target; // 记录所有epoches和所有场景的 target
	vector<double> log_prediction; // 与log_target对应，记录所有的模型 prediction

	vector<future<map<string, mat>>> vec_future; // 存储多线程的vector
	mat dWxh1Sum, dWh1h1Sum, dWh1h2Sum, dWh2h2Sum, dWh2ySum, dbh1Sum, dbh2Sum, dbySum, loss;

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

				h1prev = arma::zeros<mat>(this->n_hidden, 1); // for each scenario, hprev is 0
				h2prev = arma::zeros<mat>(this->n_hidden, 1); // for each scenario, hprev is 0
				vec_future.push_back(async(RNN::lossFun, matData, score, lambda, h1prev, h2prev, std::ref(true_false), std::ref(log_target), std::ref(log_prediction)));
			}

			// 清零。在多个线程计算之后，get结果 dW db
			dWxh1Sum = arma::zeros<mat>(Wxh1.n_rows, Wxh1.n_cols);
			dWh1h1Sum = arma::zeros<mat>(Wh1h1.n_rows, Wh1h1.n_cols);
			dWh1h2Sum = arma::zeros<mat>(Wh1h2.n_rows, Wh1h2.n_cols);
			dWh2h2Sum = arma::zeros<mat>(Wh2h2.n_rows, Wh2h2.n_cols);
			dWh2ySum = arma::zeros<mat>(Wh2y.n_rows, Wh2y.n_cols);
			dbh1Sum = arma::zeros<mat>(bh1.n_rows, bh1.n_cols);
			dbh2Sum = arma::zeros<mat>(bh2.n_rows, bh2.n_cols);
			dbySum = arma::zeros<mat>(by.n_rows, by.n_cols);

			for (int s = 0; s < vec_future.size(); s++)
			{
				map<string, mat> res = vec_future[s].get(); // get 是阻塞型的

				loss = res["loss"];
				dWxh1Sum += res["dWxh1"];
				dWh1h1Sum += res["dWh1h1"];
				dWh1h2Sum += res["dWh1h2"];
				dWh2h2Sum += res["dWh2h2"];
				dWh2ySum += res["dWh2y"];
				dbh1Sum += res["dbh1"];
				dbh2Sum += res["dbh2"];
				dbySum += res["dby"];

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
			opt->optimize(this->Wxh1, this->alpha, dWxh1Sum, mdWxh1, i);
			opt->optimize(this->Wh1h1, this->alpha, dWh1h1Sum, mdWh1h1, i);
			opt->optimize(this->Wh1h2, this->alpha, dWh1h2Sum, mdWh1h2, i);
			opt->optimize(this->Wh2h2, this->alpha, dWh2h2Sum, mdWh2h2, i);
			opt->optimize(this->Wh2y, this->alpha, dWh2ySum, mdWh2y, i);
			opt->optimize(this->bh1, this->alpha, dbh1Sum, mdbh1, i);
			opt->optimize(this->bh2, this->alpha, dbh2Sum, mdbh2, i);
			opt->optimize(this->by, this->alpha, dbySum, mdby, i);
		}

		// lossVec mean, accuracy
		double loss_this_epoch = MyLib<double>::mean_vector(loss_one_epoch); // 记录每个epoch的 loss
		double accu_this_epoch = MyLib<double>::mean_vector(true_false); // 记录每个epoch的 accu
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 10 == 0)
		{
			cout << "epoch: " << i << ", loss_mean_this_epoch: " << loss_this_epoch
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
	MyParams::n_hidden = Whh.n_cols; // predict 时修改
	MyParams::n_output_classes = by.n_rows; // predict时修改

	mat targets = score2onehot(score); // score => targets(format: onehot)

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
	this->Wxh1.save("Wxh1.txt", file_type::raw_ascii);
	this->Wh1h1.save("Wh1h1.txt", file_type::raw_ascii);
	this->Wh1h2.save("Wh1h2.txt", file_type::raw_ascii);
	this->Wh2h2.save("Wh2h2.txt", file_type::raw_ascii);
	this->Wh2y.save("Wh2y.txt", file_type::raw_ascii);
	this->bh1.save("bh1.txt", file_type::raw_ascii);
	this->bh2.save("bh2.txt", file_type::raw_ascii);
	this->by.save("by.txt", file_type::raw_ascii);

	mat loss_all = MyLib<double>::vector2mat(this->lossAllVec);
	mat loss_mean_each_epoch = MyLib<double>::vector2mat(this->loss_mean_each_epoch);
	mat accuracy_each_epoch = MyLib<double>::vector2mat(this->accuracy_each_epoch);

	loss_all.save("loss_all.txt", file_type::raw_ascii);
	loss_mean_each_epoch.save("loss_mean_each_epoch.txt", file_type::raw_ascii);
	accuracy_each_epoch.save("accuracy_each_epoch.txt", file_type::raw_ascii);
	log_target_prediction.save("log_target_prediction.txt", file_type::raw_ascii);

}

map<string, mat> RNN::lossFun(mat inputs, double score, double lambda, mat h1prev, mat h2prev, vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction)
{
	// 注：参数要在这个函数体外部提前初始化。
	mat targets = score2onehot(score); // score => targets(format: onehot)

	map<int, mat> xs, h1s, h2s, ys, ps; // 使用map的原因：前传中计算的值会被保存，在BPTT中可使用。
	h1s[-1] = h1prev; // 默认是 深拷贝
	h2s[-1] = h2prev; // 默认是 深拷贝
	mat loss = arma::zeros<mat>(1, 1);

	// forward pass, inputs.n_rows 是 samples的数量，inputs.n_cols 是 features数量
	for (int t = 0; t < inputs.n_rows; t++)
	{
		// 对inputs一行一行进行，到最后一行时，有一个label

		xs[t] = inputs.row(t).t(); // 把行转置为列向量, (#features 20,1)
		h1s[t] = arma::tanh(Wxh1 * xs[t] + Wh1h1 * h1s[t - 1] + bh1); // (100, 20)(20,1) + (100,100)(100,1) + (100,1) = (100,1)
		h2s[t] = arma::tanh(Wh1h2 * h1s[t] + Wh2h2 * h2s[t-1] + bh2);

		if (t == inputs.n_rows - 1)
		{
			ys[t] = Wh2y * h2s[t] + by;
			// (#classes 3, 100)(100,1) + (3,1) = (3,1)

			mat sum_exp = arma::sum(arma::exp(ys[t]), 0);
			ps[t] = arma::exp(ys[t]) / sum_exp(0, 0); // (3,1) softmax

			uvec fuvec = arma::find(targets == 1); // index_targets
			loss += -log(ps[t](fuvec(0)));

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 0); // index_prediction

			mtx.lock();
			log_target.push_back(fuvec(0));
			log_prediction.push_back(idx_max_ps(0));
			if (fuvec(0) == idx_max_ps(0))
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

	// BPTT
	mat dWxh1, dWh1h1, dWh1h2, dWh2h2, dWh2y, dbh1, dbh2, dby, dh1next, dh2next;
	dWxh1 = arma::zeros<mat>(Wxh1.n_rows, Wxh1.n_cols);
	dWh1h1 = arma::zeros<mat>(Wh1h1.n_rows, Wh1h1.n_cols);
	dWh1h2 = arma::zeros<mat>(Wh1h2.n_rows, Wh1h2.n_cols);
	dWh2h2 = arma::zeros<mat>(Wh2h2.n_rows, Wh2h2.n_cols);
	dWh2y = arma::zeros<mat>(Wh2y.n_rows, Wh2y.n_cols);
	dbh1 = arma::zeros<mat>(bh1.n_rows, bh1.n_cols);
	dbh2 = arma::zeros<mat>(bh2.n_rows, bh2.n_cols);
	dby = arma::zeros<mat>(by.n_rows, by.n_cols);
	dh1next = arma::zeros<mat>(h1s[0].n_rows, h1s[0].n_cols);
	dh2next = arma::zeros<mat>(h2s[0].n_rows, h2s[0].n_cols);

	mat dy, dh1, dh1raw, dh2, dh2raw;
	for (int t = (int)inputs.n_rows - 1; t >= 0; t--)
	{
		if (t == inputs.n_rows - 1)
		{
			// 只有最后一个时刻，h2s才和label有关系。
			dy = ps[t];
			uvec fuvec = arma::find(targets == 1);
			dy[fuvec(0)] -= 1;
			dWh2y += dy * h2s[t].t() + lambda * Wh2y; // ys[t] = Why * hs[t] + b
			dby += dy;


			dh2 = Wh2y.t() * dy + dh2next; 
			dh2raw = (1 - h2s[t] % h2s[t]) % dh2;
			dbh2 += dh2raw;

			dWh1h2 += dh2raw * h1s[t].t() + lambda * Wh1h2;
			dWh2h2 += dh2raw * h2s[t-1].t() + lambda * Wh2h2;
			dh2next = Wh2h2.t() * dh2raw;

			dh1 = Wh1h2.t() * dh2raw + dh1next;
			dh1raw = (1 - h1s[t] % h1s[t]) % dh1;
			dbh1 += dh1raw;
			dWxh1 += dh1raw * xs[t].t() + lambda * Wxh1;
			dWh1h1 += dh1raw * h1s[t - 1].t() + lambda * Wh1h1;
			dh1next = Wh1h1.t() * dh1raw;
		}
		else // if/else 都有 W，勿忘 都 加惩罚
		{
			// 在其余大部分时刻，h2s没有通过 Wh2y by 与lable关联。
			dh2 = dh2next;
			dh2raw = (1 - h2s[t] % h2s[t]) % dh2;
			dbh2 += dh2raw;

			dWh1h2 += dh2raw * h1s[t].t() + lambda * Wh1h2;
			dWh2h2 += dh2raw * h2s[t - 1].t() + lambda * Wh2h2;
			dh2next = Wh2h2.t() * dh2raw;

			dh1 = Wh1h2.t() * dh2raw + dh1next;
			dh1raw = (1 - h1s[t] % h1s[t]) % dh1;
			dbh1 += dh1raw;
			dWxh1 += dh1raw * xs[t].t() + lambda * Wxh1;
			dWh1h1 += dh1raw * h1s[t - 1].t() + lambda * Wh1h1;
			dh1next = Wh1h1.t() * dh1raw;
		}
	}

	clip(dWxh1, 5.0, -5.0);
	clip(dWh1h1, 5.0, -5.0);
	clip(dWh1h2, 5.0, -5.0);
	clip(dWh2h2, 5.0, -5.0);
	clip(dWh2y, 5.0, -5.0);
	clip(dbh1, 5.0, -5.0);
	clip(dbh2, 5.0, -5.0);
	clip(dby, 5.0, -5.0);

	map<string, mat>mymap;
	mymap["loss"] = loss;
	mymap["dWxh1"] = dWxh1;
	mymap["dWh1h1"] = dWh1h1;
	mymap["dWh1h2"] = dWh1h2;
	mymap["dWh2h2"] = dWh2h2;
	mymap["dWh2y"] = dWh2y;
	mymap["dbh1"] = dbh1;
	mymap["dbh2"] = dbh2;
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


mat RNN::score2onehot(double score)
{
	double part = 1.0 / n_output_classes;

	double pos = (score - score_min)
		/ (score_max - score_min + pow(10, -8)); // 分母加上 1e-8, 避免score是 max时的index越界

	double pos_idx = std::floor(pos / part);

	mat zs = arma::zeros<mat>(n_output_classes, 1);
	zs(pos_idx, 0) = 1;

	return zs;
}

map<string, arma::mat> RNN::getParams()
{
	map<string, arma::mat> mymap;
	mymap["Wxh1"] = this->Wxh1;
	mymap["Wh1h1"] = this->Wh1h1;
	mymap["Wh1h2"] = this->Wh1h2;
	mymap["Wh2h2"] = this->Wh2h2;
	mymap["Wh2y"] = this->Wh2y;
	mymap["bh1"] = this->bh1;
	mymap["bh2"] = this->bh2;
	mymap["by"] = this->by;

	mymap["lossAllVec"] = this->lossAllVec;

	return mymap;
}

void RNN::setParams(mat Wxh1, mat Wh1h1, mat Wh1h2, mat Wh2h2, mat Wh2y, mat bh1, mat bh2, mat by)
{
	this->Wxh1 = Wxh1;
	this->Wh1h1 = Wh1h1;
	this->Wh1h2 = Wh1h2;
	this->Wh2h2 = Wh2h2;
	this->Wh2y = Wh2y;
	this->bh1 = bh1;
	this->bh2 = bh2;
	this->by = by;
}
