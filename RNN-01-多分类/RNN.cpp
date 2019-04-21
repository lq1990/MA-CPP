#include "RNN.h"


RNN::RNN()
{
	this->alpha = 0.01;
	this->total_epoches = 501;
	this->n_hidden = 100;
	this->n_output_classes = 3;  // 回归转为分类，默认输出类别数目
	this->score_max = 8.9;
	this->score_min = 6.0;
	/*
	cout << "Without setting parameters in constructor by yourself," << 
		"the default parameters:" <<
		"\nlearning rate:	" << this->alpha  <<
		"\ntotal_steps:	"<< this->total_steps << 
		"\nn_features:	"<< this->n_features << 
		"\nn_hidden:	"<< this->n_hidden << 
		"\nn_output_classes:	"<< this->n_output_classes <<
		"\nscore_max:	"<< this->score_max <<
		"\nscore_min:	"<< this->score_min <<

		"\n---------------------" << endl;
	*/
}


RNN::~RNN()
{
}

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
}

void RNN::initParams(map<const char*, MyStruct> myMap)
{
	map<const char*, MyStruct>::iterator it;
	it = myMap.begin();
	MyStruct sec = it->second;
	mat matData = sec.matData;
	this->n_features = (int)matData.n_cols;

	this->Wxh = arma::randn(this->n_hidden, this->n_features) * 0.01;
	this->Whh = arma::randn(this->n_hidden, this->n_hidden) * 0.01;
	this->Why = arma::randn(this->n_output_classes, this->n_hidden) * 0.01;
	this->bh = arma::zeros(this->n_hidden, 1);
	this->by = arma::zeros(this->n_output_classes, 1);
}

void RNN::train(map<const char*, MyStruct> myMap, AOptimizer* opt)
{

	// 初始化参数 Wxh, Whh, Why, bh, by
	this->initParams(myMap);

	// init memory，用于在 Adagrad中，计算L2范式，平方 求和 开根
	mat mdWxh = arma::zeros<mat>(Wxh.n_rows, Wxh.n_cols);
	mat mdWhh = arma::zeros<mat>(Whh.n_rows, Whh.n_cols);
	mat mdWhy = arma::zeros<mat>(Why.n_rows, Why.n_cols);
	mat mdbh = arma::zeros<mat>(bh.n_rows, bh.n_cols);
	mat mdby = arma::zeros<mat>(by.n_rows, by.n_cols);

	//mat hprev = zeros<mat>(this->n_hidden, 1); // init hprev
	mat hprev;
	vector<double> lossAllVec; // 记录所有loss
	vector<double> loss_one_epoch;
	vector<double> loss_mean_each_epoch; // 只记录每个epoch中 loss mean
	vector<double> true_false; // 存储一个epoch中，所有场景的正确率
	vector<double> accuracy_each_epoch; 
	/*
		外层循环: 遍历所有场景为一个 epoch。
		total_steps: 是 epoch 数量
	*/
	for (int i = 0; i < this->total_epoches; i++)
	{
		map<const char*, MyStruct>::iterator ascenario;
		true_false.clear(); // 清空
		loss_one_epoch.clear();
		/* 
			内层循环，
			遍历 allScenarios 中所有场景，拿到 matData & score 进行训练模型
		*/
		for (ascenario = myMap.begin();
			ascenario != myMap.end(); 
			ascenario++)
		{
			MyStruct sec = ascenario->second;
			arma::mat matData = sec.matData;
			double score = sec.score;
			hprev = arma::zeros<mat>(this->n_hidden, 1);

			// init hprev，使用每一个场景训练模型时，hprev 都是0，即hs[-1]=0
			map<string, mat> mp = lossFun(matData, score, hprev, true_false);

			arma::mat loss = mp["loss"];
			arma::mat dWxh = mp["dWxh"];
			arma::mat dWhh = mp["dWhh"];
			arma::mat dWhy = mp["dWhy"];
			arma::mat dbh = mp["dbh"];
			arma::mat dby = mp["dby"];
			//hprev = mp["last_hs"];

			// update params。把每个场景看做一个样本的话，则是sgd。======= 此处考虑用 多线程并行计算dWi，求dWi和用来更新参数，这样 it +=2 或更多
			opt->optimize(this->Wxh, this->alpha, dWxh, mdWxh, i);
			opt->optimize(this->Whh, this->alpha, dWhh, mdWhh, i);
			opt->optimize(this->Why, this->alpha, dWhy, mdWhy, i);
			opt->optimize(this->bh, this->alpha, dbh, mdbh, i);
			opt->optimize(this->by, this->alpha, dby, mdby, i);
			
			lossAllVec.push_back(loss(0, 0));
			loss_one_epoch.push_back(loss(0, 0));
			// print loss
			if (i % 50 == 0)
			{
				cout << "	scenario: " << ascenario->first << endl;
				cout << "				loss: " << loss(0,0) << endl;
			}

		}

		//lossVec mean, accuracy
		double loss_this_epoch = MyLib::mean_vector(loss_one_epoch);
		double accu_this_epoch = MyLib::mean_vector(true_false);
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 10 == 0)
		{
			cout << "epoch: " << i << ", loss_this_epoch: " << loss_this_epoch 
				<< ", accu_this_epoch: " << accu_this_epoch <<endl;
		}

	}

	this->lossAllVec = lossAllVec;
	this->loss_mean_each_epoch = loss_mean_each_epoch;
	this->accuracy_each_epoch = accuracy_each_epoch;
}

void RNN::trainMultiThread(map<const char*, MyStruct> myMap)
{
}


void RNN::saveParams()
{
	this->Wxh.save("Wxh.txt", file_type::raw_ascii);
	this->Whh.save("Whh.txt", file_type::raw_ascii);
	this->Why.save("Why.txt", file_type::raw_ascii);
	this->bh.save("bh.txt", file_type::raw_ascii);
	this->by.save("by.txt", file_type::raw_ascii);

	mat loss_all = MyLib::vector2mat(this->lossAllVec);
	mat loss_mean_each_epoch = MyLib::vector2mat(this->loss_mean_each_epoch);
	mat accuracy_each_epoch = MyLib::vector2mat(this->accuracy_each_epoch);

	loss_all.save("loss_all.txt", file_type::raw_ascii);
	loss_mean_each_epoch.save("loss_mean_each_epoch.txt", file_type::raw_ascii);
	accuracy_each_epoch.save("accuracy_each_epoch.txt", file_type::raw_ascii);
	
}


map<string, mat> RNN::lossFun(mat inputs, double score, mat hprev, vector<double>& true_false)
{
	// 注：参数要在这个函数体外部提前初始化。

	// score => targets(format: onehot)
	mat targets = this->score2onehot(score);

	//
	map<int, mat> xs, hs, ys, ps; // 使用map的原因：前传中计算的值会被保存，在BPTT中可使用。
	hs[-1] = hprev; // 默认是 深拷贝
	mat loss = arma::zeros<mat>(1, 1);

	// forward pass
	// inputs.n_rows 是 samples的数量，inputs.n_cols 是 features数量
	for (int t = 0; t < inputs.n_rows; t++)
	{
		// 对inputs一行一行进行，到最后一行时，有一个label
		xs[t] = inputs.row(t).t(); // 把行转置为列向量, (#features 20,1)
		hs[t] = arma::tanh(this->Wxh * xs[t] + this->Whh*hs[t-1] + this->bh);
		// (100, 20)(20,1) + (100,100)(100,1) + (100,1) = (100,1)
		if (t == inputs.n_rows - 1)
		{
			ys[t] = this->Why * hs[t] + by; 
			// (#classes 3, 100)(100,1) + (3,1) = (3,1)

			mat sum_exp = arma::sum(arma::exp(ys[t]), 0);
			ps[t] = arma::exp(ys[t]) / sum_exp(0, 0); // (3,1)

			uvec fuvec = arma::find(targets == 1); // index_targets
			loss += -log(ps[t](fuvec(0)));

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 0); // index_prediction
			if (fuvec(0) == idx_max_ps(0))
			{
				true_false.push_back(1);
			}
			else
			{
				true_false.push_back(0);
			}
		}
	}

	// BPTT
	mat dWxh, dWhh, dWhy, dbh, dby, dhnext;
	dWxh = arma::zeros<mat>(this->Wxh.n_rows, this->Wxh.n_cols);
	dWhh = arma::zeros<mat>(this->Whh.n_rows, this->Whh.n_cols);
	dWhy = arma::zeros<mat>(this->Why.n_rows, this->Why.n_cols);
	dbh = arma::zeros<mat>(this->bh.n_rows, this->bh.n_cols);
	dby = arma::zeros<mat>(this->by.n_rows, this->by.n_cols);
	dhnext = arma::zeros<mat>(hs[0].n_rows, hs[0].n_cols);

	mat dy, dh, dhraw;
	for (int t = (int)inputs.n_rows - 1; t >= 0 ; t--)
	{
		if (t == inputs.n_rows - 1)
		{
			dy = ps[t];
			uvec fuvec = arma::find(targets == 1);
			dy[fuvec(0)] -= 1;
			dWhy += dy * hs[t].t();
			dby += dy;

			dh = this->Why.t() * dy + dhnext;
			dhraw = (1 - hs[t] % hs[t]) % dh;
			dbh += dhraw;
			dWxh += dhraw * xs[t].t();
			dWhh += dhraw * hs[t-1].t();
			dhnext = this->Whh.t() * dhraw;
		}
		else
		{
			dh = dhnext;
			dhraw = (1 - hs[t] % hs[t]) % dh;
			dbh += dhraw;
			dWxh += dhraw * xs[t].t();
			dWhh += dhraw * hs[t - 1].t();
			dhnext = this->Whh.t() * dhraw;
		}
	}

	// clip 梯度
	dWxh = this->clip(dWxh, 5.0, -5.0);
	dWhh = this->clip(dWhh, 5.0, -5.0);
	dWhy = this->clip(dWhy, 5.0, -5.0);
	dbh = this->clip(dbh, 5.0, -5.0);
	dby = this->clip(dby, 5.0, -5.0);
	
	map<string, mat>mymap;
	mymap["loss"] = loss;
	mymap["dWxh"] = dWxh;
	mymap["dWhh"] = dWhh;
	mymap["dWhy"] = dWhy;
	mymap["dbh"] = dbh;
	mymap["dby"] = dby;
	//mymap["last_hs"] = hs[inputs.n_rows-1];

	return mymap;
}

mat RNN::clip(mat matrix, double maxVal, double minVal)
{
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
	return matrix;
}

mat RNN::score2onehot(double score)
{
	double part = 1.0 / this->n_output_classes;

	double pos = (score - this->score_min) 
					/ (this->score_max - this->score_min +pow(10, -8)); // 分母加上 1e-8, 避免score是 max时的index越界

	//cout << "pos/part: " << pos / part << endl;
	double pos_idx = std::floor(pos / part);
	//cout << "pos_idx: " << pos_idx << endl;

	mat zs = arma::zeros<mat>(this->n_output_classes, 1);
	zs(pos_idx, 0) = 1;

	return zs;
}

map<string, arma::mat> RNN::getParams()
{
	map<string, arma::mat> mymap;
	mymap["Wxh"] = this->Wxh;
	mymap["Whh"] = this->Whh;
	mymap["Why"] = this->Why;
	mymap["bh"] = this->bh;
	mymap["by"] = this->by;
	mymap["lossAllVec"] = this->lossAllVec;

	return mymap;
}

void RNN::setParams(mat Wxh, mat Whh, mat Why, mat bh, mat by)
{
	this->Wxh = Wxh;
	this->Whh = Whh;
	this->Why = Why;
	this->bh = bh;
	this->by = by;
}


