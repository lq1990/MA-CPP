#include "RNN.h"



RNN::RNN()
{
	this->alpha = 0.01;
	this->totalSteps = 501;
	this->n_features = 17;
	this->n_hidden = 100;
	this->n_output_classes = 30;  // 回归转为分类，默认输出类别数目
	this->score_max = 9.0;
	this->score_min = 6.1;
	/*
	cout << "Without setting parameters in constructor by yourself," << 
		"the default parameters:" <<
		"\nlearning rate:	" << this->alpha  <<
		"\ntotalSteps:	"<< this->totalSteps << 
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

RNN::RNN(int n_features, int n_hidden, int n_output_classes, 
	double alpha, int totalSteps,
	double score_max, double score_min)
{
	this->n_features = n_features;
	this->n_hidden = n_hidden;
	this->n_output_classes = n_output_classes;
	this->alpha = alpha;
	this->totalSteps = totalSteps;
	this->score_max = score_max;
	this->score_min = score_min;
}

void RNN::initParams()
{
	this->Wxh = arma::randn(this->n_hidden, this->n_features) * 0.01;
	this->Whh = arma::randn(this->n_hidden, this->n_hidden) * 0.01;
	this->Why = arma::randn(this->n_output_classes, this->n_hidden) * 0.01;
	this->bh = arma::zeros(this->n_hidden, 1);
	this->by = arma::zeros(this->n_output_classes, 1);
}

/*
	train params of rnn-model.
	myMap: 存储所有场景score和matData的map。
*/
void RNN::train(map<const char*, MyStruct> myMap)
{
	map<const char*, MyStruct>::iterator it;
	it = myMap.begin();
	MyStruct sec = it->second;
	mat matData = sec.matData;
	this->n_features = matData.n_cols;

	// 初始化参数 Wxh, Whh, Why, bh, by
	this->initParams();

	//mat hprev = zeros<mat>(this->n_hidden, 1); // init hprev
	mat hprev;
	vector<double> lossVec; // 记录所有loss
	vector<double> loss_mean_each_epoch; // 只记录每个epoch中 loss mean
	/*
		外层循环: 遍历所有场景为一个 epoch。
		totalSteps: 是 epoch 数量
	*/
	for (int i = 0; i < this->totalSteps; i++)
	{
		/* 
			内层循环，
			遍历 allScenarios 中所有场景，拿到 matData & score 进行训练模型
		*/
		map<const char*, MyStruct>::iterator ascenario;
		for (ascenario = myMap.begin();
			ascenario != myMap.end(); 
			ascenario++)
		{
			MyStruct sec = ascenario->second;
			arma::mat matData = sec.matData;
			double score = sec.score;
			hprev = arma::zeros<mat>(this->n_hidden, 1); 
			// init hprev，使用每一个场景训练模型时，hprev 都是0，即hs[-1]=0
			map<string, mat> mp = lossFun(matData, score, hprev);

			arma::mat loss = mp["loss"];
			arma::mat dWxh = mp["dWxh"];
			arma::mat dWhh = mp["dWhh"];
			arma::mat dWhy = mp["dWhy"];
			arma::mat dbh = mp["dbh"];
			arma::mat dby = mp["dby"];
			//hprev = mp["last_hs"];
			
			lossVec.push_back(loss(0,0));
			// print loss
			if (i % 50 == 0)
			{
				cout << "	scenario: " << ascenario->first << endl;
				cout << "				loss: " << loss(0,0) << endl;
			}

			// update params
			this->Wxh -= this->alpha * dWxh;
			this->Whh -= this->alpha * dWhh;
			this->Why -= this->alpha * dWhy;
			this->bh -= this->alpha * dbh;
			this->by -= this->alpha * dby;
		}

		//lossVec mean
		double loss_mean = MyLib::mean_vector(lossVec);
		loss_mean_each_epoch.push_back(loss_mean);

		if (i % 10 == 0)
		{
			cout << "epoch: " << i << ", loss_mean: " << loss_mean <<endl;
		}


	}

	this->lossVec = lossVec;
	this->loss_mean_each_epoch = loss_mean_each_epoch;
}

void RNN::saveParams()
{
	this->Wxh.save("Wxh.txt", file_type::raw_ascii);
	this->Whh.save("Whh.txt", file_type::raw_ascii);
	this->Why.save("Why.txt", file_type::raw_ascii);
	this->bh.save("bh.txt", file_type::raw_ascii);
	this->by.save("by.txt", file_type::raw_ascii);

	mat loss_all = MyLib::vector2mat(this->lossVec);
	mat loss_mean_each_epoch = MyLib::vector2mat(this->loss_mean_each_epoch);

	loss_all.save("loss_all.txt", file_type::raw_ascii);
	loss_mean_each_epoch.save("loss_mean_each_epoch.txt", file_type::raw_ascii);
}

/*
	inputs: 某一个场景的matData
	score: 某一个场景的score，即label
*/
map<string, mat> RNN::lossFun(mat inputs, double score, mat hprev)
{
	// 注：参数要在这个函数体外部提前初始化。

	// score => targets(format: onehot)
	mat targets = this->score2onehot(score);

	//
	map<int, mat> xs, hs, ys, ps;
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

			uvec fuvec = arma::find(targets == 1);
			loss += -log(ps[t](fuvec(0)));
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
	for (int t = inputs.n_rows - 1; t >= 0 ; t--)
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
	mymap["last_hs"] = hs[inputs.n_rows-1];

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
					/ (this->score_max - this->score_min + pow(10, -8)); // 分母加上 1e-8, 避免score是 max时的index越界

	double pos_idx = std::floor(pos / part);

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
	mymap["lossVec"] = this->lossVec;

	return mymap;
}
