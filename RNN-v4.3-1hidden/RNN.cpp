#include "RNN.h"

std::mutex RNN::mtx;

int		RNN::total_epoches = 51;
double	RNN::alpha = 0.015; 
double	RNN::score_max = 10; // start: 8.9, gearShiftUp: 9.4
double	RNN::score_min = 4; // start: 6.0, gearShiftUp: 4.9
int		RNN::n_features = 16;
int		RNN::n_hidden = 30;
int		RNN::n_output_classes = 5;

double	RNN::dropout = 0.; // 去除neuron的占比
int		RNN::n_threads = 8;
map<std::thread::id, HiddenLayer*> RNN::threadPool;

int RNN::tmp = 11; // 试验是否可以 实例操作静态对象

Params* RNN::ph1 = new Params(RNN::n_features, RNN::n_hidden, RNN::n_output_classes);
//Params* RNN::ph2 = new Params(RNN::n_hidden, RNN::n_hidden, RNN::n_output_classes);


RNN::RNN()
{

}


RNN::~RNN()
{
}


void RNN::trainMultiThread(vector<SceStruct> listStructTrain, 
	AOptimizer* opt, int n_threads, double lambda)
{

	// memory of dP, used for Adagrad
	MDParams* mdPh1 = new MDParams();
	//MDParams* mdPh2 = new MDParams();
	mdPh1->setZeros(RNN::ph1);
	//mdPh2->setZeros(RNN::ph2);

	//mat hprev = zeros<mat>(this->n_hidden, 1); // init hprev
	mat hprev, cprev;
	vector<double> lossAllVec; // 记录所有loss
	vector<double> loss_one_epoch;
	vector<double> loss_mean_each_epoch; // 只记录每个epoch中 loss mean
	vector<double> true_false; // 存储一个epoch中，所有场景的正确率
	vector<double> accuracy_each_epoch;
	vector<double> log_target; // 记录所有epoches和所有场景的 target
	vector<double> log_prediction; // 与log_target对应，记录所有的模型 prediction

	vector<future<LossFunctionReturn>> vec_future; // 存储多线程的vector, future的泛型是 方法返回值

	// deltaP
	DParams* dPh1 = new DParams();
	//DParams* dPh2 = new DParams();

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
			dPh1->setZeros(RNN::ph1);
			//dPh2->setZeros(RNN::ph2);

			// 把惩罚项放到lossFun即一次前传反传的 外层，这样W在更新一次时，无论多少个线程，W只是加一次惩罚项
			dPh1->dWfSum += lambda * ph1->Wf;
			dPh1->dWiSum += lambda * ph1->Wi;
			dPh1->dWcSum += lambda * ph1->Wc;
			dPh1->dWoSum += lambda * ph1->Wo;
			dPh1->dWhhSum += lambda * ph1->Whh;

			for (int s = 0; s < vec_future.size(); s++)
			{
				LossFunctionReturn ret = vec_future[s].get(); // get 是阻塞型的

				loss = ret.loss;
				map<string, mat> deltaParamsH1 = ret.deltaParamsH1;
				map<string, mat> deltaParamsH2 = ret.deltaParamsH2;

				dPh1->dWfSum += deltaParamsH1["dWf"];
				dPh1->dWiSum += deltaParamsH1["dWi"];
				dPh1->dWcSum += deltaParamsH1["dWc"];
				dPh1->dWoSum += deltaParamsH1["dWo"];
				dPh1->dWhhSum += deltaParamsH1["dWhh"];
				dPh1->dbfSum += deltaParamsH1["dbf"];
				dPh1->dbiSum += deltaParamsH1["dbi"];
				dPh1->dbcSum += deltaParamsH1["dbc"];
				dPh1->dboSum += deltaParamsH1["dbo"];
				dPh1->dbhhSum += deltaParamsH1["dbhh"];

				/*dPh2->dWfSum += deltaParamsH2["dWf"];
				dPh2->dWiSum += deltaParamsH2["dWi"];
				dPh2->dWcSum += deltaParamsH2["dWc"];
				dPh2->dWoSum += deltaParamsH2["dWo"];
				dPh2->dWhhSum += deltaParamsH2["dWhh"];
				dPh2->dbfSum += deltaParamsH2["dbf"];
				dPh2->dbiSum += deltaParamsH2["dbi"];
				dPh2->dbcSum += deltaParamsH2["dbc"];
				dPh2->dboSum += deltaParamsH2["dbo"];
				dPh2->dbhhSum += deltaParamsH2["dbhh"];*/

				// store in vec
				lossAllVec.push_back(loss(0, 0));
				loss_one_epoch.push_back(loss(0, 0));

				// print loss
				if (i % 50 == 0)
				{
					// 把此epoch中的每个场景的 loss 打印
					cout << "				loss: " << loss(0, 0) << endl;
				}
			}

			// clip
			double maxVal, minVal;
			maxVal = 5.0;
			minVal = -5.0;

			clip(dPh1, maxVal, minVal);
			//clip(dPh2, maxVal, minVal);
			

			// update params。把每个场景看做一个样本的话，则是sgd。
			opt->optimize(ph1, this->alpha, dPh1, mdPh1, i);
			//opt->optimize(ph2, this->alpha, dPh2, mdPh2, i);

		}

		// lossVec mean, accuracy
		double loss_this_epoch = MyLib<double>::mean_vector(loss_one_epoch); // 记录每个epoch的 loss
		double accu_this_epoch = MyLib<double>::mean_vector(true_false); // 记录每个epoch的 accu
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 2 == 0)
		{
			cout << "dropout: " << RNN::dropout << ", lambda: " << lambda << ", epoch: " << i << " / " << RNN::total_epoches
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


	在RNN-v4.2版本中，LSTM多隐层使用 类HiddenLayer进行new出来。


*/
LossFunctionReturn RNN::lossFun(mat inputs,
	double score, double lambda, mat hprev, mat cprev,
	vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction)
{
	// 由于new对象耗时，使用 HiddenLayerPool，n个线程则在pool中创建n个hiddenLayer对象。根据线程id进行分配

	// 方式1：使用map
	/*
	*/
	//printf("in lossFun, thread id: %d\n", std::this_thread::get_id());
	auto id = std::this_thread::get_id();

	if (RNN::threadPool.size() < RNN::n_threads) 
	{
		mtx.lock();
		if (RNN::threadPool.size() < RNN::n_threads)
		{
			RNN::threadPool[id] = new HiddenLayer(RNN::n_features, RNN::n_hidden, RNN::n_output_classes, RNN::ph1);
		}
		mtx.unlock();
	}

	
	// 从线程池中拿线程，不同线程按照自己的id
	HiddenLayer* hiddenLayer1 = RNN::threadPool[id];
	hiddenLayer1->setParams(RNN::ph1); // set参数原因：将update过的RNN参数赋值到隐层参数

	// 方法2：使用vector，在此fn开始时取出，fn最后还回
	

	/* 
		多线程情况下，必须每个线程都有自己的HiddenLayer实例，才能保证HiddenLayer中属性数据线程独有.
		通过前传、反传，得到dP return。
		在train fn中，多个线程得到的dP sum. 最终用sum进行优化p

	*/


	
	// 注：参数要在这个函数体外部提前初始化。
	int idx1 = -1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)


	map<int, mat> ys, ps; // ys: prediction


	mat loss = arma::zeros<mat>(1, 1);

	// ===================== forward pass =========================

		// 把场景数据inputs传给H1
		hiddenLayer1->hiddenForward(inputs, hprev, cprev, RNN::dropout);

		// 把H1计算结果hs传给H2
		//hiddenLayer2->hiddenForward(HiddenLayer::map2mat(hiddenLayer1->hs, 0, inputs.n_rows-1), hprev, cprev); // map2mat()中的hs[idx=-1,0,1,...]第一行是tprev状态的，不用
		//hiddenLayer2->hiddenForward(hiddenLayer1->hs, hprev, cprev, RNN::dropout); // map2mat()中的hs[idx=-1,0,1,...]第一行是tprev状态的，不用


		// 计算output // 在最后一个时刻，last hidden 的hs[t] 传给output
		int t = inputs.n_rows - 1;
			ys[t] = hiddenLayer1->hs[t] * hiddenLayer1->Whh + hiddenLayer1->bhh; // (1,h)(h,y) = (1,y)
			ps[t] = RNN::softmax(ys[t]); // (1,y)

			loss += -log( ps[t](idx1) );

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


	// =============== BPTT ============================

	map<int, mat> dy;
	// init dy
	for (int i=0; i < inputs.n_rows; i++) 
	{
		dy[i] = zeros<mat>(1, RNN::n_output_classes);
	}
	
	t = inputs.n_rows - 1;
		
			// softmax loss gradient
			dy[t] = ps[t];
			dy[t][idx1] -= 1;

			// H2
			//map<int, mat> d_outputs_out2;
			//map<string,mat> deltaParamsH2 = hiddenLayer2->hiddenBackward(HiddenLayer::map2mat(hiddenLayer1->hs, 0, inputs.n_rows-1), dy, d_outputs_out2, lambda);
			//map<string,mat> deltaParamsH2 = hiddenLayer2->hiddenBackward(hiddenLayer1->hs, dy, d_outputs_out2, lambda);

			// H1
			map<int, mat> d_outputs_out1;
			map<string, mat> deltaParamsH1 = hiddenLayer1->hiddenBackward(inputs, dy, d_outputs_out1, lambda);


	LossFunctionReturn ret;
	ret.loss = loss;
	ret.deltaParamsH1 = deltaParamsH1;
	//ret.deltaParamsH2 = deltaParamsH2;

	return ret;
}



/*
	H1 --> H2 计算出 hs of H2, 
	--> ps
*/
void RNN::predictOneScenario(Params* ph1,
	mat inputs,
	double score, double& loss, int& idx_target, int& idx_prediction)
{
	/*
	n_hidden = bf.n_cols; // predict 时修改
	n_output_classes = by.n_cols; // predict时修改
	*/

	int idx1;
	mat targets = score2onehot(score, idx1); // score => targets(format: onehot)

	/*
	map<int, mat> xs, X,
		h_fs, h_is, h_os, h_cs, // hidden gate state
		cs, hs;
	*/
	map<int,mat> ys, ps;

	mat h1prev = arma::zeros<mat>(1, RNN::n_hidden);
	mat c1prev = arma::zeros<mat>(1, RNN::n_hidden);

	mat h2prev = arma::zeros<mat>(1, RNN::n_hidden);
	mat c2prev = arma::zeros<mat>(1, RNN::n_hidden);
	/*
	hs[-1] = hprev; // 默认是 深拷贝
	cs[-1] = cprev; // 默认是 深拷贝
	*/


	mat loss_tmp = arma::zeros<mat>(1, 1);

	
	HiddenLayer* hiddenLayer1 = new HiddenLayer(RNN::n_features, RNN::n_hidden, RNN::n_hidden, ph1);
	//HiddenLayer* hiddenLayer2 = new HiddenLayer(RNN::n_hidden, RNN::n_hidden, RNN::n_output_classes, ph2);
	hiddenLayer1->hiddenForward(inputs, h1prev, c1prev, 0); // 预测时，dropout=0
	//hiddenLayer2->hiddenForward(hiddenLayer1->hs, h2prev, c2prev, 0);

	int t = inputs.n_rows - 1;
			//ys[t] = hs[t] * Wy + by; // (1,h)(h,y) = (1,y)
	ys[t] = hiddenLayer1->hs[t] * hiddenLayer1->Whh + hiddenLayer1->bhh;
	ps[t] = RNN::softmax(ys[t]);

			loss_tmp += -log(ps[t](idx1));

			// accuracy
			uvec idx_max_ps = arma::index_max(ps[t], 1); // index_prediction

			loss = loss_tmp(0, 0);
			idx_target = idx1;
			idx_prediction = idx_max_ps(0);
	
}

void RNN::saveParams()
{
	RNN::ph1->save("1");
	//RNN::ph2->save("2");

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
	return 1. / (1 + arma::exp(-mx)); 
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

void RNN::initParams()
{
	RNN::ph1 = new Params(RNN::n_features, RNN::n_hidden, RNN::n_output_classes);

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

void RNN::clip(DParams * dP, double maxVal, double minVal)
{
	RNN::clip(dP->dWfSum, maxVal, minVal);
	RNN::clip(dP->dWiSum, maxVal, minVal);
	RNN::clip(dP->dWcSum, maxVal, minVal);
	RNN::clip(dP->dWoSum, maxVal, minVal);
	RNN::clip(dP->dWhhSum, maxVal, minVal);

	RNN::clip(dP->dbfSum, maxVal, minVal);
	RNN::clip(dP->dbiSum, maxVal, minVal);
	RNN::clip(dP->dbcSum, maxVal, minVal);
	RNN::clip(dP->dboSum, maxVal, minVal);
	RNN::clip(dP->dbhhSum, maxVal, minVal);
}

mat RNN::score2onehot(double score, int& idx1)
{
	if (score > score_max || score < score_min)
	{
		// cout << "score > score_max || score < score_min" << endl;
	}

	if (score > score_max)
	{
		score = score_max;
	}

	if (score < score_min)
	{
		score = score_min;
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

}

map<string, arma::mat> RNN::getParams()
{
	map<string, arma::mat> mymap;

	/*mymap["Wf"] = this->Wf;
	mymap["Wi"] = this->Wi;
	mymap["Wc"] = this->Wc;
	mymap["Wo"] = this->Wo;
	mymap["Wy"] = this->Wy;

	mymap["bf"] = this->bf;
	mymap["bi"] = this->bi;
	mymap["bc"] = this->bc;
	mymap["bo"] = this->bo;
	mymap["by"] = this->by;*/

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

	ph1->load("1");
	//ph2->load("2");

	this->tmp = 33; // 可以由实例操作 静态变量

}
