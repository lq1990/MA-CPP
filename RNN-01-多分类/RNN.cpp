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
mat RNN::Wxh = MyParams::Wxh;
mat RNN::Whh = MyParams::Whh;
mat RNN::Why = MyParams::Why;
mat RNN::bh = MyParams::bh;
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
	this->Wxh = MyParams::Wxh;
	this->Whh = MyParams::Whh;
	this->Why = MyParams::Why;
	this->bh = MyParams::bh;
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


void RNN::train(map<string, MyStruct> myMap, AOptimizer* opt)
{
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
	vector<double> log_target; // 记录所有epoches和所有场景的 target
	vector<double> log_prediction; // 与log_target对应，记录所有的模型 prediction

	/*
		外层循环: 遍历所有场景为一个 epoch。
		total_steps: 是 epoch 数量
	*/
	for (int i = 0; i < this->total_epoches; i++)
	{
		map<string, MyStruct>::iterator ascenario;
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
			map<string, mat> mp = lossFun(matData, score, hprev, true_false, log_target, log_prediction);

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

void RNN::trainMultiThread(map<string, MyStruct> myMap, AOptimizer* opt, int n_threads)
{
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
	vector<double> log_target; // 记录所有epoches和所有场景的 target
	vector<double> log_prediction; // 与log_target对应，记录所有的模型 prediction
	
	vector<future<map<string, mat>>> vec_future; // 存储多线程的vector
	mat dWxhSum, dWhhSum, dWhySum, dbhSum, dbySum, loss;

	/*
		外层循环: 遍历所有场景为一个 epoch。
		total_steps: 是 epoch 数量
	*/
	for (int i = 0; i < this->total_epoches; i++)
	{
		map<string, MyStruct>::iterator ascenario;
		true_false.clear(); // 清空
		loss_one_epoch.clear();
		/*
			内层循环，
			遍历 allScenarios 中所有场景，拿到 matData & score 进行训练模型
		*/
		ascenario = myMap.begin(); // 每个epoch的开始，都要重置 ascenario
		while (ascenario != myMap.end())
		{
			vec_future.clear(); // 每次多线程计算后，clear
			for (int t = 0; t < n_threads && ascenario != myMap.end(); t++, ascenario++)
			{
				/*if (ascenario != myMap.end()) // 把这个if条件写在 for（）里
				{	*/

				arma::mat matData = ascenario->second.matData;
				double score = ascenario->second.score;

				hprev = arma::zeros<mat>(this->n_hidden, 1);
				vec_future.push_back(async(RNN::lossFun, matData, score, hprev, std::ref(true_false), std::ref(log_target), std::ref(log_prediction)));

				//	ascenario = std::next(ascenario); // ascenario++
				//}
			}

			// 清零。在多个线程计算之后，get结果 dW db
			dWxhSum = arma::zeros<mat>(Wxh.n_rows, Wxh.n_cols);
			dWhhSum = arma::zeros<mat>(Whh.n_rows, Whh.n_cols);
			dWhySum = arma::zeros<mat>(Why.n_rows, Why.n_cols);
			dbhSum = arma::zeros<mat>(bh.n_rows, bh.n_cols);
			dbySum = arma::zeros<mat>(by.n_rows, by.n_cols);

			for (int s = 0; s < vec_future.size(); s++)
			{
				map<string, mat> res = vec_future[s].get();
				
				loss = res["loss"];
				dWxhSum += res["dWxh"];
				dWhhSum += res["dWhh"];
				dWhySum += res["dWhy"];
				dbhSum += res["dbh"];
				dbySum += res["dby"];

				// store in vec
				lossAllVec.push_back(loss(0, 0));
				loss_one_epoch.push_back(loss(0, 0));

				// print loss
				if (i % 100 == 0)
				{
					cout << "				loss: " << loss(0, 0) << endl;
				}
			}

			// update params。把每个场景看做一个样本的话，则是sgd。
			opt->optimize(this->Wxh, this->alpha, dWxhSum, mdWxh, i);
			opt->optimize(this->Whh, this->alpha, dWhhSum, mdWhh, i);
			opt->optimize(this->Why, this->alpha, dWhySum, mdWhy, i);
			opt->optimize(this->bh, this->alpha, dbhSum, mdbh, i);
			opt->optimize(this->by, this->alpha, dbySum, mdby, i);

		}

		/*
		for (ascenario = myMap.begin();
			ascenario != myMap.end();
			ascenario++, ascenario++, ascenario++, ascenario++)
		{
			MyStruct sec = ascenario->second; // 多线程中第1个处理
			arma::mat matData = sec.matData;
			double score = sec.score;

			MyStruct sec_next1 = std::next(ascenario)->second; // 多线程中第2个处理
			arma::mat matData_next1 = sec_next1.matData;
			double score_next1 = sec_next1.score;

			MyStruct sec_next2 = std::next(std::next(ascenario))->second; // 多线程中第3个处理
			arma::mat matData_next2 = sec_next2.matData;
			double score_next2 = sec_next2.score;

			MyStruct sec_next3 = std::next(std::next(std::next(ascenario)))->second; // 多线程中第4个处理
			arma::mat matData_next3 = sec_next3.matData;
			double score_next3 = sec_next3.score;

			// init hprev，使用每一个场景训练模型时，hprev 都是0，即hs[-1]=0。对每个场景都是一样的，test时也这样。
			hprev = arma::zeros<mat>(this->n_hidden, 1);

			// 多线程，并行计算。此处用 vec_future存储async对象，async(fn, )。后期可用 get方法拿到future对象的计算结果 ========================
			map<string, mat> mp1, mp2, mp3, mp4;
			std::thread myt1(lossFunMultiThread, matData, score, hprev, std::ref(true_false), std::ref(mp1));
			std::thread myt2(lossFunMultiThread, matData_next1, score_next1, hprev, std::ref(true_false));
			std::thread myt3(lossFunMultiThread, matData_next2, score_next2, hprev, std::ref(true_false));
			std::thread myt4(lossFunMultiThread, matData_next3, score_next3, hprev, std::ref(true_false));
			myt1.join();
			myt2.join();
			myt3.join();
			myt4.join();

			//map<string, mat> mp = lossFun(matData, score, hprev, true_false);
			arma::mat loss1 = mp1["loss"];
			arma::mat dWxh1 = mp1["dWxh"];
			arma::mat dWhh1 = mp1["dWhh"];
			arma::mat dWhy1 = mp1["dWhy"];
			arma::mat dbh1 = mp1["dbh"];
			arma::mat dby1 = mp1["dby"];

			arma::mat loss2 = mp2["loss"];
			arma::mat dWxh2 = mp2["dWxh"];
			arma::mat dWhh2 = mp2["dWhh"];
			arma::mat dWhy2 = mp2["dWhy"];
			arma::mat dbh2 = mp2["dbh"];
			arma::mat dby2 = mp2["dby"];

			arma::mat loss3 = mp3["loss"];
			arma::mat dWxh3 = mp3["dWxh"];
			arma::mat dWhh3 = mp3["dWhh"];
			arma::mat dWhy3 = mp3["dWhy"];
			arma::mat dbh3 = mp3["dbh"];
			arma::mat dby3 = mp3["dby"];

			arma::mat loss4 = mp4["loss"];
			arma::mat dWxh4 = mp4["dWxh"];
			arma::mat dWhh4 = mp4["dWhh"];
			arma::mat dWhy4 = mp4["dWhy"];
			arma::mat dbh4 = mp4["dbh"];
			arma::mat dby4 = mp4["dby"];

			mat dWxh = dWxh1 + dWxh2 + dWxh3 + dWxh4;
			mat dWhh = dWhh1 + dWhh2 + dWhh3 + dWhh4;
			mat dWhy = dWhy1 + dWhy2 + dWhy3 + dWhy4;
			mat dbh = dbh1 + dbh2 + dbh3 + dbh4;
			mat dby = dby1 + dby2 + dby3 + dby4;

			// update params。把每个场景看做一个样本的话，则是sgd。
			opt->optimize(this->Wxh, this->alpha, dWxh, mdWxh, i);
			opt->optimize(this->Whh, this->alpha, dWhh, mdWhh, i);
			opt->optimize(this->Why, this->alpha, dWhy, mdWhy, i);
			opt->optimize(this->bh, this->alpha, dbh, mdbh, i);
			opt->optimize(this->by, this->alpha, dby, mdby, i);

			// store in vec
			lossAllVec.push_back(loss1(0, 0));
			lossAllVec.push_back(loss2(0, 0));
			lossAllVec.push_back(loss3(0, 0));
			lossAllVec.push_back(loss4(0, 0));
			loss_one_epoch.push_back(loss1(0, 0));
			loss_one_epoch.push_back(loss2(0, 0));
			loss_one_epoch.push_back(loss3(0, 0));
			loss_one_epoch.push_back(loss4(0, 0));
			// print loss
			if (i % 100 == 0)
			{
				cout << "	scenario: " << ascenario->first << endl;
				cout << "				loss: " << loss1(0, 0) << endl;
				cout << "	scenario_next1: " << std::next(ascenario)->first << endl;
				cout << "				loss: " << loss2(0, 0) << endl;
				cout << "	scenario_next2: " << std::next(std::next(ascenario))->first << endl;
				cout << "				loss: " << loss3(0, 0) << endl;
				cout << "	scenario_next3: " << std::next(std::next(std::next(ascenario)))->first << endl;
				cout << "				loss: " << loss4(0, 0) << endl;
			}

		}

*/

		// lossVec mean, accuracy
		double loss_this_epoch = MyLib::mean_vector(loss_one_epoch); // 记录每个epoch的 loss
		double accu_this_epoch = MyLib::mean_vector(true_false); // 记录每个epoch的 accu
		loss_mean_each_epoch.push_back(loss_this_epoch);
		accuracy_each_epoch.push_back(accu_this_epoch);

		if (i % 20 == 0)
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
	log_target_prediction.save("log_target_prediction.txt", file_type::raw_ascii);
	
}

map<string, mat> RNN::lossFun(mat inputs, double score, mat hprev, vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction)
{
	// 注：参数要在这个函数体外部提前初始化。

	// score => targets(format: onehot)
	mat targets = score2onehot(score);

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
		hs[t] = arma::tanh(Wxh * xs[t] + Whh*hs[t-1] + bh);
		// (100, 20)(20,1) + (100,100)(100,1) + (100,1) = (100,1)
		if (t == inputs.n_rows - 1)
		{
			ys[t] = Why * hs[t] + by; 
			// (#classes 3, 100)(100,1) + (3,1) = (3,1)

			// ps[t] = softmax(ys[t])
			mat sum_exp = arma::sum(arma::exp(ys[t]), 0);
			ps[t] = arma::exp(ys[t]) / sum_exp(0, 0); // (3,1)

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
	mat dWxh, dWhh, dWhy, dbh, dby, dhnext;
	dWxh = arma::zeros<mat>(Wxh.n_rows, Wxh.n_cols);
	dWhh = arma::zeros<mat>(Whh.n_rows, Whh.n_cols);
	dWhy = arma::zeros<mat>(Why.n_rows, Why.n_cols);
	dbh = arma::zeros<mat>(bh.n_rows, bh.n_cols);
	dby = arma::zeros<mat>(by.n_rows, by.n_cols);
	dhnext = arma::zeros<mat>(hs[0].n_rows, hs[0].n_cols); // init dhnext = 0

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

			dh = Why.t() * dy + dhnext;
			dhraw = (1 - hs[t] % hs[t]) % dh; // mul elemwise
			dbh += dhraw;
			dWxh += dhraw * xs[t].t(); // 惩罚项，只需要在loop中 加一次
			dWhh += dhraw * hs[t-1].t();
			dhnext = Whh.t() * dhraw;
		}
		else
		{
			dh = dhnext;
			dhraw = (1 - hs[t] % hs[t]) % dh;
			dbh += dhraw;
			dWxh += dhraw * xs[t].t();
			dWhh += dhraw * hs[t - 1].t();
			dhnext = Whh.t() * dhraw;
		}
	}

	clip(dWxh, 5.0, -5.0);
	clip(dWhh, 5.0, -5.0);
	clip(dWhy, 5.0, -5.0);
	clip(dbh, 5.0, -5.0);
	clip(dby, 5.0, -5.0);
	
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
					/ (score_max - score_min +pow(10, -8)); // 分母加上 1e-8, 避免score是 max时的index越界

	//cout << "pos/part: " << pos / part << endl;
	double pos_idx = std::floor(pos / part);
	//cout << "pos_idx: " << pos_idx << endl;

	mat zs = arma::zeros<mat>(n_output_classes, 1);
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
