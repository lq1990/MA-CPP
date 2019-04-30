#include <iostream>
#include <iterator>
#include <thread>
#include <future>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <mat.h>
#include <armadillo>
#include "RNN.h"
#include "AOptimizer.h"
#include "SGD.h"
#include "Adagrad.h"
#include "DemoMultiThread.h"

using namespace arma;
using namespace std;


map<string, MyStruct> read_write()
{
	map<string, MyStruct> myMap;

	const char* path = "C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\MA_Matlab\\Arteon\\start_loadSync\\DataFinalSave\\dataSScaling.mat";
	MATFile* file = matOpen(path, "r");
	mxArray* dataSScaling = matGetVariable(file, "dataSScaling");

	int numFdataS = mxGetNumberOfFields(dataSScaling);
	
	// 遍历 dataSScaling 中每一个场景id
	for (int i = 0; i < numFdataS; i++) {
		const char* scenarioName_char = mxGetFieldNameByNumber(dataSScaling, i);
		string scenarioName = scenarioName_char;
		//cout<< "scenarioName:" << scenarioName << "." << endl;

		mxArray* scenario = mxGetFieldByNumber(dataSScaling, 0, i);
		int numFScenario = mxGetNumberOfFields(scenario);
		//cout << "numFScenario: " << numFScenario <<endl;

		// 获取某一个场景id中的 field
		mxArray* id = mxGetField(scenario, 0, "id");
		mxArray* score = mxGetField(scenario, 0, "score");
		mxArray* EngineSpeed = mxGetField(scenario, 0, "EngineSpeed");
		int rows_es = (int)mxGetM(EngineSpeed);
		int colss_es = (int)mxGetN(EngineSpeed);

		// 所有signal放到一个 matrix中, matrix每一行是 signal随着时间变的值，matrix列是 signals (features)
		arma::mat matData(rows_es, numFScenario-4);
		for (int j = 4; j < numFScenario; j++) {
			const char* fieldName = mxGetFieldNameByNumber(scenario, j);
			//cout << "signal fieldName: " << fieldName << endl;

			mxArray* signal = mxGetFieldByNumber(scenario, 0, j);
			int rows = (int)mxGetM(signal);
			int cols = (int)mxGetN(signal);
			double* data = (double*)mxGetData(signal);
			//cout << fieldName << " data: " << data[0] << endl;

			// 将data保存到matData
			for (int m = 0; m < rows; m++) {
				matData(m, j-4) = data[m];
			}
		}
		//cout << endl;
		//matData.print("matData");

		MyStruct mystruct;
		double* idData = (double*)mxGetData(id);
		double* scoreData = (double*)mxGetData(score);

		mystruct.id = idData[0];
		mystruct.score = scoreData[0];
		mystruct.matData = matData;

		// 把所有场景id中各种数据存到map
		myMap[scenarioName] = mystruct;
	}

	// 去除 id122_start
	string id122 = "id122_start";
	cout << "erase: " << myMap.erase(id122) << ", if return 1, erase id122 successfully" << endl;

	return myMap;
}


/*
	myMap 中存储所有场景的数据: score，matData
*/
void show_myMap()
{
	map<string, MyStruct> myMap = read_write();

	// 使用map，故内部成员乱序排列
	map<string, MyStruct>::iterator it;

	// begin()
	/*it = myMap.begin();
	const char* scenario = it->first;
	MyStruct mystruct = it->second;
	double score = mystruct.score;
	mat matData = mystruct.matData;

	std::cout << "scenario: " << scenario << endl;
	std::cout << "score: " << score << endl;
	std::cout << "matData: \n" << matData << endl;*/

	// 遍历myMap中所有元素
	int count;
	for(it = myMap.begin(), count = 0; it != myMap.end(); it++, count++)
	{
			string scenario = it->first;
			MyStruct mys = it->second;
			std::cout << "scenario1: " << scenario << ", ";
			std::cout << "score1: " << mys.score << endl;

			/*
			auto next_scenario = std::next(it)->first;
			MyStruct next_mys = std::next(it)->second;
			std::cout << "scenario2: " << next_scenario << ", ";
			std::cout << "score2: " << next_mys.score << endl;
			*/

		cout << "		==> count: " << count << endl;
	}

	/*
	for (it = myMap.begin(); it!=myMap.end(); it++) {
		const char* scenario = it->first;
		MyStruct mys = it->second;

		std::cout << "scenario: " << scenario << endl;
		std::cout << "score: " << mys.score << endl;
		// matrix每一行是 signal随着时间变的值，matrix列是 signals (features)
		mat Data = mys.matData;
		//std::cout << "matData: \n" << Data << endl;
		//std::cout << "		matData.n_rows: " << Data.n_rows << endl;
	}*/


}

void train_rnn()
{
	map<string, MyStruct> myMap = read_write(); // get myMap: score & matData

	AOptimizer* opt = NULL; // optimizer
	//opt = new SGD();
	opt = new Adagrad();
	
	RNN rnn = RNN(); 
	rnn.trainMultiThread(myMap, opt, 8);
	// rnn.train(myMap, opt); // train RNN

	// n_h 100, 5, 0.01, epoches 501,	SGD:	 dura 66s, loss 
	// n_h 100, 5, 0.01, epoches 501,	Adagrad: dura 94s, loss 
	// n_h 50, 5, 0.01,	epoches 501,	Adagrad: dura 50s, loss , accu 0.9
	// n_h 50, 5, 0.1, epoches 501,		Adagrad: dura 48.6, loss 0.01, accu 1
	// n_h 50, 5, 0.01, epoches 501,	SGD:	 dura 40, loss 0.045, accu 1
	
	rnn.saveParams(); // save params in txt

	delete opt;
}

void test_rnn()
{
	// load params from txt
	mat Wxh, Whh, Why, bh, by;
	Wxh.load("Wxh.txt", raw_ascii);
	Whh.load("Whh.txt", raw_ascii);
	Why.load("Why.txt", raw_ascii);
	bh.load("bh.txt", raw_ascii);
	by.load("by.txt", raw_ascii);

	// forward 得到由模型算出的每个场景的score
	map<string, MyStruct> myMap = read_write();
	map<string, MyStruct>::iterator it;

	// 在测试中，截取训练数据中一段，来测试得分. 
	// t_begin 应按照场景t长处的比例设置。
	int totalPercent_clip_front = 50;
	mat test_percent_log(totalPercent_clip_front, 3);
	for (int percent = 0; percent < totalPercent_clip_front; percent++)
	{
		std::cout << "percent: " << percent << endl;
		//int t_begin = 20;
		mat loss = arma::zeros<mat>(1, 1);
		map<int, mat> xs, hs, ys, ps;
		RNN rnn = RNN(50, 5, 0.1, 501, 8.9, 6.0);

		// 遍历每个场景
		vector<int> true_false_vec; // 记录所有场景中，模型预测对错
		for (it = myMap.begin(); it != myMap.end(); it++)
		{
			auto scenario = it->first;
			MyStruct sec = it->second;
			double score_target = sec.score;
			mat matData = sec.matData;

			mat targets = rnn.score2onehot(score_target);

			/*cout << "scenario: " << scenario << endl;
			cout << "score: " << score_target << endl;
			cout << "score_target:\n" << targets << endl;*/

			int t_begin = round(percent / 100.0 * matData.n_rows);
			hs[t_begin-1] = arma::zeros<mat>(Whh.n_rows, 1);

			int to_rows = (int)(matData.n_rows / 10*9);
			// 遍历 一个场景的matData的每一行
			for (int t = t_begin; t < to_rows; t++)
			{
				xs[t] = matData.row(t).t();
				hs[t] = arma::tanh(Wxh * xs[t] + Whh * hs[t - 1] + bh);
				if (t == to_rows - 1)
				{
					ys[t] = Why * hs[t] + by;
					mat sum_exp = arma::sum(arma::exp(ys[t]), 0);
					ps[t] = arma::exp(ys[t]) / sum_exp(0, 0); // softmax, probab.
					uvec fuvec = arma::find(targets == 1);
					loss += -log(ps[t](fuvec(0)));

					// 从ps这个 softmax 中找到模型算出的
					uvec idx_max_predict = arma::index_max(ps[t], 0);
					uvec idx_max_target = arma::index_max(targets, 0);
					if (idx_max_predict(0) == idx_max_target(0))
					{
						true_false_vec.push_back(1);
					}
					else
					{
						true_false_vec.push_back(0);
					}
					
					/*cout << "score_predict: " << idx_max(0) << endl;

					cout << "score_predict, ps:\n" << ps[t] << endl;
					cout << "--------------------\n";*/
				}
			}

		}

		// loss mean
		//cout << "距离t=0往右偏移：" << t_begin << endl;
		mat loss_mean = arma::mean(loss);
		//cout << "loss_mean: " << loss_mean(0, 0) << endl;

		// true_false_vec 平均值
		double vec_sum = 0;
		for (int i = 0; i < true_false_vec.size(); i++)
		{
			vec_sum += true_false_vec[i];
		}
		double vec_mean = vec_sum / true_false_vec.size();

		test_percent_log(percent, 0) = percent;
		test_percent_log(percent, 1) = loss_mean(0, 0); // store loss
		test_percent_log(percent, 2) = vec_mean; // store accuracy

	}
	
	test_percent_log.save("test_percent_log.txt", raw_ascii);
}

int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();

	// ------------------------ main code -------------------------------
	//read_write();

	//show_myMap();
	
	train_rnn(); 
	// 单线程 55s，双线程49s, 四线程37s, 
	// 四线程（async）17.6s, 六线程16s，八线程14s，十线程14.2s
	// 注意：电源高性能模式。


	// 使用训练好的参数，对现有场景测试，left_clip 增加到10%, right_clip 一半
	//test_rnn();  

	//DemoMultiThread::demo_multi_thread();


	// ----------------- 测试代码 ---------------------

	// 测试 iterator 的 end。end没有具体数值，只是一个标志位，代表结束。
	/*
	map<string, MyStruct> mp = read_write();
	map<string, MyStruct>::iterator it;
	it = mp.begin();
	cout << it->first << ", " << it->second.score << endl;
	it = mp.end();
	cout << it->first << ", " << it->second.score << endl; // error
	*/

	// 测试 iterator next
	/*
	unordered_map<string, int> mp;
	mp["zbc"] = 123;
	mp["def"] = 946;
	mp["hjk"] = 6748;
	unordered_map<string, int>::iterator it;
	it = mp.begin();
	while (it != mp.end())
	{
		cout << it->first << " : " << it->second << endl;
		it = std::next(it);
	}*/

	// test clip
	/*arma::mat m1(4, 3, fill::randu);
	m1.print("m1");
	RNN rnn = RNN();
	rnn.clip(m1, 0.8, 0.2);
	m1.print("m1 after");
	*/

	/*mat m1 = randu(3, 4);
	m1.print("m1");
	mat m2 = randu(3, 4);
	m2.print("m2");

	mat res = m1 / m2;
	res.print("m1/m2");*/
	
	/*
	mat m0 = arma::randu(3, 4);
	m0.print("m0");
	mat m1 = arma::randu(3, 4);
	m1.print("m1");
	mat m2 = arma::randu(3, 4);
	m2.print("m2");

	// 平方，求和，开根
	mat m_2_sum = m0 % m0 + m1 % m1 + m2 % m2;
	mat m_2_sum_sqrt = arma::sqrt(m_2_sum);

	m_2_sum_sqrt.print("norm ");
	*/
	// norm
	/*mat m1 = randu<mat>(3, 4);
	m1.print("m1");
	double m1norm = arma::norm(m1, 2);
	cout << "m1 norm: " << m1norm << endl; // 最终是一个数
	
	vec v1 = randu<vec>(5);
	v1.print("v1");
	double n1 = arma::norm(v1);
	cout << "n1 norm: " << n1 << endl;

	vec v12 = v1 % v1;
	double v12_sum = arma::sum(v12);
	double v12_sum_sqrt = std::sqrt(v12_sum);
	cout << "my n1 norm: " << v12_sum_sqrt << endl;
	*/

	// 测试 score2onehot，对
	/*RNN rnn = RNN();
	mat scoreMat(10, 30);

	double score = 6.0;
	for (double i = 0; i <= 29; i += 1)
	{
		cout << "score: " << score + i/10 << endl;
		mat onehot = rnn.score2onehot(score+i/10 );
		scoreMat.col(i) = onehot;
		cout << "----------" << endl;
	}
	scoreMat.print();*/
	

	// 使用 MyLib
	/*vector<double> vec;
	vec.push_back(11);
	vec.push_back(12);
	vec.push_back(13.4);

	double m = MyLib::mean_vector(vec);
	cout << "vec mean: " << m << endl;*/

	/*mat m1(3, 1, fill::randu);
	m1.print("m1");
	mat m2 = arma::sum(arma::exp(m1));
	m2.print("m2");

	mat n1 = arma::exp(m1);
	mat res = n1 / m2(0, 0);
	cout << "res: " << res << endl;*/

	/*vector<int> vec;
	vec.push_back(11);
	vec.push_back(112);
	vec.push_back(113);

	cout << vec[0] << endl;*/

	// unordered_map 应该是乱序的。但在实验中，却会有一定的顺序。
	/*unordered_map<int, string> mp;
	mp[11] = "	11 abc";
	mp[1] = "	1 jidsf";
	mp[2] = "	2 jkjkdsf";
	mp[22] = "	22 ujdisajgnf";

	
	unordered_map<int, string>::iterator it;
	for (int i = 0; i < 5; i++)
	{
		for (it = mp.begin(); it != mp.end(); it++)
		{
			cout << it->first << endl;
			cout << it->second << endl;
		}
		cout << "\n-------------------------\n";
	}*/

	/*RNN rnn = RNN();
	mat onehot = rnn.score2onehot(7.2);
	onehot.print("onehot");*/

	/*mat m1(4, 5, fill::randu);
	m1.print("m1");
	RNN rnn = RNN();
	m1 = rnn.clip(m1, 0.8, 0.2);
	m1.print("after clip");*/

	/*mat m1 = randu(4, 5);
	m1.print("m1: ");

	m1.transform([](double val) {
		if (val >= 0.8)
		{
			return 0.8;
		}
		else if(val <=0.2)
		{
			return 0.2;
		}
		else
		{
			return val;
		}
	});
	
	m1.print("m1");*/

	/*mat m1, m2;
	m1 << 0.01 << endr
		<< 0.01 << endr
		<< 0.98 << endr;
	m2 << 0 << endr
		<< 0 << endr
		<< 1 << endr;
	
	uvec f = arma::find(m2 == 1);
	printf("f: %d\n", f(0));

	mat loss = -arma::log(m1 % m2);
	cout << "loss: \n" << loss << endl;
	
	printf("lossVal: %lf\n", loss(f(0)));*/
	
	// ===================================================================

	t_end = clock();
	cout << "\n---------------------\ntime needed: " 
		<< (double)(t_end - t_begin) / CLOCKS_PER_SEC << "s" << endl;

	system("pause");
	return 0;
}