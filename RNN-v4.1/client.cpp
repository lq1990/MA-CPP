#include <iostream>
#include <armadillo>
#include <mat.h>
#include <map>
#include <vector>
#include "RNN.h"
#include "Adagrad.h"
#include "MyLib.h"
#include <string>
#include <sstream>

const double INFI = 0xffffffff >> 1;

using namespace std;
using namespace arma;

typedef struct ParamStruct
{
	double lambda; // 惩罚系数
	map<string, arma::mat> paramsMap; // 对应此 lambda 的模型参数
	double cvLossMean; // 使用此处参数，得到的CV数据集所有场景loss mean
};

vector<ParamStruct> vecParamStruct; // 存储不同lambda时的对应的数据
// 目的：通过调节lambda，模型参数会变，找到使得CV数据集loss最小时，对应的模型参数


vector<SceStruct> read_write(const char* fileName)
{
	vector<SceStruct> vec;

	const char* dir = "C:/Program Files/MATLAB/MATLAB Production Server/R2015a/MA_Matlab/Arteon/start_loadSync/DataFinalSave/list_data/";
	const char* tail = ".mat";
	char path[1000];
	strcpy_s(path, dir);
	strcat_s(path, fileName);
	strcat_s(path, tail);

	MATFile* file = matOpen(path, "r");
	mxArray* listStructTrain = matGetVariable(file, fileName);

	int numElems = mxGetNumberOfElements(listStructTrain); // 27 scenarios
	int numFields = mxGetNumberOfFields(listStructTrain); // 6 fields: {id, score, details, matData, matDataPcAll, matDataZScore}
	//cout << numElems << ", " << numFields << endl;
	
	// traverse rows, i.e. scenarios
	for (int i = 0; i < numElems; i++)
	{
		SceStruct ms;

		mxArray* item_id = mxGetField(listStructTrain, i, "id");
		mxArray* item_score = mxGetField(listStructTrain, i, "score");
		mxArray* item_matData = mxGetField(listStructTrain, i, "matData");
		mxArray* item_matDataZScore = mxGetField(listStructTrain, i, "matDataZScore");

		double* id = (double*)mxGetData(item_id);
		double* score = (double*)mxGetData(item_score);
		double* matData = (double*)mxGetData(item_matData);
		int rows_matData = (int)mxGetM(item_matData); // #rows of matData 
		int cols_matData = (int)mxGetN(item_matData); // #cols of matData
		double* matDataZScore = (double*)mxGetData(item_matDataZScore);
		int rows_matDataZScore = (int)mxGetM(item_matDataZScore);
		int cols_matDataZScore = (int)mxGetN(item_matDataZScore);

		mat matTmp = mat(rows_matDataZScore*cols_matDataZScore, 1);
		for (int i = 0; i < rows_matDataZScore*cols_matDataZScore; i++) // 把matData中数据一列列读取
		{
			matTmp(i, 0) = matDataZScore[i];
		}
		matTmp.reshape(rows_matDataZScore, cols_matDataZScore);

		ms.id = id[0];
		ms.score = score[0];
		ms.matDataZScore = matTmp;

		// save ms in vector
		vec.push_back(ms);

		//if (i == 0)
		//{
		//cout << "id: " << id[0] << ", " 
		//	<<"score: "<< score[0] << ", " 
		//	<< "matData dim: " << rows_matData << " " << cols_matData << ", "
		//	<< "matDataZScore dim: " << rows_matDataZScore << " " << cols_matDataZScore
		//	<< endl;

		//	// print matData
		//	for (int i = 0; i < rows_matDataZScore*cols_matDataZScore; i++) // 把matData中数据一列列读取
		//	{
		//		matTmp(i, 0) = matDataZScore[i];
		//	}

		//	matTmp.reshape(rows_matDataZScore, cols_matDataZScore);
		//	
		//}
	}

	return vec;
}

void show_myListStruct()
{
	vector<SceStruct> vec = read_write("listStructTrain");
	std::cout << "size of vector: " << vec.size() << endl;
	// show first sce
	SceStruct first = vec[0];
	std::cout << "id: " << first.id << ", score: " << first.score << endl
		<< "dim: " << first.matDataZScore.n_rows << " " << first.matDataZScore.n_cols << endl
		<< "matDataZScore: \n" << first.matDataZScore << endl;
}

/*
	W b 经过CV验证，已经最优
	load W b to predict listStructTest
*/
void loadWbToPredictListStruct(const char* fileName)
{
	//const char* fileName = "listStructTest";

	// 注意：在predict之前，确定下 MyParams 中参数是匹配的

	//// read W b
	mat Wf, Wi, Wc, Wo, Wy;
	mat bf, bi, bc, bo, by;
	Wf.load("Wf.txt", raw_ascii);
	Wi.load("Wi.txt", raw_ascii);
	Wc.load("Wc.txt", raw_ascii);
	Wo.load("Wo.txt", raw_ascii);
	Wy.load("Wy.txt", raw_ascii);
	bf.load("bf.txt", raw_ascii);
	bi.load("bi.txt", raw_ascii);
	bc.load("bc.txt", raw_ascii);
	bo.load("bo.txt", raw_ascii);
	by.load("by.txt", raw_ascii);

	vector<SceStruct> listStruct = read_write(fileName);
	vector<double> true_false;

	vector<double> vecLossMean; // 存储所有场景预测值的loss mean
	for (int i = 0; i < listStruct.size(); i++) // 遍历所有场景
	{
		SceStruct first = listStruct[i];

		double loss;
		int idx_target, idx_pred;
		RNN::predictOneScenario(Wf, Wi, Wc, Wo, Wy,
			bf, bi, bc, bo, by,
			first.matDataZScore,
			first.score, loss, idx_target, idx_pred);

		std::cout << fileName << " predit scenario id: \t" << first.id << "\t"
			<< "target score:  " << first.score << "\t"
			<< "idx target - prediction: " << idx_target << " - " << idx_pred << "\t"
			<< "loss: " << loss 
			<< (idx_target==idx_pred ? " true " : " ---false ")
			<< endl;

		vecLossMean.push_back(loss);
		true_false.push_back((idx_target==idx_pred ? 1 : 0));
	}

	double lossMean = MyLib<double>::mean_vector(vecLossMean);
	double accu = MyLib<double>::mean_vector(true_false);
	std::cout << "loss mean: " << lossMean << "\t accu: " << accu << endl;

}

/*
	输出 listStruct CV/Train loss mean
*/
void calcPredLossMeanAccu(map<string, arma::mat> mp, 
	const char* fileName, double& lossMean, double& accu)
{
	// 注意：在predict之前，确定下 MyParams 中参数是匹配的

	mat Wf, Wi, Wc, Wo, Wy,
		bf, bi, bc, bo, by;
	Wf = mp["Wf"];
	Wi = mp["Wi"];
	Wc = mp["Wc"];
	Wo = mp["Wo"];
	Wy = mp["Wy"];
	bf = mp["bf"];
	bi = mp["bi"];
	bc = mp["bc"];
	bo = mp["bo"];
	by = mp["by"];

	vector<SceStruct> listStruct = read_write(fileName);
	vector<double> true_false;

	vector<double> vecLossMean; // 存储所有场景预测值的loss mean
	for (int i = 0; i < listStruct.size(); i++) // 遍历所有场景
	{
		SceStruct first = listStruct[i];

		double loss;
		int idx_target, idx_pred;
		RNN::predictOneScenario(Wf, Wi, Wc, Wo, Wy,
			bf, bi, bc, bo, by,
			first.matDataZScore,
			first.score, loss, idx_target, idx_pred);

		std::cout << fileName << " predit scenario id: \t" << first.id << "\t"
			<< "target score:  " << first.score << "\t"
			<< "idx target - prediction: " << idx_target << " - " << idx_pred << "\t"
			<< "loss: " << loss
			<< (idx_target == idx_pred ? " true " : " ---false ")
			<< endl;

		vecLossMean.push_back(loss);
		true_false.push_back((idx_target == idx_pred ? 1 : 0));
	}

	lossMean = MyLib<double>::mean_vector(vecLossMean);
	accu = MyLib<double>::mean_vector(true_false);
	std::cout << "loss mean: " << lossMean << "\t accu: " << accu << endl;
}

/*
	目的：通过 train/cv dataset寻找 optimal lambda
*/
void train_rnn()
{
	vector<SceStruct> listStructTrain = read_write("listStructTrain");

	AOptimizer* opt = NULL; // optimizer
	opt = new Adagrad(); // opt = new SGD();

	vector<SceStruct>::iterator it; it = listStructTrain.begin(); mat matData = it->matDataZScore;
	int n_features = (int)matData.n_cols;
	RNN::n_features = n_features;
	//MyParams::alpha = 0.1; // learning_rate
	//MyParams::total_epoches = 501;
	//MyParams::score_max = 8.9;
	//MyParams::score_min = 6.0;
	//MyParams::n_features = n_features; // 注：若设置参数，必须通过修改MyParams类中静态属性
	//MyParams::n_hidden = 50;
	//MyParams::n_output_classes = 10;
	//MyParams::Wxh = arma::randn(MyParams::n_hidden, MyParams::n_features) * 0.01;
	//MyParams::Whh = arma::randn(MyParams::n_hidden, MyParams::n_hidden) * 0.01;
	//MyParams::Why = arma::randn(MyParams::n_output_classes, MyParams::n_hidden) * 0.01;
	//MyParams::bh = arma::zeros(MyParams::n_hidden, 1);
	//MyParams::by = arma::zeros(MyParams::n_output_classes, 1);

	RNN rnn = RNN();
	int n_threads = 8; // 线程数目
	double globalCVMinLossMean = INFI;
	double globalCVMaxAccu = -INFI;

	double maxLambda = 0.6000001;
	double intervalLambda = 0.1;
	mat matLambdaLossMeanAccu_CV_Train(maxLambda / intervalLambda + 1, 5, fill::zeros); // col1: lambda，col2: cvLossMean, col3: trainLossMean，col4: cvAccu, col5: trainAccu
	for (double lambda = 0, i = 0; lambda < maxLambda; lambda+= intervalLambda, i++)
	{
		// 注：第n次训练出来的参数，到了第n+1次会被作为init初值使用。
		// 优点：会提高第n+1次训练的收敛速度。
		// 缺点：有可能陷入局部最优。
		
		// 1. 改变lambda，会得到不同的参数
		rnn.trainMultiThread(listStructTrain, opt, n_threads, lambda); // train RNN
	
		// 2. 再用参数对CV数据集预测，得到lossmean，存起来
		map<string, arma::mat> mp = rnn.getParams();

		double cvLossMean, cvAccu;
		calcPredLossMeanAccu(mp, "listStructCV", cvLossMean, cvAccu); // 计算 CV数据集的 lossMean
		double trainLossMean, trainAccu;
		calcPredLossMeanAccu(mp, "listStructTrain", trainLossMean, trainAccu); // 计算 Train数据集的 lossMean

		std::cout << "lambda: " << lambda << endl
			<< "DataSet\t" << "lossMean" << "\t" << "Accu" << endl
			<< "CV     \t" << cvLossMean << "\t\t" << cvAccu << endl
			<< "Train  \t" << trainLossMean << "\t\t" << trainAccu << endl;
		matLambdaLossMeanAccu_CV_Train(i, 0) = lambda;
		matLambdaLossMeanAccu_CV_Train(i, 1) = cvLossMean;
		matLambdaLossMeanAccu_CV_Train(i, 2) = trainLossMean;
		matLambdaLossMeanAccu_CV_Train(i, 3) = cvAccu;
		matLambdaLossMeanAccu_CV_Train(i, 4) = trainAccu;
		// 3. 最后，找到 最大accu 或 最小lossmean 对应的参数，以最大accu为主
		if (cvAccu > globalCVMaxAccu || (cvAccu == globalCVMaxAccu && cvLossMean < globalCVMinLossMean))
		{
			rnn.saveParams();
			std::cout << "----------------------------------------------" << endl;
			std::cout << "saveParams, when lambda: " << lambda << ", cvAccu: " << cvAccu << ", cvLossMean: " << cvLossMean << endl;
			std::cout << "----------------------------------------------" << endl;
			
			globalCVMaxAccu = cvAccu;
			globalCVMinLossMean = cvLossMean;
		}

		std::cout << "======================================================================================" << endl << endl;

		matLambdaLossMeanAccu_CV_Train.save("matLambdaLossMeanAccu_CV_Train.txt", file_type::raw_ascii);
	}

	delete opt;
}

void train_rnn_withALambda(const char* fileName, double lambda)
{
	vector<SceStruct> listStruct;
	if (fileName == "listStructTrainCV")
	{
		vector<SceStruct> listStructTrain = read_write("listStructTrain");
		vector<SceStruct> listStructCV = read_write("listStructCV");

		for (int i = 0; i < listStructTrain.size(); i++)
		{
			listStruct.push_back(listStructTrain[i]);
		}
		for (int i = 0; i < listStructCV.size(); i++)
		{
			listStruct.push_back(listStructCV[i]);
		}
	}
	else
	{
		listStruct = read_write(fileName);
	}


	AOptimizer* opt = NULL; // optimizer
	opt = new Adagrad(); // opt = new SGD();

	vector<SceStruct>::iterator it; 
	it = listStruct.begin(); mat matData = it->matDataZScore;
	int n_features = (int)matData.n_cols;
	RNN::n_features = n_features;

	RNN rnn = RNN();

	std::cout
		<< "alpha: " << RNN::alpha << ", "
		<< "total_epoches: " << RNN::total_epoches << ", "
		<< "n_features: " << RNN::n_features << ", "
		<< "n_hidden: " << RNN::n_hidden << ", "
		<< "n_output_classes: " << RNN::n_output_classes << ", "
		<< "score_min: " << RNN::score_min  << ", "
		<< "score_max: " << RNN::score_max
		<< endl;

	int n_threads = 8; // 线程数目

	rnn.trainMultiThread(listStruct, opt, n_threads, lambda); // train RNN
	rnn.saveParams();
	
	/*
						log of time:
		epoches: 200	219.9s

	*/

	delete opt;
}

int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();

	// =================== main ===============================

	//show_myListStruct();


	//train_rnn();

	// 下一步： 训练超参数lambda

	

	// -----------------------------

	double optLambda = 0.1; // LSTM: 

	//train_rnn_withALambda("listStructTrain", optLambda);

	loadWbToPredictListStruct("listStructTrain"); std::cout << endl;
	loadWbToPredictListStruct("listStructCV"); std::cout << endl;
	loadWbToPredictListStruct("listStructTest");

	/*
		lambda					train/cv/test
		0.1, use Train,		0.95 / 0.43 / 0.286
		
	*/
	
	// epoches: 31, threads: 8  => dt 42.7s
	// epoches: 21, threads: 1  => dt 73.0s

	// ======================== try =======================

	// LSTM
	/*LSTM lstm = LSTM();
	cout
		<< "H: " << LSTM::H << ", "
		<< "D: " << LSTM::D << ", "
		<< "Z: " << LSTM::Z << endl;

	mat state;
	lstm.lstm_forward(2, state);*/


	// softmax
	/*arma::mat m1(4, 1, fill::randn);
	m1.print("m1");

	mat m2 = RNN::softmax(m1);
	m2.print("softmax m1");*/

	// sigmoid
	/*arma::mat m1(4, 3, fill::randn);
	m1.print("m1");

	mat m2 = RNN::sigmoid(m1);
	m2.print("sigmoid m1");*/

	// score2onehot
	/*RNN rnn = RNN();

	for (double i = 6.0; i < 8.90000001; i += 0.1)
	{
		mat onehot = rnn.score2onehot(i);

		stringstream ss;
		ss << "i: " << i;
		onehot.t().print(ss.str());
	}*/

	// 比较char
	/*const char* fileName = "listStructTrainCV";
	if (fileName == "listStructTrainCV")
	{
		cout << "same" << endl;
	}
	else
	{
		cout << "not same" << endl;
	}*/

	// 试验 mat的应该高
	/*double maxLambda = 0.5000001;
	double intervalLambda = 0.01;
	mat m1(maxLambda/intervalLambda + 1, 1, fill::zeros);
	for (double lambda = 0, i = 0; lambda < maxLambda; lambda += intervalLambda, i++)
	{
		cout << "lambda: " << lambda << ",\t i: " << i << endl;
		m1(i, 0) = i;
	}

	m1.print("m1");*/

	// 试验打印格式
	//cout << "lambda: " << 1.5 << endl
	//	<< "DataSet\t" <<"lossMean" << "\t" << "Accu" << endl
	//	<< "CV     \t" << 1.0  << "\t\t" << 0.1 << endl
	//	<< "Train  \t" << 1.0  << "\t\t" << 0.1 << endl;


	/*for (double lambda = 0.0; lambda <= 0.600000001; lambda += 0.2)
	{
		cout << lambda << endl;
	}*/

	// 测试输出格式
	//printf("%lld", INFI);
	//cout << (INFI) << endl;
	//cout << (INFI - 10) << endl;

	// 移位运算符，需要左移 无符号的数，否则有可能会变成负的
 	/*unsigned int n = 1;
	unsigned int n2 = (n << 31);
	cout << n2 << endl;*/

	// 试验mat 乘以一个数
	/*mat m1(3, 2, fill::randn);
	m1.print();

	mat m2 = m1 * 10;
	m2.print();*/

	// char 拼接
	//const char* c1 = "abc";
	//const char* c2 = "def";
	//char c3[20];

	//strcpy_s(c3, c1);
	//strcat_s(c3, c2);

	//cout << c3 << endl;

	// 试验泛型，且打印vector
	/*vector<int> vec;
	vec.push_back(11);
	vec.push_back(21);
	vec.push_back(31);
	MyLib<int>::printVector(vec);*/

	//show_myMap();

	// 试验vector中存储SceStruct
	/*vector<SceStruct> vec;
	SceStruct ms1, ms2, ms3;
	ms1.id = 101;
	ms1.score = 100;
	ms2.id = 102;
	ms2.score = 99;
	ms3.id = 103;
	ms3.score = 98;

	vec.push_back(ms1);
	vec.push_back(ms2);
	vec.push_back(ms3);

	for (int i = 0; i < vec.size(); i++)
	{
		cout << vec[i].id << ", " << vec[i].score << endl;
	}*/

	// ===========================================================

	t_end = clock();
	std::cout << "---------------------------------\ntime needed: "
		<< (double)(t_end - t_begin) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;
}