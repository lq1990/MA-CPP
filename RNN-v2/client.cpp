#include <iostream>
#include <armadillo>
#include <mat.h>
#include <map>
#include <vector>
#include "RNN.h"
#include "Adagrad.h"

using namespace std;
using namespace arma;


vector<MyStruct> read_write()
{
	vector<MyStruct> vec;

	const char* path = "C:/Program Files/MATLAB/MATLAB Production Server/R2015a/MA_Matlab/Arteon/start_loadSync/DataFinalSave/listStructTrain.mat";
	MATFile* file = matOpen(path, "r");
	mxArray* listStructTrain = matGetVariable(file, "listStructTrain");

	int numElems = mxGetNumberOfElements(listStructTrain); // 27 scenarios
	int numFields = mxGetNumberOfFields(listStructTrain); // 6 fields: {id, score, details, matData, matDataPcAll, matDataZScore}
	//cout << numElems << ", " << numFields << endl;
	
	// traverse rows, i.e. scenarios
	for (int i = 0; i < numElems; i++)
	{
		MyStruct ms;

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
	vector<MyStruct> vec = read_write();
	cout << "size of vector: " << vec.size() << endl;
	// show first sce
	MyStruct first = vec[0];
	cout << "id: " << first.id << ", score: " << first.score << endl
		<< "dim: " << first.matDataZScore.n_rows << " " << first.matDataZScore.n_cols << endl
		<< "matDataZScore: \n" << first.matDataZScore << endl;
}


void train_rnn()
{
	vector<MyStruct> listStructTrain = read_write();

	AOptimizer* opt = NULL; // optimizer
	opt = new Adagrad(); // opt = new SGD();

	vector<MyStruct>::iterator it; it = listStructTrain.begin(); mat matData = it->matDataZScore;
	int n_features = (int)matData.n_cols;
	MyParams::alpha = 0.1; // learning_rate
	MyParams::total_epoches = 201;
	MyParams::score_max = 8.9;
	MyParams::score_min = 6.0;
	MyParams::n_features = n_features; // 注：若设置参数，必须通过修改MyParams类中静态属性
	MyParams::n_hidden = 50;
	MyParams::n_output_classes = 10;
	MyParams::Wxh = arma::randn(MyParams::n_hidden, MyParams::n_features) * 0.01;
	MyParams::Whh = arma::randn(MyParams::n_hidden, MyParams::n_hidden) * 0.01;
	MyParams::Why = arma::randn(MyParams::n_output_classes, MyParams::n_hidden) * 0.01;
	MyParams::bh = arma::zeros(MyParams::n_hidden, 1);
	MyParams::by = arma::zeros(MyParams::n_output_classes, 1);

	RNN rnn = RNN();
	int n_threads = 8; // 线程数目
	rnn.trainMultiThread(listStructTrain, opt, n_threads); // train RNN

	rnn.saveParams();
	delete opt;
}


int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();
	// =================== main ===============================
	//show_myListStruct();

	train_rnn();
	

	// ======================== test =======================

	// 试验泛型，且打印vector
	/*vector<int> vec;
	vec.push_back(11);
	vec.push_back(21);
	vec.push_back(31);
	MyLib<int>::printVector(vec);*/

	//show_myMap();

	// 试验vector中存储MyStruct
	/*vector<MyStruct> vec;
	MyStruct ms1, ms2, ms3;
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
	cout << "\n---------------------\ntime needed: "
		<< (double)(t_end - t_begin) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;
}