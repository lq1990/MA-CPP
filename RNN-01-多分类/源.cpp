#include <iostream>
#include <mat.h>
#include <string>
#include <armadillo>
#include <vector>
#include <map>
#include "RNN.h"

using namespace arma;
using namespace std;

//typedef struct MyStruct
//{
//	double id;
//	double score;
//	arma::mat matData;
//};

map<const char*, MyStruct> read_write()
{
	map<const char*, MyStruct> myMap;

	const char* path = "C:\\Program Files\\MATLAB\\MATLAB Production Server\\R2015a\\MA_Matlab\\Arteon\\start_loadSync\\DataFinalSave\\dataSScaling.mat";
	MATFile* file = matOpen(path, "r");
	mxArray* dataSScaling = matGetVariable(file, "dataSScaling");

	int numFdataS = mxGetNumberOfFields(dataSScaling);

	// 遍历 dataSScaling 中每一个场景id
	for (int i = 0; i < numFdataS; i++) {
		const char* scenarioName = mxGetFieldNameByNumber(dataSScaling, i);
		//cout<< "scenarioName: " << scenarioName << endl;
		mxArray* scenario = mxGetFieldByNumber(dataSScaling, 0, i);
		int numFScenario = mxGetNumberOfFields(scenario);
		//cout << "numFScenario: " << numFScenario <<endl;

		// 获取某一个场景id中的 field
		mxArray* id = mxGetField(scenario, 0, "id");
		mxArray* score = mxGetField(scenario, 0, "score");
		mxArray* EngineSpeed = mxGetField(scenario, 0, "EngineSpeed");
		int rows_es = mxGetM(EngineSpeed);
		int colss_es = mxGetN(EngineSpeed);

		// 所有signal放到一个 matrix中, matrix每一行是 signal随着时间变的值，matrix列是 signals (features)
		arma::mat matData(rows_es, numFScenario-4);
		for (int j = 4; j < numFScenario; j++) {
			const char* fieldName = mxGetFieldNameByNumber(scenario, j);
			//cout << "fieldName: " << fieldName << endl;

			mxArray* signal = mxGetFieldByNumber(scenario, 0, j);
			int rows = mxGetM(signal);
			int cols = mxGetN(signal);
			double* data = (double*)mxGetData(signal);
			//cout << fieldName << " data: " << data[0] << endl;

			// 将data保存到matData
			for (int m = 0; m < rows; m++) {
				matData(m, j-4) = data[m];
			}
		}

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

	return myMap;
}


void run()
{
	map<const char*, MyStruct> myMap = read_write();
	// 使用map，故内部成员乱序排列
	map<const char*, MyStruct>::iterator it;

	it = myMap.begin();
	const char* scenario = it->first;
	MyStruct mystruct = it->second;
	double score = mystruct.score;
	mat matData = mystruct.matData;

	cout << "scenario: " << scenario << endl;
	cout << "score: " << score << endl;
	cout << "matData: \n" << matData << endl;

	// 遍历myMap中所有元素
	/*
	for (it = myMap.begin(); it!=myMap.end(); it++) {
		const char* scenario = it->first;
		MyStruct mys = it->second;

		cout << "scenario: " << scenario << endl;
		cout << "score: " << mys.score << endl;
		// matrix每一行是 signal随着时间变的值，matrix列是 signals (features)
		mat Data = mys.matData;
		cout << "matData: \n" << Data << endl;
	}
	*/

}


int main()
{
	clock_t t_begin, t_end;
	t_begin = clock();

	// ------------------------ main code -------------------------------

	//run();

	cout << "abc" << endl;
	cout << "	123" << endl;

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