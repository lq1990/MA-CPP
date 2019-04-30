#pragma once
#ifndef   RNN_H       //如果没有定义这个宏  
#define   RNN_H       //定义这个宏  

#include "MyLib.h"
#include <iterator>
#include <thread>
#include <mutex>
#include <future>
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include "AOptimizer.h"


using namespace std;
using namespace arma;

typedef struct MyStruct
{
	double id;
	double score;
	arma::mat matData;
};

class RNN
{
public:
	RNN();
	~RNN();

	/*
		n_out_classes: 将回归问题转化为分类问题，类别数目可先设置少些，再慢慢增加。可更好理解特征与参数关系。
	*/
	RNN(int n_hidden, int n_output_classes, 
		double alpha, int total_steps, 
		double score_max, double score_min);
	
	/* 
		初始化模型参数
	*/
	void initParams(map<string, MyStruct> myMap);

	/* 
		核心：前传、反传。
		这是针对一个场景的数据（一个matData加一个score）
		此函数：一个场景计算一次 dW, db。类似sgd，但把一个场景当成一个样本。
		inputs: 某一个场景的matData
		score: 某一个场景的score，即label。
	*/

	static void clip(mat& matrix, double maxVal, double minVal);

	static mat score2onehot(double score);

	map<string, mat> lossFun(mat inputs, double score, mat hprev, vector<double>& true_false);

	/*
		静态方法。供给多线程train
	*/
	static map<string, mat> lossFunMultiThread(mat inputs, double score, mat hprev, vector<double>& true_false);

	/*
		train params of rnn-model.
		myMap: 存储所有场景score和matData的map。
	*/
	void train(map<string, MyStruct> myMap, AOptimizer* opt); // myMap: scenarios: id, score, matData

	/*
		若在 train中，实验单线程进行for循环 it+=2，一次计算两个场景，求和dW再update参数。若可行，则可用多线程了。
	*/
	void trainMultiThread(map<string, MyStruct> myMap, AOptimizer* opt, int n_threads);

	void test();

	/*
		封装前传，可供train test使用，提高代码复用。
	*/
	void forwardProp();

	map<string, arma::mat> getParams();

	/*
		设置模型参数。
		目的：模型训练后，W b存储以txt格式到本地。在test中，读取txt拿到W b，用其设置RNN参数。
		保存参数到本地，到使用时即test时，读取本地文件，可避免再训练模型耗时。
	*/
	void setParams(mat Wxh, mat Whh, mat Why, mat bh, mat by);

	/*
		将 params save到本地
	*/
	void saveParams();

private:
	/*
		有多个 static属性，因为当使用多线程并行计算时，用到lossFunMultiThread是static方法，此方法用到的属性必须是静态。
		静态方法不能访问动态属性。
	*/
	static int n_features;
	static int n_hidden;
	static int n_output_classes;
	double alpha; // learning_rate
	int total_epoches;
	static double score_max; 
	static double score_min;
	// score_max, score_min 作用：
	// 结合n_output_classes ，将具体的一个场景的score转换为 onehot
	// eg. max: 9.0, min:6.1, 分3份，
	// 则 6.1-7.0: [1,0,0]; 7.1-8.0: [0,1,0]; 8.1-9.0: [0,0,1]; 
	static mat Wxh;
	static mat Whh;
	static mat Why;
	static mat bh;
	static mat by;
	vector<double> lossAllVec; // 记录loss
	vector<double> loss_mean_each_epoch;
	vector<double> accuracy_each_epoch;
	static std::mutex mtx;
};

#endif 