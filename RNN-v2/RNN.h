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
#include "MyParams.h"


using namespace std;
using namespace arma;

/*
	SceStruct stores infos of Scenarios
*/
typedef struct SceStruct
{
	double id;
	double score;
	arma::mat matData;
	arma::mat matDataZScore; // normalized matData by using zscore-norm i.e. (data-mean)/std
};

class RNN
{
public:
	RNN();
	~RNN();

	static void clip(mat& matrix, double maxVal, double minVal);

	/*
		按照 n_output_classes 将score生成 一定长度的 onehot vector
	*/
	static mat score2onehot(double score);

	/*
		核心：前传、反传。
		这是针对一个场景的数据（一个matData加一个score）
		此函数：一个场景计算一次 dW, db。类似sgd，但把一个场景当成一个样本。
		inputs: 某一个场景的matData
		score: 某一个场景的score，即label。
	*/
	static map<string, mat> lossFun(mat inputs, double score, double lambda, mat h1prev, mat h2prev, vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction);

	
	/*
		若在 train中，实验单线程进行for循环 it+=2，一次计算两个场景，求和dW再update参数。若可行，则可用多线程了。
	*/
	void trainMultiThread(vector<SceStruct> listStructTrain, AOptimizer* opt, int n_threads, double lambda);

	static void predictOneScenario(mat Wxh, mat Whh, mat Why, mat bh, mat by, mat inputs, double score, double& loss, int& idx_target, int& idx_prediction);

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
	void setParams(mat Wxh1, mat Wh1h1, mat Wh1h2, mat Wh2h2, mat Wh2y, mat bh1, mat bh2, mat by);

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
	static mat Wxh1;
	static mat Wh1h1;
	static mat Wh1h2;
	static mat Wh2h2;
	static mat Wh2y;
	static mat bh1;
	static mat bh2;
	static mat by;
	vector<double> lossAllVec; // 记录loss
	vector<double> loss_mean_each_epoch;
	vector<double> accuracy_each_epoch;
	arma::Mat<short> log_target_prediction;
	static std::mutex mtx;
};

#endif 