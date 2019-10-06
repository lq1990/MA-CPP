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
#include "HiddenLayer.h"
#include "Params.h"
#include "DParams.h"
#include "MDParams.h"


using namespace std;
using namespace arma;


typedef struct LossFunctionReturn
{
	mat loss;
	map<string, mat> deltaParamsH1;
	map<string, mat> deltaParamsH2;
};

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

	static void clip(DParams* dP, double maxVal, double minVal);

	/*
		按照 n_output_classes 将score生成 一定长度的 onehot vector
	*/
	static mat score2onehot(double score, int& idx1);

	/*
		核心：前传、反传。
		这是针对一个场景的数据（一个matData加一个score）
		此函数：一个场景计算一次 dW, db。类似sgd，但把一个场景当成一个样本。
		inputs: 某一个场景的matData
		score: 某一个场景的score，即label。
		lambda:
		hprev:
		...

		naive RNN or LSTM 的区别的核心，是在此处修改
	*/
	static LossFunctionReturn lossFun(mat inputs,
		double score, double lambda, mat hprev, mat cprev,
		vector<double>& true_false, vector<double>& log_target, vector<double>& log_prediction);

	
	/*
		若在 train中，实验单线程进行for循环 it+=2，一次计算两个场景，求和dW再update参数。若可行，则可用多线程了。
	*/
	void trainMultiThread(vector<SceStruct> listStructTrain, AOptimizer* opt, int n_threads, double lambda);

	static void predictOneScenario(
		mat Wf, mat Wi, mat Wc, mat Wo, mat Wy,
		mat bf, mat bi, mat bc, mat bo, mat by,
		mat inputs,
		double score, double& loss, int& idx_target, int& idx_prediction);

	/*
		封装前传，可供train test使用，提高代码复用。

		在隐层的基础上 只需要添加从hidden到output的部分
	*/
	void forward(mat inputs, double score);

	map<string, arma::mat> getParams();

	/*
		设置模型参数。
		目的：模型训练后，W b存储以txt格式到本地。在test中，读取txt拿到W b，用其设置RNN参数。
		保存参数到本地，到使用时即test时，读取本地文件，可避免再训练模型耗时。
	*/
	void loadParams();

	/*
		将 params save到本地
	*/
	void saveParams();

	static mat sigmoid(arma::mat mx);

	static mat softmax(arma::mat mx);

public:
	/*
		有多个 static属性，因为当使用多线程并行计算时，用到lossFunMultiThread是static方法，此方法用到的属性必须是静态。
		静态方法不能访问动态属性。
	*/
	static int n_features;
	static int n_hidden;
	static int n_output_classes;
	static double alpha; // learning_rate
	static int total_epoches;
	static double score_max;
	static double score_min;
	// score_max, score_min 作用：
	// 结合n_output_classes ，将具体的一个场景的score转换为 onehot
	// eg. max: 9.0, min:6.1, 分3份，
	// 则 6.1-7.0: [1,0,0]; 7.1-8.0: [0,1,0]; 8.1-9.0: [0,0,1]; 

	//static map<string, mat> model; // model saves params of LSTM
	/*
	static mat Wf; // concat [Whf, Wxf]
	static mat Wi;
	static mat Wc;
	static mat Wo;
	static mat Wy;

	static mat bf;
	static mat bi;
	static mat bc;
	static mat bo;
	static mat by;

	*/
	static int tmp;

	vector<double> lossAllVec; // 记录loss
	vector<double> loss_mean_each_epoch;
	vector<double> accuracy_each_epoch;
	arma::Mat<short> log_target_prediction;
	static std::mutex mtx;


public:
	static Params* ph1; // params of hidden 1
	static Params* ph2;



};

#endif 