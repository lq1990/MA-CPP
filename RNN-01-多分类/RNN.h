#pragma once
#ifndef   MY_H_RNN       //如果没有定义这个宏  
#define   MY_H_RNN       //定义这个宏  

#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>

using namespace arma;
using namespace std;

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
	RNN(int n_features, int n_hidden, int n_output_classes, 
		double alpha, int totalSteps, 
		double score_max, double score_min);
	
	/* 
		初始化模型参数
	*/
	void initParams();

	/* 
		这是针对一个场景的数据（一个matData加一个score）
	*/
	map<string, mat> lossFun(mat inputs, double score, mat hprev);

	mat clip(mat matrix, double maxVal, double minVal);

	mat score2onehot(double score);

	void train(map<const char*, MyStruct> myMap);

	map<string, arma::mat> getParams();

	/*
		将 params save到本地
	*/
	void saveParams();

private:
	int n_features;
	int n_hidden;
	int n_output_classes;
	double alpha; // learning_rate
	int totalSteps;
	double score_max; 
	double score_min;
	// score_max, score_min 作用：
	// 结合n_output_classes ，将具体的一个场景的score转换为 onehot
	// eg. max: 9.0, min:6.1, 分3份，
	// 则 6.1-7.0: [1,0,0]; 7.1-8.0: [0,1,0]; 8.1-9.0: [0,0,1]; 
	mat Wxh;
	mat Whh;
	mat Why;
	mat bh;
	mat by;
	vector<double> lossVec; // 记录loss
	vector<double> loss_mean_each_epoch;
};

#endif 