#pragma once
#include <iostream>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include <string>

/*
	LSTM模型参数类

	每个隐层都有一个Params类实例

	好处：避免在RNN类中写太多遍参数
*/
class Params
{
public:
	Params(int n_input_dim, int n_hidden_cur, int n_hidden_next);
	~Params();

	void save(string title);

	void load(string title);

public:
	mat Wf;
	mat Wi;
	mat Wc;
	mat Wo;
	mat Whh; // 当前hidden与下个hidden之间的W，对于最后隐层 Whh=Wy

	mat bf;
	mat bi;
	mat bc;
	mat bo;
	mat bhh;

};

