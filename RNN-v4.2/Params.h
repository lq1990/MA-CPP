#pragma once
#include <iostream>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include <string>

/*
	LSTMģ�Ͳ�����

	ÿ�����㶼��һ��Params��ʵ��

	�ô���������RNN����д̫������
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
	mat Whh; // ��ǰhidden���¸�hidden֮���W������������� Whh=Wy

	mat bf;
	mat bi;
	mat bc;
	mat bo;
	mat bhh;

};

