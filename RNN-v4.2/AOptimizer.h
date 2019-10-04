#pragma once

#include <armadillo>
#include <iostream>
#include <vector>
#include "MyLib.h"

using namespace arma;
using namespace std;

class AOptimizer
{
public:
	AOptimizer() {};
	virtual ~AOptimizer() {};


	/*
		P 是某一个参数.
		并不是所有optimizer都用到最后一个参数。
		Adagrad 中用到 memory。
	*/
	virtual void optimize(mat& P, double alpha, mat dP, mat& mdP, int i) = 0;
};

