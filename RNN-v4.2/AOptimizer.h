#pragma once

#include <armadillo>
#include <iostream>
#include <vector>
#include "MyLib.h"
#include "Params.h"
#include "DParams.h"
#include "MDParams.h"


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
	virtual void optimize(Params* P, double alpha, DParams* dP, MDParams* mdP, int i) = 0;
};

