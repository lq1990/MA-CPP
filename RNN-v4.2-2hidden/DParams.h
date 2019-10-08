#pragma once
#include <iostream>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include "Params.h"

class DParams
{
public:
	DParams();
	~DParams();

	/*
		设置dP为0，根据Params中P的维度
	*/
	void setZeros(Params* p);

public:
	mat dWfSum;
	mat dWiSum;
	mat dWcSum;
	mat dWoSum;
	mat dWhhSum;

	mat dbfSum;
	mat dbiSum;
	mat dbcSum;
	mat dboSum;
	mat dbhhSum;
};

