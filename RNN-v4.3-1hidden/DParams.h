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
		����dPΪ0������Params��P��ά��
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

