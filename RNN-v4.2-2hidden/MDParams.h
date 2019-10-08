#pragma once
#include <iostream>
#include "MyLib.h"
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include "Params.h"

/*
	memory of dP
*/
class MDParams
{
public:
	MDParams();
	~MDParams();

	/*
	����dPΪ0������Params��P��ά��
*/
	void setZeros(Params* p);

public:
	mat mdWfSum;
	mat mdWiSum;
	mat mdWcSum;
	mat mdWoSum;
	mat mdWhhSum;

	mat mdbfSum;
	mat mdbiSum;
	mat mdbcSum;
	mat mdboSum;
	mat mdbhhSum;
};

