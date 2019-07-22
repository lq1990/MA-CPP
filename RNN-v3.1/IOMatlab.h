#pragma once
#include <mat.h>
#include <vector>
#include <iostream>

#include "Cuda4RNN.h"

using namespace std;


/*
	Input/Output Of Matlab Data
*/
class IOMatlab
{
public:
	IOMatlab();
	~IOMatlab();

	/*
		fileName:: "listStructTrain", "listStructCV"

		vector<arma::mat> 不能使用，用于存储所有场景数据。
		替代方案：
		4个device_vector<float>
		此4个d_vec 在传进来之前都是size=0

		return numSces
	*/
	static int read(const char* fileName,
		sces_struct* sces_s);

};

