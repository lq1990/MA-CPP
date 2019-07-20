#pragma once
#include <mat.h>
#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "MyArray.h"

using namespace std;
using namespace thrust;


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
	*/
	static void read(const char* fileName, 
		device_vector<float>& sces_id_score,
		device_vector<float>& sces_data,
		device_vector<int>& sces_data_mn,
		device_vector<int>& sces_data_idx_begin);

};

