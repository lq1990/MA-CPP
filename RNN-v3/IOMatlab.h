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
	For one Scenario.
	Struct stores data about Scenarios from matlab.
	<items>
	float id;
	float score;
	MyArray* matDataZScore;
	</item>
*/
typedef struct SceStruct
{
	float id;
	float score;
	MyArray* matDataZScore;
	// normalized matData by using zscore-norm i.e. (data-mean)/std

	//arma::mat matData;
	//arma::mat matDataZScore;
};

/*
	Input/Output Of Matlab Data
*/
class IOMatlab
{
public:
	IOMatlab();
	~IOMatlab();

	static thrust::device_vector<SceStruct> read(const char* fileName);

	/*
		arma::mat (M, N)
		To 
		array stored in column-major format with dimensions (M*N, 1).

		ld: leading dimension of original m
	*/
	//static MyArray* mat2arr(arma::mat m);
};

