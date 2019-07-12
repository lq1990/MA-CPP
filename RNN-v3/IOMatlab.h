#pragma once
#include <mat.h>
#include <armadillo>
#include <vector>
#include <iostream>

#include "MyStruct.h"

using namespace arma;
using namespace std;

/*
	Struct stores data about Scenarios from matlab
*/
typedef struct SceStruct
{
	float id;
	float score;
	arma::mat matData;
	arma::mat matDataZScore;
	// normalized matData by using zscore-norm i.e. (data-mean)/std
};

/*
	Input/Output Of Matlab Data
*/
class IOMatlab
{
public:
	IOMatlab();
	~IOMatlab();

	static vector<SceStruct> read(const char* fileName);

	/*
		arma::mat (M, N)
		To 
		array stored in column-major format with dimensions (M*N, 1).

		ld: leading dimension of original m
	*/
	static MyArray mat2arr(arma::mat m);
};

