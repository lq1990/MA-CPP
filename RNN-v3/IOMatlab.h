#pragma once
#include <mat.h>
#include <armadillo>
#include <vector>
#include <iostream>


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

};

