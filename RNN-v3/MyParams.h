#pragma once
#include "MyArray.h"

/*
	set params of Model
*/

class MyParams
{
public:
	MyParams();
	~MyParams();

	static int n_features; // num of columns os matData
	static int n_hidden;	// num of hidden neurons
	static int n_output_classes;	// num of predicted classes of Scenarios
	static float score_min;
	static float score_max;

	static MyArray* Wxh;
	static MyArray* Whh;
	static MyArray* Why;
	static MyArray* bh;
	static MyArray* by;
};

