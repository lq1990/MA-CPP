#pragma once

#include <armadillo>

using namespace std;
using namespace arma;

class MyParams
{
public:
	MyParams();
	~MyParams();
	static double alpha;
	static int total_epoches;
	static double score_max;
	static double score_min;
	static int n_features;
	static int n_hidden;
	static int n_output_classes;
	static mat Wxh;
	static mat Whh;
	static mat Why;
	static mat bh;
	static mat by;
};


