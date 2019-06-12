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
	static mat Wxh1;
	static mat Wh1h1;
	static mat Wh1h2;
	static mat Wh2h2;
	static mat Wh2y;
	static mat bh1;
	static mat bh2;
	static mat by;
};


