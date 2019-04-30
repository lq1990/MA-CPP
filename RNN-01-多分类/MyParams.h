#pragma once

#include <armadillo>

using namespace std;
using namespace arma;

class MyParams
{
public:
	MyParams();
	~MyParams();
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


//double RNN::score_max = 8.9;
//double RNN::score_min = 6.0;
//int RNN::n_features = 17;
//int RNN::n_hidden = 50;
//int RNN::n_output_classes = 5;
//mat RNN::Wxh = arma::randn(n_hidden, n_features) * 0.01;
//mat RNN::Whh = arma::randn(n_hidden, n_hidden) * 0.01;
//mat RNN::Why = arma::randn(n_output_classes, n_hidden) * 0.01;
//mat RNN::bh = arma::zeros(n_hidden, 1);
//mat RNN::by = arma::zeros(n_output_classes, 1);
