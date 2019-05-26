#include "MyParams.h"

double	MyParams::alpha = 0.1; // learning_rate
int		MyParams::total_epoches = 301;
double	MyParams::score_max = 8.9;
double	MyParams::score_min = 6.0;
int		MyParams::n_features = 17;
int		MyParams::n_hidden = 50; // predict 时注意修改
int		MyParams::n_output_classes = 10; // predict时注意修改
mat		MyParams::Wxh = arma::randn(n_hidden, n_features) * 0.01;
mat		MyParams::Whh = arma::randn(n_hidden, n_hidden) * 0.01;
mat		MyParams::Why = arma::randn(n_output_classes, n_hidden) * 0.01;
mat		MyParams::bh = arma::zeros(n_hidden, 1);
mat		MyParams::by = arma::zeros(n_output_classes, 1);

MyParams::MyParams()
{
}


MyParams::~MyParams()
{
}

