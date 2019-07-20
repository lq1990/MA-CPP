#include "MyParams.h"

float	MyParams::alpha = 0.1;
int		MyParams::total_epoches = 501;

int		MyParams::n_features = 17; // num of columns os matData
int		MyParams::n_hidden = 50;	// num of hidden neurons
int		MyParams::n_output_classes = 10;	// num of predicted classes of Scenarios

float	MyParams::score_min = 6.0f;
float	MyParams::score_max = 8.9f;



MyParams::MyParams()
{
}


MyParams::~MyParams()
{
}
